import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from tqdm import tqdm

from bindsnet.datasets import MNIST
from bindsnet.encoding import PoissonEncoder
from bindsnet.network import Network
from bindsnet.network.topology_features import Weight
from bindsnet.learning.MCC_learning import PostPre
from bindsnet.network.monitors import Monitor
from bindsnet.network.nodes import Input, LIFNodes
from bindsnet.network.topology import MulticompartmentConnection

# Checks if the script is running on SLURM
job_id = os.environ.get("SLURM_JOB_ID", "local")
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--n_neurons", type=int, default=100)
parser.add_argument("--n_epochs", type=int, default=1)
parser.add_argument("--examples", type=int, default=10000)  
parser.add_argument("--n_workers", type=int, default=0)
parser.add_argument("--time", type=int, default=250)
parser.add_argument("--dt", type=float, default=1.0)
parser.add_argument("--intensity", type=float, default=64)
parser.add_argument("--progress_interval", type=int, default=10)
parser.add_argument("--update_interval", type=int, default=250)
parser.add_argument("--block_size", type=int, default=1000, help="Images per evaluation/saving block")
parser.add_argument("--gpu", dest="gpu", action="store_true")
parser.set_defaults(gpu=False)

args = parser.parse_args()

seed = args.seed
n_neurons = args.n_neurons
n_epochs = args.n_epochs
examples = args.examples
n_workers = args.n_workers
time = args.time
dt = args.dt
intensity = args.intensity
progress_interval = args.progress_interval
update_interval = args.update_interval
block_size = args.block_size
gpu = args.gpu

np.random.seed(seed)
torch.cuda.manual_seed_all(seed)
torch.manual_seed(seed)

# Sets up GPU use
device = torch.device("cuda" if torch.cuda.is_available() and gpu else "cpu")
if device.type == "cuda":
    torch.cuda.manual_seed_all(seed)
else:
    gpu = False

# Respect SLURM CPU allocation if available
slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
if slurm_cpus is not None:
    torch.set_num_threads(int(slurm_cpus))
else:
    torch.set_num_threads(max(1, os.cpu_count() - 1))

print("Running on Device =", device)

# Build network
network = Network(dt=dt)

# Input layer - 784 neurons
inpt = Input(784, shape=(1, 28, 28), traces=True)
network.add_layer(inpt, name="I")

# Reservoir layer
reservoir = LIFNodes(n_neurons, traces=True, thresh=-52 + np.random.randn(n_neurons).astype(float))
network.add_layer(reservoir, name="R")

# Output layer - 10 neurons (representing digits 0-9)
output = LIFNodes(10, traces=True, thresh=-70)
network.add_layer(output, name="O")

# Input -> Reservoir connection
C1_w = 0.5 * torch.randn(inpt.n, reservoir.n)
IR_weight_feature = Weight(name="IRweight", value=C1_w)
C1 = MulticompartmentConnection(source=inpt, target=reservoir, device=device, pipeline=[IR_weight_feature])

# Reservoir -> Reservoir connection (recurrent)
C2_w = 0.5 * torch.randn(reservoir.n, reservoir.n)
RR_weight_feature = Weight(name="RRweight", value=C2_w)
C2 = MulticompartmentConnection(source=reservoir, target=reservoir, device=device, pipeline=[RR_weight_feature])

# Reservoir -> Output connection (with STDP)
C3_w = 0.5 * torch.rand(reservoir.n, output.n)
RO_weight_feature = Weight(
    name="ROweight",
    value=C3_w,
    learning_rule=PostPre,
    nu=(1e-2, 1e-2),
    enforce_polarity=True,
)
C3 = MulticompartmentConnection(source=reservoir, target=output, device=device, pipeline=[RO_weight_feature])

# Output -> Output recurrent inhibition
inh = -5.0 * (torch.ones(output.n, output.n) - torch.eye(output.n))
output_inhibition_weight_feature = Weight(name="O_inh", value=inh)
C4 = MulticompartmentConnection(source=output, target=output, 
                                device=device, pipeline=[output_inhibition_weight_feature])

network.add_connection(C1, source="I", target="R")
network.add_connection(C2, source="R", target="R")
network.add_connection(C3, source="R", target="O")
network.add_connection(C4, source="O", target="O")

# Monitors
spikes = {}
for l in network.layers:
    spikes[l] = Monitor(network.layers[l], ["s"], time=time, device=device)
    network.add_monitor(spikes[l], name="%s_spikes" % l)

voltages = {
    "R": Monitor(network.layers["R"], ["v"], time=time, device=device),
    "O": Monitor(network.layers["O"], ["v"], time=time, device=device),
}
network.add_monitor(voltages["R"], name="R_voltages")
network.add_monitor(voltages["O"], name="O_voltages")

if gpu:
    network.to("cuda")

# Load MNIST Test Dataset for evaluation
dataset = MNIST(
    PoissonEncoder(time=time, dt=dt),
    None,
    root=os.path.join("..", "..", "data", "MNIST"),
    download=True,
    train=False,
    transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Lambda(lambda x: x * intensity)]
    ),
)

dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=1, shuffle=False, num_workers=n_workers, pin_memory=gpu
)

# Decoders
def winner_take_all_decoder(output_spikes, output_digit_matrix):
    output_spike_counts = output_spikes.sum(0).squeeze()
    if output_spike_counts.sum() == 0:
        return 0, 0, 0
    winning_neuron = output_spike_counts.argmax().item()
    winning_spike_count = output_spike_counts[winning_neuron].item()
    winning_neuron_digit_scores = output_digit_matrix[winning_neuron]
    prediction = winning_neuron_digit_scores.argmax().item()
    return prediction, winning_neuron, winning_spike_count

def first_spike_decoder(output_spikes, output_digit_matrix):
    for t in range(output_spikes.shape[0]):
        spiking_neurons = torch.where(output_spikes[t] > 0)[0]
        if len(spiking_neurons) > 0:
            first_neuron = spiking_neurons[0].item()
            neuron_digit_scores = output_digit_matrix[first_neuron]
            prediction = neuron_digit_scores.argmax().item()
            return prediction, first_neuron
    return 0, 0


# Initialize Tracking Matrices
reservoir_digit_matrix = torch.zeros(100, 10, device=device) # 100 reservoir neurons x 10 digit classes
output_digit_matrix = torch.zeros(10, 10, device=device)     # 10 output neurons x 10 digit classes
conf_matrix = torch.zeros(10, 10, device=device)             # True vs Prediction confusion matrix
previous_block_output_matrix = torch.zeros(10, 10, device=device)

correct = 0
total = 0

# Base Directories for the 5 graph categories
dir_res_assoc = "/cluster/home/spal02/bindsnet_graphs/100x10_reservoir/"
dir_out_assoc = "/cluster/home/spal02/bindsnet_graphs/10x10 output/"
dir_conf = "/cluster/home/spal02/bindsnet_graphs/10x10truepredict/"
dir_res_spike = "/cluster/home/spal02/bindsnet_graphs/reservoirspiking/"
dir_out_spike = "/cluster/home/spal02/bindsnet_graphs/outputspiking/"

for d in [dir_res_assoc, dir_out_assoc, dir_conf, dir_res_spike, dir_out_spike]:
    os.makedirs(d, exist_ok=True)

pbar = tqdm(enumerate(dataloader), total=examples)
for i, dataPoint in pbar:
    if i >= examples:
        break
    
    datum = dataPoint["encoded_image"].view(int(time / dt), 1, 1, 28, 28).to(device)
    true_label = dataPoint["label"].item()

    network.run(inputs={"I": datum}, time=time)
    
    output_spikes = spikes["O"].get("s").squeeze(1)     # Shape: [time, 10]
    reservoir_spikes = spikes["R"].get("s").squeeze(1)  # Shape: [time, 100]

    # Scoring & Prediction (using WTA decoder)
    scoring_matrix = previous_block_output_matrix if previous_block_output_matrix.sum() > 0 else output_digit_matrix
    prediction, winning_neuron, winning_spike_count = winner_take_all_decoder(output_spikes, scoring_matrix)

    total += 1
    if prediction == true_label:
        correct += 1

    # Update True vs Prediction Confusion Matrix
    conf_matrix[true_label, prediction] += 1
    
    # Update Output Association Matrix (10 x 10)
    output_spike_counts = output_spikes.sum(0)
    if output_spike_counts.sum() > 0:
        winning_out_neuron = output_spike_counts.argmax().item()
        output_digit_matrix[winning_out_neuron, true_label] += output_spike_counts[winning_out_neuron].item()

    # Update Reservoir Association Matrix (100 x 10)
    reservoir_spike_counts = reservoir_spikes.sum(0)
    if reservoir_spike_counts.sum() > 0:
        reservoir_digit_matrix[:, true_label] += reservoir_spike_counts

    # Save checkpoint graphs at the end of each block
    if (i + 1) % block_size == 0:
        completed_images = i + 1
        block_number = completed_images // block_size
        block_start = completed_images - block_size + 1
        running_acc = correct / total if total > 0 else 0.0

        print("\n" + "=" * 46)
        print(f"Completed Block {block_number}")
        print(f"Images processed: {block_start}-{completed_images}")
        print(f"Block Accuracy: {running_acc * 100:.2f}%")
        print("=" * 46)

        res_cpu = reservoir_digit_matrix.detach().cpu().numpy()
        out_cpu = output_digit_matrix.detach().cpu().numpy()
        conf_cpu = conf_matrix.detach().cpu().numpy()

        # 1. Reservoir Association Matrix Graph (100x10)
        plt.figure(figsize=(8, 6))
        plt.imshow(res_cpu, aspect="auto", interpolation="nearest", cmap="viridis")
        plt.title(f"Reservoir Association Matrix (100x10)\nImages {block_start}-{completed_images}")
        plt.xlabel("Digit Class (0-9)")
        plt.ylabel("Reservoir Neuron Index (0-99)")
        plt.colorbar()
        plt.savefig(os.path.join(dir_res_assoc, f"reservoir_assoc_images_{block_start}_{completed_images}_job_{job_id}.png"), dpi=300, bbox_inches="tight")
        plt.close()

        # 2. Output Association Matrix Graph (10x10)
        plt.figure(figsize=(6, 6))
        plt.imshow(out_cpu, interpolation="nearest", cmap="viridis")
        plt.title(f"Output Association Matrix (10x10)\nImages {block_start}-{completed_images}")
        plt.xlabel("True Digit Class (0-9)")
        plt.ylabel("Output Neuron Index (0-9)")
        plt.colorbar()
        plt.xticks(range(10))
        plt.yticks(range(10))
        plt.savefig(os.path.join(dir_out_assoc, f"output_assoc_images_{block_start}_{completed_images}_job_{job_id}.png"), dpi=300, bbox_inches="tight")
        plt.close()

        # 3. True vs Prediction Confusion Matrix Graph (10x10)
        plt.figure(figsize=(6, 6))
        plt.imshow(conf_cpu, interpolation="nearest", cmap="Blues")
        plt.title(f"Confusion Matrix (True vs Pred)\nImages {block_start}-{completed_images}")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.colorbar()
        plt.xticks(range(10))
        plt.yticks(range(10))
        plt.savefig(os.path.join(dir_conf, f"conf_matrix_images_{block_start}_{completed_images}_job_{job_id}.png"), dpi=300, bbox_inches="tight")
        plt.close()

        # 4. Reservoir Layer Spiking Raster Graph (Dead Network Check)
        res_spike_np = reservoir_spikes.detach().cpu().numpy() # [time, 100]
        time_steps_idx, neuron_idx = np.where(res_spike_np > 0)
        plt.figure(figsize=(10, 4))
        plt.scatter(time_steps_idx, neuron_idx, s=1, color="black")
        plt.title(f"Reservoir Spikes (Last Image in Block {block_number})\nImages {block_start}-{completed_images}")
        plt.xlabel("Time (ms)")
        plt.ylabel("Reservoir Neuron Index")
        plt.xlim(0, time)
        plt.ylim(-1, 100)
        plt.savefig(os.path.join(dir_res_spike, f"reservoir_spikes_images_{block_start}_{completed_images}_job_{job_id}.png"), dpi=300, bbox_inches="tight")
        plt.close()

        # 5. Output Layer Spiking Raster Graph (Dead Network Check)
        out_spike_np = output_spikes.detach().cpu().numpy() # [time, 10]
        time_steps_out, neuron_out = np.where(out_spike_np > 0)
        plt.figure(figsize=(10, 3))
        plt.scatter(time_steps_out, neuron_out, s=5, color="red")
        plt.title(f"Output Spikes (Last Image in Block {block_number})\nImages {block_start}-{completed_images}")
        plt.xlabel("Time (ms)")
        plt.ylabel("Output Neuron Index")
        plt.xlim(0, time)
        plt.ylim(-1, 10)
        plt.yticks(range(10))
        plt.savefig(os.path.join(dir_out_spike, f"output_spikes_images_{block_start}_{completed_images}_job_{job_id}.png"), dpi=300, bbox_inches="tight")
        plt.close()

        # Carry over and Reset for the next 1,000-image chunk
        previous_block_output_matrix = output_digit_matrix.clone()
        reservoir_digit_matrix.zero_()
        output_digit_matrix.zero_()
        conf_matrix.zero_()
        correct = 0
        total = 0

    network.reset_state_variables()

print("Evaluation completed successfully!")