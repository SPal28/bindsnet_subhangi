import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from tqdm import tqdm

from bindsnet.analysis.plotting import (plot_input, plot_spikes,plot_voltages,plot_weights,
)
from bindsnet.datasets import MNIST
from bindsnet.encoding import PoissonEncoder
from bindsnet.network import Network
from bindsnet.network.topology_features import Probability, Weight, Mask
from bindsnet.learning.MCC_learning import PostPre
from bindsnet.network.monitors import Monitor
from bindsnet.network.nodes import Input, LIFNodes
from bindsnet.network.topology import Connection, MulticompartmentConnection
from bindsnet.utils import get_square_weights

# Keep output filenames tied to the SLURM job when running on the cluster.
job_id = os.environ.get("SLURM_JOB_ID", "local")
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--n_neurons", type=int, default=100)
parser.add_argument("--n_epochs", type=int, default=1)
parser.add_argument("--examples", type=int, default=1000)  
parser.add_argument("--n_workers", type=int, default=0)
parser.add_argument("--time", type=int, default=250)
parser.add_argument("--dt", type=float, default=1.0)
parser.add_argument("--intensity", type=float, default=64)
parser.add_argument("--progress_interval", type=int, default=10)
parser.add_argument("--update_interval", type=int, default=250)
parser.add_argument("--block_size", type=int, default=2000, help="Images per evaluation/saving block")
parser.add_argument("--plot", dest="plot", action="store_true")
parser.add_argument("--gpu", dest="gpu", action="store_true")
parser.set_defaults(plot=False, gpu=False)

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
plot = args.plot
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
output = LIFNodes(10, traces=True, thresh=-55)
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

print("Initial C3 weights:")
print(C3_w[:5])
print("Mean C3 weight:", C3_w.mean())

# Output -> Output recurrent inhibition
inh = -5.0 * (torch.ones(output.n, output.n) - torch.eye(output.n))
output_inhibition_weight_feature = Weight(name="O_inh", value=inh)
C4 = MulticompartmentConnection(source=output, target=output, device=device, pipeline=[output_inhibition_weight_feature])

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
    train=False,  # Using test set as requested
    transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Lambda(lambda x: x * intensity)]
    ),
)

dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=1, shuffle=False, num_workers=n_workers, pin_memory=gpu
)

# Decoders reading from Output layer spikes
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

# Association Matrix: 10 Output neurons x 10 Digit classes
output_digit_matrix = torch.zeros(10, 10, device=device)
previous_block_matrix = torch.zeros(10, 10, device=device)

correct = 0
total = 0
acc_history = []
iter_history = []
conf_matrix = torch.zeros(10, 10, device=device)

# Base directory for outputs (safe fallback if cluster path doesn't exist)
output_dir = f"/cluster/home/spal02/bindsnet_graphs/conf_matrix/"
os.makedirs(output_dir, exist_ok=True)

pbar = tqdm(enumerate(dataloader), total=examples)
for i, dataPoint in pbar:
    if i >= examples:
        break
    
    datum = dataPoint["encoded_image"].view(int(time / dt), 1, 1, 28, 28).to(device)
    true_label = dataPoint["label"].item()

    network.run(inputs={"I": datum}, time=time)

    # Get spikes from the output layer
    output_spikes = spikes["O"].get("s")

    # Predict using the previous block's accumulated association matrix (prevents zero-score learning loops)
    scoring_matrix = previous_block_matrix if previous_block_matrix.sum() > 0 else output_digit_matrix
    prediction, winning_neuron, winning_spike_count = winner_take_all_decoder(output_spikes, scoring_matrix)

    total += 1
    if prediction == true_label:
        correct += 1

    conf_matrix[true_label, prediction] += 1
    
    # Update current block's association matrix with output layer spikes
    output_spike_counts = output_spikes.sum(0).squeeze()
    if output_spike_counts.sum() > 0:
        winning_out_neuron = output_spike_counts.argmax().item()
        output_digit_matrix[winning_out_neuron, true_label] += output_spike_counts[winning_out_neuron].item()

    running_acc = correct / total if total > 0 else 0.0
    acc_history.append(running_acc)
    iter_history.append(i)

    if (i + 1) % block_size == 0:
        completed_images = i + 1
        block_number = completed_images // block_size
        block_start = completed_images - block_size + 1

        print("\n" + "=" * 46)
        print(f"Completed Block {block_number}")
        print(f"Images processed: {block_start}-{completed_images}")
        print(f"Block Accuracy: {running_acc * 100:.2f}%")
        print("=" * 46)

        # Save matrices and heatmaps
        torch.save(output_digit_matrix.detach().cpu(), os.path.join(output_dir, f"output_digit_matrix_images_{block_start}_{completed_images}_job_{job_id}.pt"))
        torch.save(conf_matrix.detach().cpu(), os.path.join(output_dir, f"conf_matrix_images_{block_start}_{completed_images}_job_{job_id}.pt"))

        # Plot Confusion Matrix Heatmap
        plt.figure(figsize=(8, 6))
        plt.imshow(conf_matrix.detach().cpu().numpy(), interpolation="nearest")
        plt.title(f"Confusion Matrix\nImages {block_start}-{completed_images}")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.colorbar()
        plt.xticks(range(10))
        plt.yticks(range(10))
        plt.savefig(os.path.join(output_dir, f"WTA_conf_matrix_images_{block_start}_{completed_images}_job_{job_id}.png"), dpi=300, bbox_inches="tight")
        plt.close()

        # Carry over current matrix to previous block matrix for continuous evaluation
        previous_block_matrix = output_digit_matrix.clone()
        conf_matrix.zero_()
        output_digit_matrix.zero_()
        correct = 0
        total = 0

    network.reset_state_variables()

print("Evaluation completed successfully!")