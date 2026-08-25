import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from tqdm import tqdm

from bindsnet.analysis.plotting import (
    plot_input,
    plot_spikes,
    plot_voltages,
    plot_weights,
)
from bindsnet.datasets import MNIST
from bindsnet.encoding import PoissonEncoder
from bindsnet.network import Network
# note: delete mask
from bindsnet.network.topology_features import Probability, Weight, Mask
from bindsnet.learning.MCC_learning import PostPre

# Build a simple two-layer, input-output network.
from bindsnet.network.monitors import Monitor
from bindsnet.network.nodes import Input, LIFNodes
from bindsnet.network.topology import Connection
from bindsnet.utils import get_square_weights
from bindsnet.network.topology import MulticompartmentConnection

job_id = os.environ.get("SLURM_JOB_ID", "local")
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--n_neurons", type=int, default=100)
parser.add_argument("--n_epochs", type=int, default=1)
parser.add_argument("--examples", type=int, default=60000)
parser.add_argument("--n_workers", type=int, default=-1)
parser.add_argument("--time", type=int, default=250)
parser.add_argument("--dt", type=int, default=1.0)
parser.add_argument("--intensity", type=float, default=64)
parser.add_argument("--progress_interval", type=int, default=10)
parser.add_argument("--update_interval", type=int, default=250)
parser.add_argument("--plot", dest="plot", action="store_true")
parser.add_argument("--gpu", dest="gpu", action="store_true")
parser.set_defaults(plot=True, gpu=False, train=True)

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
train = args.train
plot = args.plot
gpu = args.gpu

np.random.seed(seed)
torch.cuda.manual_seed_all(seed)
torch.manual_seed(seed)

# Sets up Gpu use
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if gpu and torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
else:
    torch.manual_seed(seed)
    device = "cpu"
    if gpu:
        gpu = False
torch.set_num_threads(os.cpu_count() - 1)
print("Running on Device = ", device)
# dt = simulation timestep
network = Network(dt=dt)

# Input layer - 784 neurons, 1 image channel (grayscale), 28x28 grid, traces stores the spikes
inpt = Input(784, shape=(1, 28, 28), traces = True)
network.add_layer(inpt, name="I")

# reservoir layer- each neurons recieves a voltage, reaches threshold spike, and resets 
# thresh represents biologically inspired neurons
reservoir = LIFNodes(n_neurons, traces=True, thresh = -52 + np.random.randn(n_neurons).astype(float),)
network.add_layer(reservoir, name = "R")

# output layer - creates 10 neurons 
output = LIFNodes(10, traces=True, thresh=-55)
network.add_layer(output, name="O")

# input -> reservoir 
C1_w = 0.5 * torch.randn(inpt.n, reservoir.n)

IR_weight_feature = Weight(name="IRweight", value=C1_w)
pipeline = [IR_weight_feature]

C1 = MulticompartmentConnection(source = inpt, target = reservoir, device = device, pipeline = pipeline)
# orginal
# C1 = Connection(source=inpt,target=reservoir,w=0.5 * torch.randn(inpt.n, reservoir.n),)


# reservoir -> reservoir (recurrent)
# rand - biological (meaning that it shouldnt be both negative or positive)
C2_w = 0.5 * torch.randn(reservoir.n, reservoir.n)

RR_weight_feature = Weight(name = "RRweight", value = C2_w)
pipeline = [RR_weight_feature]

C2 = MulticompartmentConnection(source = reservoir, target = reservoir, device = device, pipeline = pipeline)
# C2 = Connection(source=reservoir,target=reservoir,w=0.5 * torch.randn(reservoir.n, reservoir.n),)


# reservoir -> output (STDP)
# rand goes from 0 to 1 meaning that all the weights will be positive aka excitaory
C3_w = 0.5 * torch.rand(reservoir.n, output.n)
RO_weight_feature = Weight(
    name="ROweight",
    value=C3_w,
    learning_rule=PostPre,
    nu=(1e-2, 1e-2),
    enforce_polarity=True,
)
pipeline = [RO_weight_feature]

C3 = MulticompartmentConnection(source=reservoir, target=output, device=device, pipeline=pipeline)
# C3 = Connection(source=reservoir,target=output,w=0.1 * torch.rand(reservoir.n, output.n),update_rule=PostPre,nu=(1e-2, 1e-2),)

# debug: prints the first 5 neuron weights in the 500x10 matrix 
print("Initial C3 weights:")
print(C3_w[:5])
print("Mean C3 weight:", C3_w.mean())

# output -> output (recurrent)
inh = -5 * (torch.ones(output.n, output.n)- torch.eye(output.n))

output_inhibition_weight_feature = Weight(name="",value=inh)
pipeline = [output_inhibition_weight_feature]

C4 = MulticompartmentConnection(source=output,target=output, device = device, pipeline = pipeline)
# C4 = Connection(source=output,target=output,w=inh,)

network.add_connection(C1, source="I", target="R")
network.add_connection(C2, source="R", target="R")
network.add_connection(C3, source="R", target="O")
network.add_connection(C4, source="O", target="O")

# Monitors for visualizing activity
spikes = {}
for l in network.layers:
    # "s" is the spiking variable which records the num of spikes in each layer 
    spikes[l] = Monitor(network.layers[l], ["s"], time=time, device=device)
    network.add_monitor(spikes[l], name="%s_spikes" % l)


voltages = {"R": Monitor(network.layers["R"],["v"],time=time,device=device,),"O": Monitor(network.layers["O"],["v"],time=time,device=device,),}

network.add_monitor(voltages["R"],name="R_voltages",)

network.add_monitor(voltages["O"],name="O_voltages",)

# Directs network to GPU
if gpu:
    network.to("cuda")

# Get MNIST training images and labels.
# Load MNIST data.
dataset = MNIST(
    PoissonEncoder(time=time, dt=dt),
    None,
    root=os.path.join("..", "..", "data", "MNIST"),
    download=True,
    transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Lambda(lambda x: x * intensity)]
    ),
)

inpt_axes = None
inpt_ims = None
spike_axes = None
spike_ims = None
weights_im = None
weights_im2 = None
voltage_ims = None
voltage_axes = None

# Create a dataloader to iterate and batch data
dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=gpu
)

# Run training data on reservoir computer and store (spikes per neuron, label) per example.
# Note: Because this is a reservoir network, no adjustments of neuron parameters occurs in this phase.
n_iters = examples  # 60,000 images 
neuron_spike_history = []
weight_sum_history = []
# dataloader - holds every mnist image (image, enocded_image, label
# (1, imag0),, etc
pbar = tqdm(enumerate(dataloader))


for i, dataPoint in pbar:
    if i >= n_iters:
        break
    # Extract & resize the MNIST samples image data for training
    #       int(time / dt)  -> length of spike train
    #       28 x 28         -> size of sample
    # datum is holding the spike data (ex. 1011101)
    # .view is rehsping the spkie train (250 timesteps, 1 batch, 1 channel, 28 rwos, 28 cols)
    datum = dataPoint["encoded_image"].view(int(time / dt), 1, 1, 28, 28).to(device)
    #extract label 
    label = dataPoint["label"]
    pbar.set_description_str("Train progress: (%d / %d)" % (i, n_iters))

    # takes the spike train, datum, and feeds it into the input later 
    # for each timestep it does this: input spikes, update reservoir neurons, reservoir neurons spike, update output neurons, output neurons spike, apply STDP to reservoir,-> output weights 
    network.run(inputs={"I": datum}, time=time)

    # debug: record per-neuron spikes and C3 weight sums this iteration
    neuron_spike_history.append(spikes["O"].get("s").sum(0).squeeze().clone())
    weight_sum_history.append(C3_w.sum(0).clone())


    # Plot spiking activity using monitors
    if plot:
        # Plot the current image and reconstructed/encoded image
        inpt_axes, inpt_ims = plot_input(
            dataPoint["image"].view(28, 28),
            datum.view(int(time / dt), 784).sum(0).view(28, 28),
            label=label,
            axes=inpt_axes,
            ims=inpt_ims,
        )
        # Plot spikes
        spike_ims, spike_axes = plot_spikes(
            {layer: spikes[layer].get("s").view(time, -1) for layer in spikes},
            axes=spike_axes,
            ims=spike_ims,
        )
        # Plot voltages
        voltage_ims, voltage_axes = plot_voltages(
            {layer: voltages[layer].get("v").view(time, -1) for layer in voltages},
            ims=voltage_ims,
            axes=voltage_axes,
        )
        # Plot weights between input and output
        weights_im = plot_weights(
            get_square_weights(C1_w, 23, 28), im=weights_im, wmin=-2, wmax=2
        )
        # Plot weights between output and output
        weights_im2 = plot_weights(C2_w, im=weights_im2, wmin=-2, wmax=2)

        #FIX: make sure all connections are displayed 

        plt.pause(1e-8)
    network.reset_state_variables()

# debug: plot per-neuron spike activity and weight sums over training
spike_history_tensor = torch.stack(neuron_spike_history)   
weight_sum_tensor = torch.stack(weight_sum_history)        

plt.figure()
for neuron in range(10):
    plt.plot(spike_history_tensor[:, neuron].numpy(), label=f"N{neuron}")
plt.xlabel("Training iteration")
plt.ylabel("Spikes per example")
plt.legend(fontsize=6)
plt.title("Per-neuron spiking activity over training")
plt.savefig(f"/cluster/home/spal02/bindsnet_graphs/spike_graphs/spike_activity_job_{job_id}.png", dpi=300, bbox_inches="tight")
plt.close()

plt.figure()
for neuron in range(10):
    plt.plot(weight_sum_tensor[:, neuron].numpy(), label=f"N{neuron}")
plt.xlabel("Training iteration")
plt.ylabel("Sum of incoming C3 weights")
plt.legend(fontsize=6)
plt.title("Per-neuron C3 weight sum over training")
plt.savefig(f"/cluster/home/spal02/bindsnet_graphs/spike_graphs/weight_sum_job_{job_id}.png", dpi=300, bbox_inches="tight")
plt.close()

#freezes STDP weights 
RO_weight_feature.learning_rule.nu = (0, 0)


# DEBUG
print("After training C3 weights:")
print(C3_w[:5])
print("Mean C3 weight:", C3_w.mean())

# reservoir to digit cumulative matrix 
# rows = 100 reservoir neurons
# columns = 10 MNIST digit classes
#
# each entry represents the cumulative association between
# a reservoir neuron and a digit

reservoir_digit_matrix = torch.zeros(
    n_neurons,
    10,
    device=device
)

print("Initial reservoir-digit matrix shape:")
print(reservoir_digit_matrix.shape)

# read out methods

# WTA: counts spikes form eacg of the reservoir neurons and finds the neuron
# with the most spikes and the digit with the strongest association is the 
# prediction
def winner_take_all_decoder(reservoir_spikes, reservoir_digit_matrix):

    reservoir_spike_counts = reservoir_spikes.sum(0).squeeze()
    winning_neuron = reservoir_spike_counts.argmax().item()
    winning_spike_count = reservoir_spike_counts[winning_neuron]
    winning_neuron_digit_scores = reservoir_digit_matrix[winning_neuron]
    prediction = winning_neuron_digit_scores.argmax().item()

    return prediction, winning_neuron, winning_spike_count
# first spike wins!
def first_spike_decoder(reservoir_spikes, reservoir_digit_matrix):

    for t in range(reservoir_spikes.shape[0]):
        spiking_neurons = torch.where(reservoir_spikes[t] > 0)[0]
        if len(spiking_neurons) > 0:
            first_neuron = spiking_neurons[0].item()

           
            neuron_digit_scores = reservoir_digit_matrix[first_neuron]
            prediction = neuron_digit_scores.argmax().item()

            return prediction, first_neuron

    #if no neuron spikes, return a default prediction
    return 0, 0


acc_history = []
iter_history = []

# section off the 60,000 images 
block_size = 10000

#accuarcy counters 
correct = 0
total = 0

acc_history = []
iter_history = []

#10 x 10 confusion matrix for the CURRENT 10,000 image sections
#rows = true digit
#columns = predicted digit
conf_matrix = torch.zeros(10, 10, device=device)

pbar = tqdm(enumerate(dataloader))

for i, dataPoint in pbar:
    if i >= n_iters:
        break
    #taking the poisson-encoded images and reshaping it 
    datum = dataPoint["encoded_image"].view(int(time / dt), 1, 1, 28, 28).to(device)
    # get the actual label, not used when running the network
    true_label = dataPoint["label"].item()

    #run network
    network.run(inputs={"I": datum},time=time,)

    #get the spkies from the r layer (not the one that spiked the msot)
    #FIX:  get the spikes from the very last layer
    reservoir_spikes = spikes["R"].get("s")

   # find which reservoir neuron spiked the most 
    # FIX=
    prediction, winning_neuron, winning_spike_count = (
        winner_take_all_decoder(reservoir_spikes, reservoir_digit_matrix)
    )

    total += 1

    if prediction == true_label:
        correct += 1

    #update the 10x10 matrix 
    conf_matrix[true_label, prediction] += 1

    #update the 500x10 matrix 

    reservoir_digit_matrix[winning_neuron,true_label] += winning_spike_count

    running_acc = correct / total

    acc_history.append(running_acc)
    iter_history.append(i)

    # for every 10,000 images save both the matrices and heatmaps , then reset
    # note: some graph code was generated with the help of ChatGPT for clarity
    if (i + 1) % block_size == 0:

        completed_images = i + 1
        block_number = completed_images // block_size
        block_start = completed_images - block_size + 1

        print("\n")
        print("==============================================")
        print(f"completed block {block_number}")
        print(f"images processed: {block_start}-{completed_images}")
        print(f"block accuracy: {100 * running_acc:.2f}%")
        print("==============================================")

        # save the 100x10 reservoir-digit matrix
        reservoir_matrix_path = (
            f"/cluster/home/spal02/bindsnet_graphs/"
            f"conf_matrix/"
            f"reservoir_digit_matrix_"
            f"images_{block_start}_{completed_images}_"
            f"job_{job_id}.pt"
        )

        torch.save(
            reservoir_digit_matrix.detach().cpu(),
            reservoir_matrix_path
        )

        print(
            f"Saved 100x10 reservoir-digit matrix: "
            f"{reservoir_matrix_path}"
        )

        # save 10 x 10 matrix 

        conf_matrix_path = (
            f"/cluster/home/spal02/bindsnet_graphs/"
            f"conf_matrix/"
            f"conf_matrix_"
            f"images_{block_start}_{completed_images}_"
            f"job_{job_id}.pt"
        )

        torch.save(
            conf_matrix.detach().cpu(),
            conf_matrix_path
        )

        print(
            f"Saved 10x10 confusion matrix: "
            f"{conf_matrix_path}"
        )

        # save 10x10 matrix has a heatmap 

        plt.figure(figsize=(8, 6))

        plt.imshow(
            conf_matrix.detach().cpu().numpy(),
            interpolation="nearest"
        )

        plt.title(
            f"WTA Confusion Matrix\n"
            f"Images {block_start}-{completed_images}"
        )

        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")

        plt.colorbar()

        plt.xticks(range(10))
        plt.yticks(range(10))

        conf_heatmap_path = (
            f"/cluster/home/spal02/bindsnet_graphs/"
            f"conf_matrix/"
            f"WTA_conf_matrix_"
            f"images_{block_start}_{completed_images}_"
            f"job_{job_id}.png"
        )

        plt.savefig(
            conf_heatmap_path,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

        print(
            f"Saved confusion matrix heatmap: "
            f"{conf_heatmap_path}"
        )

        # save 100x10 matrix as a heatmap

        plt.figure(figsize=(10, 8))

        plt.imshow(
            reservoir_digit_matrix.detach().cpu().numpy(),
            aspect="auto",
            interpolation="nearest"
        )

        plt.title(
            f"WTA Reservoir-Digit Matrix\n"
            f"Images {block_start}-{completed_images}"
        )

        plt.xlabel("Digit Class")
        plt.ylabel("Reservoir Neuron")

        plt.colorbar()

        plt.xticks(range(10))
        plt.yticks(range(n_neurons))

        reservoir_heatmap_path = (
            f"/cluster/home/spal02/bindsnet_graphs/"
            f"conf_matrix/"
            f"WTA_reservoir_digit_matrix_"
            f"images_{block_start}_{completed_images}_"
            f"job_{job_id}.png"
        )

        plt.savefig(
            reservoir_heatmap_path,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

        print(
            f"Saved reservoir-digit heatmap: "
            f"{reservoir_heatmap_path}"
        )

        #print matrices 

        print("\n100x10 Reservoir-Digit Matrix for this block:")
        print(reservoir_digit_matrix)

        print("\n10x10 Confusion Matrix for this block:")
        print(conf_matrix)

        #reset matrices and accuarcy counters 

        conf_matrix.zero_()
        reservoir_digit_matrix.zero_()

        correct = 0
        total = 0

        print(
            f"\nReset matrices and counters for block "
            f"{block_number + 1}"
        )

    #reset network

    network.reset_state_variables()