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
parser.add_argument("--n_neurons", type=int, default=500)
parser.add_argument("--n_epochs", type=int, default=1)
parser.add_argument("--examples", type=int, default=500)
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

# Reservoir layer - each neurons recieves a voltage, reaches threshold, spike, and resets 
# thresh represents biologically inspired neurons
reservoir = LIFNodes(n_neurons, traces=True, thresh = -56 + np.random.randn(n_neurons).astype(float),)
network.add_layer(reservoir, name = "R")

# Output layer - creates 10 neurons 
output = LIFNodes(10, traces=True)
network.add_layer(output, name="O")

# Input -> Reservoir 
C1_w = 0.5 * torch.randn(inpt.n, reservoir.n)

IR_weight_feature = Weight(name="IRweight", value=C1_w)
pipeline = [IR_weight_feature]

C1 = MulticompartmentConnection(source = inpt, target = reservoir, device = device, pipeline = pipeline)
# orginal
# C1 = Connection(source=inpt,target=reservoir,w=0.5 * torch.randn(inpt.n, reservoir.n),)


# Reservoir -> Reservoir (recurrent)
# rand - biological (meaning that it shouldnt be both negative or positive)
C2_w = 0.5 * torch.randn(reservoir.n, reservoir.n)

RR_weight_feature = Weight(name = "RRweight", value = C2_w)
pipeline = [RR_weight_feature]

C2 = MulticompartmentConnection(source = reservoir, target = reservoir, device = device, pipeline = pipeline)
# C2 = Connection(source=reservoir,target=reservoir,w=0.5 * torch.randn(reservoir.n, reservoir.n),)


# Reservoir -> Output (STDP)
# Rand goes from 0 to 1 meaning that all the weights will be positive aka excitaory
C3_w = 0.1 * torch.rand(reservoir.n, output.n)
RO_weight_feature = Weight(
    name="ROweight",
    value=C3_w,
    learning_rule=PostPre,
    nu=(1e-2, 1e-2),
    enforce_polarity=False,
)
pipeline = [RO_weight_feature]

C3 = MulticompartmentConnection(source=reservoir, target=output, device=device, pipeline=pipeline)
# C3 = Connection(source=reservoir,target=output,w=0.1 * torch.rand(reservoir.n, output.n),update_rule=PostPre,nu=(1e-2, 1e-2),)

# DEBUG: prints the first 5 neuron weights in the 500x10 matrix 
print("Initial C3 Outgoing weights:")
print(C3_w[:5])
print("Mean C3 weight:", C3_w.mean())

# Output -> Output (recurrent)
inh = -20 * (torch.ones(output.n, output.n)- torch.eye(output.n))

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
    dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=gpu
)

# Run training data on reservoir computer and store (spikes per neuron, label) per example.
# Note: Because this is a reservoir network, no adjustments of neuron parameters occurs in this phase.
n_iters = examples  # 500 images 

# dataloader - holds every mnist image (image, enocded_image, label
# (1, imag0),, etc
pbar = tqdm(enumerate(dataloader))


for i, dataPoint in pbar:
    if i > n_iters:
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

    if i % 50 == 0:
        print(f"\nIteration {i}")

        for neuron in range(10):
            print(
                f"Output neuron {neuron}: "
                f"mean={C3_w[:, neuron].mean():.4f}, "
                f"max={C3_w[:, neuron].max():.4f}, "
                f"min={C3_w[:, neuron].min():.4f}"
            )
    

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


# bc STDP is still active when i run them throguh neurons assignments + accuracy, the weights are still changing 
# set nu to 0 so it doesnt change anymore 
#RO_weight_feature.learning_rule.nu = (0, 0)


# DEBUG
print("After training C3 outgoing weights:")
print(C3_w[:5])
print("Mean C3 weight:", C3_w.mean())



# Run same simulation on reservoir with testing data instead of training data
# (see training section for intuition)
n_iters = examples

pbar = tqdm(enumerate(dataloader))
for i, dataPoint in pbar:
    if i > n_iters:
        break
    datum = dataPoint["encoded_image"].view(int(time / dt), 1, 1, 28, 28).to(device)
    label = dataPoint["label"]
    pbar.set_description_str("Testing progress: (%d / %d)" % (i, n_iters))

    network.run(inputs={"I": datum}, time=time)

    

    # ---------------------------------------------
    # Reservoir statistics
    # ---------------------------------------------
    reservoir_spikes = spikes["R"].get("s")          # (time x 500)
    reservoir_counts = reservoir_spikes.sum(0)       # spikes per reservoir neuron

    active_reservoir = (reservoir_counts > 0).sum().item()
    total_reservoir_spikes = reservoir_counts.sum().item()

    # ---------------------------------------------
    # Output statistics
    # ---------------------------------------------
    output_spikes = spikes["O"].get("s")             # (time x 10)
    output_counts = output_spikes.sum(0).squeeze()   # spikes per output neuron

    active_output = (output_counts > 0).sum().item()

    sorted_counts, _ = torch.sort(output_counts, descending=True)
    winner_spikes = sorted_counts[0].item()
    runnerup_spikes = sorted_counts[1].item()

    # ---------------------------------------------
    # Print results
    # ---------------------------------------------
    print(f"\nImage {i}")
    print(f"Reservoir active neurons : {active_reservoir}/500")
    print(f"Reservoir total spikes   : {total_reservoir_spikes}")
    print(f"Average spikes per active reservoir neuron : " f"{total_reservoir_spikes / max(active_reservoir,1):.2f}")

    print(f"Output active neurons    : {active_output}/10")
    print(f"Winner spike count       : {winner_spikes}")
    print(f"Runner-up spike count    : {runnerup_spikes}")

    print("Output spike counts:", output_counts)

   
    if plot:
        inpt_axes, inpt_ims = plot_input(dataPoint["image"].view(28, 28),datum.view(time, 784).sum(0).view(28, 28),label=label,axes=inpt_axes,ims=inpt_ims,)
        spike_ims, spike_axes = plot_spikes({layer: spikes[layer].get("s").view(time, -1) for layer in spikes},axes=spike_axes,ims=spike_ims,)
        voltage_ims, voltage_axes = plot_voltages({layer: voltages[layer].get("v").view(time, -1) for layer in voltages},ims=voltage_ims,axes=voltage_axes,)
        weights_im = plot_weights(get_square_weights(C1_w, 23, 28), im=weights_im, wmin=-2, wmax=2)
        #recurrent weights
        weights_im2 = plot_weights(C2_w, im=weights_im2, wmin=-2, wmax=2)
        #FIX: get the rest of the connections
        plt.pause(1e-8)
    network.reset_state_variables()



#creates empty tensor with 10 zero avlues 
assignments = torch.zeros(10, dtype=torch.long)

# proportions[output neuron][digit]
# creates a 10x10 matrix (digits x neurons num) tp track how many spikes each neuron porduces when seeing each digit
# reservoir_response[digit][reservoir neuron]
reservoir_response = torch.zeros(10, n_neurons)

# Number of examples seen for each digit
digit_counts = torch.zeros(10)


# Run training images through the trained network
# to determine what digit each neuron represents
for i, dataPoint in enumerate(dataloader):

    if i > n_iters:
        break

    # preprocess image ( shapes the image to what bindsnet expects)
    datum = dataPoint["encoded_image"].view(int(time / dt), 1, 1, 28, 28).to(device)

    #retrives the tru digital label
    label = dataPoint["label"]

    # run network
    network.run(inputs={"I": datum},time=time,)

   
    # Count reservoir spikes (500 neurons)
    reservoir_counts = spikes["R"].get("s").sum(0).squeeze()

    digit = label.item()

    # Add this image's spike counts
    reservoir_response[digit] += reservoir_counts

    # Count how many examples of this digit we've seen
    digit_counts[digit] += 1


    network.reset_state_variables()

for digit in range(10):
    if digit_counts[digit] > 0:
        reservoir_response[digit] /= digit_counts[digit]



print("Neuron assignments:")
print(assignments)

##############################################################
# READOUT METHODS
##############################################################

def winner_take_all_decoder(output_spikes, assignments):
    spike_counts = output_spikes.sum(0)

    winning_neuron = spike_counts.argmax().item()

    return assignments[winning_neuron].item()


def first_spike_decoder(output_spikes, assignments):

    for t in range(output_spikes.shape[0]):

        spiking = torch.where(output_spikes[t] > 0)[0]

        if len(spiking) > 0:

            first_neuron = spiking[0].item()

            return assignments[first_neuron].item()

    # fallback if nothing spikes
    return winner_take_all_decoder(output_spikes, assignments)


def threshold_decoder(output_spikes, assignments, threshold=2):

    spike_counts = output_spikes.sum(0)

    above_threshold = torch.where(spike_counts >= threshold)[0]

    if len(above_threshold) > 0:

        neuron = above_threshold[0].item()

        return assignments[neuron].item()

    return winner_take_all_decoder(output_spikes, assignments)

acc_history = []
iter_history = []


#accuaracy calcultion 

correct = 0
total = 0

conf_matrix = torch.zeros(10, 10)

pbar = tqdm(enumerate(dataloader))

for i, dataPoint in pbar:

    if i > n_iters:
        break


    # preprocess image
    datum = dataPoint["encoded_image"].view(int(time / dt), 1, 1, 28, 28).to(device)

    label = dataPoint["label"]


    # run network
    network.run(
        inputs={"I": datum},time=time,)



    output_spikes = spikes["O"].get("s")

    prediction = winner_take_all_decoder(output_spikes, assignments)


    total += 1

    true_label = label.item()
    pred_label = prediction

    if prediction == true_label:
        correct += 1

    conf_matrix[true_label, pred_label] += 1

    running_acc = correct / total

    # store values for plotting
    acc_history.append(running_acc)
    iter_history.append(i)

    network.reset_state_variables()



print("\nAccuracy: %.2f %%" % (100.0 * correct / total))


print("\nConfusion Matrix:")
print(conf_matrix)


plt.figure()
plt.imshow(conf_matrix, interpolation="nearest")
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.colorbar()
plt.xticks(range(10))
plt.yticks(range(10))
conf_matrix_path = f"/cluster/home/spal02/bindsnet_graphs/conf_matrix/conf_matrix_job_{job_id}.png"
plt.savefig(conf_matrix_path, dpi=300, bbox_inches="tight")
plt.close()



plt.figure()
plt.plot(iter_history, acc_history)
plt.xlabel("Iteration")
plt.ylabel("Accuracy")
plt.title("Accuracy over time")
plt.grid(True)
accuracy_path = f"/cluster/home/spal02/bindsnet_graphs/spike_graphs/accuracy_job_{job_id}.png"
plt.savefig(accuracy_path, dpi=300, bbox_inches="tight")
plt.close()


plt.figure(figsize=(18,6))

plt.imshow(
    reservoir_response.numpy(),
    aspect="auto",
    cmap="hot",
    interpolation="nearest",
)

plt.xlabel("reservoir neuron")
plt.ylabel("digit")
plt.title("reservoir neuron activity")

plt.xticks(np.arange(0, n_neurons + 1, 50))
plt.yticks(range(10))

plt.colorbar(label="avg spike count")

response_path = (
    f"/cluster/home/spal02/bindsnet_graphs/"
    f"reservoir_response_job_{job_id}.png"
)

plt.savefig(response_path, dpi=300, bbox_inches="tight")
plt.close()