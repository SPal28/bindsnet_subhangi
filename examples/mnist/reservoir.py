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
reservoir = LIFNodes(n_neurons, traces=True, thresh = -52 + np.random.randn(n_neurons).astype(float),)
network.add_layer(reservoir, name = "R")

# Output layer - creates 10 neurons 
output = LIFNodes(10, traces=True, thresh=-55)
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

# DEBUG: prints the first 5 neuron weights in the 500x10 matrix 
print("Initial C3 weights:")
print(C3_w[:5])
print("Mean C3 weight:", C3_w.mean())

# Output -> Output (recurrent)
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
    dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=gpu
)

# Run training data on reservoir computer and store (spikes per neuron, label) per example.
# Note: Because this is a reservoir network, no adjustments of neuron parameters occurs in this phase.
n_iters = examples  # 500 images 
neuron_spike_history = []
weight_sum_history = []

training_pairs = []
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



    #NEW: record per-neuron spikes and C3 weight sums this iteration
    neuron_spike_history.append(spikes["O"].get("s").sum(0).squeeze().clone())
    weight_sum_history.append(C3_w.sum(0).clone())

    # NEW: save output spike trains for logistic regression
    training_pairs.append(
        (
            spikes["O"].get("s").clone(),
            label.clone()
        )
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

print("Number of training pairs:", len(training_pairs))
print("Spike tensor shape:", training_pairs[0][0].shape)
print("First label:", training_pairs[0][1].item())
print("First sample spike counts:", training_pairs[0][0].sum(0))

# --- NEW: plot per-neuron spike activity and weight sums over training ---
spike_history_tensor = torch.stack(neuron_spike_history)   # [n_iters, 10]
weight_sum_tensor = torch.stack(weight_sum_history)         # [n_iters, 10]

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

# bc STDP is still active when i run them throguh neurons assignments + accuracy, the weights are still changing 
# set nu to 0 so it doesnt change anymore 
RO_weight_feature.learning_rule.nu = (0, 0)


class NN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NN, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, x):
        x = x.float().view(-1)
        return torch.sigmoid(self.linear(x))
    
lr_epochs = 100
model = NN(time * 10, 10).to(device)
criterion = torch.nn.MSELoss(reduction="sum")
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)

for epoch in range(lr_epochs):
    avg_loss = 0
    for spikes_train, label in training_pairs:
        optimizer.zero_grad()
        outputs = model(spikes_train)
        target = torch.zeros(10, device=device)
        target[label.item()] = 1
        loss = criterion(outputs, target)
        avg_loss += loss.item()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}/{lr_epochs}: "
          f"{avg_loss/len(training_pairs):.4f}")

# DEBUG
print("After training C3 weights:")
print(C3_w[:5])
print("Mean C3 weight:", C3_w.mean())

# --- NEW: reservoir sanity check --- 
print("Reservoir mean spikes/neuron:", spikes["R"].get("s").sum(0).float().mean().item()) 
print("Reservoir active neurons (>0 spikes):", (spikes["R"].get("s").sum(0) > 0).sum().item(), "/", n_neurons)


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

    

    #retrives rhe output spikes (250x10), sum(0) sums scross time
    print(spikes["O"].get("s").sum(0)) # important

    # DEBUG: confirm C3 weights are frozen ---
    if i % 50 == 0:
        print(f"[iter {i}] C3_w mean: {C3_w.mean().item():.6f}")

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

    # Prepare image
    datum = dataPoint["encoded_image"].view(
        int(time / dt), 1, 1, 28, 28
    ).to(device)

    label = dataPoint["label"]

    # Run SNN
    network.run(inputs={"I": datum}, time=time)

    # Get output spikes
    output_spikes = spikes["O"].get("s")

    # Logistic regression prediction
    outputs = model(output_spikes)

    prediction = outputs.argmax().item()

    true_label = label.item()

    total += 1

    if prediction == true_label:
        correct += 1

    conf_matrix[true_label, prediction] += 1

    running_acc = correct / total

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