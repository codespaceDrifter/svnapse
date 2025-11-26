# stebian neuron population
import torch
import torch.nn as nn

# this uses rate coding: activations are floats. representing how many spikes in a period. 
# neuron count: size 
# weight count: size * size 
# each column is a neuron each row is a synapse
# weight [i,j] can be interpreted as the weight from neuron j to neuron i
class StebPop:
    def __init__ (self, size):
        self.size = size
        self.weights = torch.Tensor(size, size)
        self.vol = torch.zeros(size)
        self.relu = nn.ReLU()
        self.voltage_decay = 0.9
        self.surprise_threshold = 0.01
        self.hebian_rate = 0.1
        self.anti_hebian_rate = 0.01
        self.decay = 0.99

        # random gaussian initial weights
        self.weights = torch.randn(size, size)

    # prev_activation : Tensor(size)
    # for the first time prev activation is default 0
    # input : Tensor(size). this is a mask for pred. 
    # when no inputs it's default 1. cause assume just go along with the prediction
    def forward (self, input = None, prev_activation = None, learning = True):
        # starting input inference input
        if input is None:
            input = torch.ones(self.size)
        if prev_activation is None:
            prev_activation = torch.zeros(self.size)

        # predictive priming. 
        pred = self.relu(self.vol)
        activation = pred * input

        # inhibit voltage of all input neurons. 
        # later maybe do not FULL inhibt just inhibt a percentage of the voltage
        self.vol *= (1 - input)

        # surprise
        if activation.sum() < self.surprise_threshold * input.sum():
            activation = input * 1.0 #assume a default firing rate of 1.0

        # get post and update vol
        post = self.weights @ activation
        self.vol += post
        self.vol *= self.voltage_decay

        # hebian antihebian learn. then decay
        if learning:
            hebian= torch.outer(activation, prev_activation)
            #anti_hebian = torch.outer(activation, 1-prev_activation)
            #update = hebian * self.hebian_rate - anti_hebian * self.anti_hebian_rate
            update = hebian * self.hebian_rate
            self.weights += update

            # normalize weights 
            self.weights = self.weights.clamp(min=-1, max=1)

            self.weights *= self.decay

        #print(f"Voltage: {self.vol}")
        #print(f"Weights: {self.weights}")

        # return pred for debugging precision 
        return activation, pred