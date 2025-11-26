# stebian neuron population
import torch
import torch.nn as nn

# neuron count: size 
# weight count: size * size 
# each column is a neuron each row is a synapse
# weight [i,j] can be interpreted as the weight from neuron j to neuron i
class SpikingStebPop:
    def __init__ (self, size):
        self.size = size
        self.weights = torch.Tensor(size, size)
        self.vol = torch.zeros(size)
        self.voltage_decay = 0.9
        self.surprise_threshold = 0.01
        self.hebian_rate = 0.1
        self.anti_hebian_rate = 0.01
        self.decay = 0.99

        # random gaussian initial weights
        self.weights = torch.randn(size, size)

    # prev_activation : Tensor(size)
    # for the first time prev activation is default 0
    # input : Tensor(size). 
    # when no inputs it's default 1. cause assume just go along with the prediction
    def forward (self, input = None, prev_activation = None, learning = True):
        # starting input inference input
        if input is None:
            input = torch.ones(self.size)
        if prev_activation is None:
            prev_activation = torch.zeros(self.size)

        # predictive priming. 
        pred = (self.vol > 0).float()
        activation = pred * input

        # inhibit voltage of all input neurons. 
        self.vol *= (1 - input)

        # surprise
        if activation.sum() < self.surprise_threshold * input.sum():
            activation = input

        # get post and update vol
        post = self.weights @ activation
        self.vol += post
        self.vol *= self.voltage_decay

        # hebian antihebian learn. then decay
        if learning:
            hebian= torch.outer(activation, prev_activation)
            anti_hebian = torch.outer(activation, 1-prev_activation)
            update = hebian * self.hebian_rate - anti_hebian * self.anti_hebian_rate
            self.weights += update
            self.weights *= self.decay

        #print(f"Voltage: {self.vol}")
        #print(f"Weights: {self.weights}")

        # return pred for debugging precision 
        return activation, pred