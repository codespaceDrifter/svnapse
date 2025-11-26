'''
gated mlp has double the parameters in the first half. one gate and one up. chunked together for parallelism. 
first half function: 
hidden = sigmoid(gate) * gate * up
think of it as a AND relationship using multiplication. or a input dependent magnitude scaling of the up. 
'''


import torch
import torch.nn as nn
import torch.nn.functional as F

class GatedMLP(nn.Module):
    def __init__(self, hidden_size=3072, intermediate_size=8192):
        super().__init__()
        # one big linear that we'll chunk into gate and up
        self.gate_up_proj = nn.Linear(hidden_size, 2 * intermediate_size)
        self.down_proj = nn.Linear(intermediate_size, hidden_size)
    
    def forward(self, x):
        # x: (batch, seq, 3072)
        
        gate_up = self.gate_up_proj(x)
        # (batch, seq, 16384)
        
        gate, up = gate_up.chunk(2, dim=-1)
        # gate: (batch, seq, 8192)
        # up: (batch, seq, 8192)
        
        # SiLU(gate) * up - this is the gating
        hidden = F.silu(gate) * up
        # (batch, seq, 8192)
        
        out = self.down_proj(hidden)
        # (batch, seq, 3072)
        
        return out


