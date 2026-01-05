'''
gated mlp has double the parameters in the first half. one gate and one up. chunked together for parallelism.
first half function:
hidden = sigmoid(gate) * gate * up
think of it like the gate part deciding IF and the up part deciding WHAT. they can target DIFFERENT PARTS OF THE ACTIVATION.
for example, say a two dimensional vector, i want to pass coordinate_2 through IF coordinate_1 is positive:
in a traditional single no gate MLP this would be impossible:
[0,1] : misses coordinate_1 conditional. [1,0]: misses coordinate_2 value. [1,1]: mixes values of coordinate_1 and coordinate_2.
but in swiglu:
gate: [1 , 0]. up : [0,1]. this passes through value of coordinate_2 IF coordinate_1 is positive.

it is sigmoid(gate) * gate * up rather than sigmoid (gate) * up due to the potential for gate to amplify the up.  
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


