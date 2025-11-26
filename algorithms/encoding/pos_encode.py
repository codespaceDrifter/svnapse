import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len=50000):
        super().__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Apply sin to even indices and cos to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension and register as buffer (not parameter)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # x shape: (batch, seq_len, d_model)
        # Add positional encoding to input
        return x + self.pe[:, :x.size(1),:]

"""

This explains positional encoding for a sequence length of 2 and embedding dimension of 4.

Initialize empty positional encoding matrix
pe = torch.zeros(max_seq_len, d_model)
Shape: [2, 4]
Values: [[0, 0, 0, 0],
         [0, 0, 0, 0]]
   
Create position vector (vertical column of positions)
position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
Shape: [2, 1]
Values: [[0],
         [1]]
   
Create frequency scaling factors (one for each pair of dimensions)
div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
This creates frequencies that decrease exponentially
torch.arange(0, 4, 2) gives [0, 2]
Multiplying by -log(10000)/4 ≈ -2.3/4 ≈ -0.575
exp() gives [e^0, e^(-1.15)] ≈ [1.0, 0.32]
Shape: [2]
Values: [1.0, 0.32]
   
When multiplying position * div_term, broadcasting gives:
[[0] * [1.0, 0.32]] = [[0*1.0, 0*0.32]] = [[0, 0]]
[[1] * [1.0, 0.32]] = [[1*1.0, 1*0.32]] = [[1.0, 0.32]]
Result shape: [2, 2]
   
Apply sin to even indices (0, 2)
pe[:, 0::2] = torch.sin(position * div_term)
sin(0) = 0, sin(0) = 0
sin(1.0) ≈ 0.84, sin(0.32) ≈ 0.31
PE now contains:
[[0,    ?,   0,    ?],
[0.84, ?, 0.31,  ?]]
   
Apply cos to odd indices (1, 3)
pe[:, 1::2] = torch.cos(position * div_term)
cos(0) = 1, cos(0) = 1
cos(1.0) ≈ 0.54, cos(0.32) ≈ 0.95
PE now contains:
[[0,    1,   0,    1],
[0.84, 0.54, 0.31, 0.95]]
   
Add batch dimension (for model compatibility)
pe = pe.unsqueeze(0)
Shape: [1, 2, 4]
Values: [[[0, 1, 0, 1],
          [0.84, 0.54, 0.31, 0.95]]]
   
"""