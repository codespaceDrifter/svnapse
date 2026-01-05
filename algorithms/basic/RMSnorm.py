'''
RMSnorm scales by sqrt(n) and then multiplies by a learnable weight. 
RMS is applied at the token level across the embed_dim. the total would be sqrt(n)*gamma rather than normal norm would be 1. 
without gamma the mean(x_norm^2) = 1
RMS(x) = sqrt (1/n * sum(x^2)) * gamma
'''


import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        # x: (batch, seq, dim)
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        x_norm = x / rms
        return x_norm * self.weight


