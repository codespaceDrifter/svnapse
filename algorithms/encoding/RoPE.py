'''
RoPE (Rotary Positional Encoding) is a positional encoding used on the Q and K vectors per layer.
smaller embed coordinate rotate faster and larger sentence pos rotates more 
the formula is for angle to rotate by is: 
theta_p,d = p * 1.0 / (base ** (2d/D)) 
we rotate by dimension pairs so each even dimension is the x and each odd dimension is the y. we rotate with the cos and sin rotation matrix.
a benefit of this is dot product only sees the difference in angles. at the same embed coordinate pair, the angle difference between pos 100 and 98 is the same difference as pos 5 and 3. 
Qrot dot Krot = |Q||K|cos(theta_original + a - b)
theta_original is the unrotated angle between Q and K. a is angle of how Q is rotated and b is angle of how K is rotated.
a = pos_a * inv_freq, b = pos_b * inv_freq, so a - b = (pos_a - pos_b) * inv_freq. so it is only determined by the difference in positions.

earlier coordinates have larger inv_freq so rotate faster, more useful for earlier positions  
however at later positions it rotates too many circles and becomes indistinguishable for the model so we rely on latter coordinates with smaller inv_freq to maintain distinctiveness.
'''

import torch
import torch.nn as nn

class RoPE(nn.Module):
    def __init__(self, dim, max_seq_len=8192, base=10000):
        super().__init__()
        # one freq per dimension pair
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
    
    def forward(self, q, positions):
        theta = torch.outer(positions.float(), self.inv_freq)
        cos = theta.cos().to(q.dtype)
        sin = theta.sin().to(q.dtype)
        
        q1, q2 = q[..., ::2], q[..., 1::2]
        q1_rot = q1 * cos - q2 * sin
        q2_rot = q1 * sin + q2 * cos
        # q1_rot: (batch, seq, dim/2) q2_rot: (batch, seq, dim/2)
        # stacked: (batch, seq, dim/2, 2)
        # flattened: (batch, seq, dim)
        return torch.stack([q1_rot, q2_rot], dim=-1).flatten(-2)
