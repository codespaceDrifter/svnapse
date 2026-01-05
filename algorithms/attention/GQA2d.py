'''
GQA2d - Grouped Query Attention with 2D RoPE for Vision Transformers

2D RoPE splits head_dim in half, applying separate rotations for row and column positions.
This preserves spatial structure: attention naturally captures both vertical and horizontal
relative distances between patches.

    first half of head_dim: rotated by row position
    second half of head_dim: rotated by column position

Refer to GQA.py for grouped query attention and flash attention details.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..encoding.RoPE import RoPE


class GQA2d(nn.Module):
    def __init__(self, embed_dim, num_q_heads, num_kv_heads, grid_size):
        super().__init__()
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = embed_dim // num_q_heads
        self.num_kv_groups = num_q_heads // num_kv_heads
        
        qkv_size = num_q_heads * self.head_dim + 2 * num_kv_heads * self.head_dim
        self.qkv_proj = nn.Linear(embed_dim, qkv_size, bias=False)
        self.o_proj = nn.Linear(num_q_heads * self.head_dim, embed_dim, bias=False)
        
        # separate RoPE for row and col, each handles half the head_dim
        self.rope_row = RoPE(self.head_dim // 2, grid_size)
        self.rope_col = RoPE(self.head_dim // 2, grid_size)
    
    def apply_2d_rope(self, x, row_pos, col_pos):
        # x: (batch, num_heads, seq, head_dim)
        first_half = x[..., :self.head_dim // 2]   # (batch, num_heads, seq, head_dim/2)
        second_half = x[..., self.head_dim // 2:]  # (batch, num_heads, seq, head_dim/2)
        
        first_half = self.rope_row(first_half, row_pos)
        second_half = self.rope_col(second_half, col_pos)
        
        return torch.cat([first_half, second_half], dim=-1)
        # (batch, num_heads, seq, head_dim)
    
    def forward(self, x, row_pos, col_pos):
        # x: (batch, seq, embed_dim)
        # row_pos: (seq,) col_pos: (seq,)
        batch, seq, _ = x.shape
        
        qkv = self.qkv_proj(x)
        # (batch, seq, qkv_size)
        
        q_size = self.num_q_heads * self.head_dim
        kv_size = self.num_kv_heads * self.head_dim
        
        q = qkv[..., :q_size].view(batch, seq, self.num_q_heads, self.head_dim).transpose(1, 2)
        k = qkv[..., q_size:q_size + kv_size].view(batch, seq, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = qkv[..., q_size + kv_size:].view(batch, seq, self.num_kv_heads, self.head_dim).transpose(1, 2)
        # q: (batch, num_q_heads, seq, head_dim)
        # k, v: (batch, num_kv_heads, seq, head_dim)
        
        q = self.apply_2d_rope(q, row_pos, col_pos)
        k = self.apply_2d_rope(k, row_pos, col_pos)
        # (batch, num_heads, seq, head_dim)
        
        k = k.repeat_interleave(self.num_kv_groups, dim=1)
        v = v.repeat_interleave(self.num_kv_groups, dim=1)
        # (batch, num_q_heads, seq, head_dim)
        
        out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        # (batch, num_q_heads, seq, head_dim)
        # is_causal=False: patches attend bidirectionally
        
        out = out.transpose(1, 2).reshape(batch, seq, -1)
        # (batch, seq, embed_dim)
        
        return self.o_proj(out)
