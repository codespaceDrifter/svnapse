'''
GQA has multiple Q heads matched to one K head and V head. 
i.e. 32 Q heads, 8 KV heads, 4 KV groups. say embed coord 0-4 go to Q head 1, then embed coord 0-16 would go to K and V head 1. and Q head 1-4 would attend and scale by K head 1 and V head 1. 
this is for reduced KV cache. 
'''


import torch
import torch.nn as nn
import torch.nn.functional as F
from ..basic.RMSnorm import RMSNorm
from ..encoding.RoPE import RoPE

class GQA(nn.Module):
    def __init__(self, embed_dim=3072, num_q_heads=32, num_kv_heads=8, max_seq_len=4096):
        super().__init__()
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        assert embed_dim % num_q_heads == 0, "embed_dim must be divisible by num_q_heads"
        self.head_dim = embed_dim // num_q_heads
        self.num_kv_groups = num_q_heads // num_kv_heads  # 4
        
        self.scaling = self.head_dim ** -0.5  # 1/sqrt(head_dim)
        
        # Q: 32 heads * 96 = 3072
        # K: 8 heads * 96 = 768
        # V: 8 heads * 96 = 768
        # total: 4608
        qkv_size = num_q_heads * self.head_dim + 2 * num_kv_heads * self.head_dim
        self.qkv_proj = nn.Linear(embed_dim, qkv_size, bias=False)
        self.o_proj = nn.Linear(num_q_heads * self.head_dim, embed_dim, bias=False)
        
        self.rope = RoPE(self.head_dim, max_seq_len)

    def forward(self, x, positions, mask = None):
        batch, seq, _ = x.shape
        
        qkv = self.qkv_proj(x)
        
        q_size = self.num_q_heads * self.head_dim
        kv_size = self.num_kv_heads * self.head_dim
        
        q = qkv[..., :q_size].view(batch, seq, self.num_q_heads, self.head_dim).transpose(1, 2)
        k = qkv[..., q_size:q_size + kv_size].view(batch, seq, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = qkv[..., q_size + kv_size:].view(batch, seq, self.num_kv_heads, self.head_dim).transpose(1, 2)
        # q: (batch, num_q_heads, seq, head_dim)
        
        q = self.rope(q, positions)
        k = self.rope(k, positions)
        
        # reshape for GQA broadcast: 
        # q: (batch, num_kv_heads, num_kv_groups, seq, head_dim)
        # k: (batch, num_kv_heads, 1,             seq, head_dim)
        q = q.view(batch, self.num_kv_heads, self.num_kv_groups, seq, self.head_dim)
        k = k.unsqueeze(2)
        v = v.unsqueeze(2)
        
        scores = (q @ k.transpose(-2, -1)) * self.scaling
        if mask is not None:
            scores = scores.masked_fill(mask, float('-inf'))
        # attn: (batch, num_kv_heads, num_kv_groups, seq, seq)
        attn = F.softmax(scores, dim=-1)
        
        # out: (batch, num_kv_heads, num_kv_groups, seq, head_dim)
        out = attn @ v
        out = out.view(batch, self.num_q_heads, seq, self.head_dim)
        # out: (batch, seq, embed_dim)
        out = out.transpose(1, 2).reshape(batch, seq, -1)
        return self.o_proj(out) 