'''
FLASH ATTENTION WITH GQA - COMPLETE EXAMPLE WALKTHROUGH

================================================================================
PART 1: GQA SETUP
================================================================================

Input: x of shape (seq_len, embed_dim) = (4, 8)
Config: num_q_heads=4, num_kv_heads=2, head_dim=2

Weight shapes:
    W_q: (8, 8)   # embed_dim -> num_q_heads * head_dim
    W_k: (8, 4)   # embed_dim -> num_kv_heads * head_dim
    W_v: (8, 4)   # embed_dim -> num_kv_heads * head_dim

After projection and reshape to (num_heads, seq, head_dim):
    Q: (4, 4, 2)   # num_q_heads, seq, head_dim
    K: (2, 4, 2)   # num_kv_heads, seq, head_dim
    V: (2, 4, 2)   # num_kv_heads, seq, head_dim

GQA expansion - repeat KV heads to match Q heads:
    K: (2, 4, 2) -> repeat_interleave(2, dim=0) -> (4, 4, 2)
    V: (2, 4, 2) -> repeat_interleave(2, dim=0) -> (4, 4, 2)

================================================================================
PART 2: FLASH ATTENTION ALGORITHM
================================================================================

For one head, chunk into seq_len into blocks of block_len (block_len=2):
    Q_blocks: [Q[0:2], Q[2:4]]   # two (2, 2) chunks
    K_blocks: [K[0:2], K[2:4]]
    V_blocks: [V[0:2], V[2:4]]

SRAM (persists through KV loop):
    Q_block: (2, 2)   # block_len, head_dim
    m:       (2,)     # running max per query, init -inf
    l:       (2,)     # running sum per query, init 0
    o:       (2, 2)   # running output, init 0

HBM -> SRAM each iteration (then discarded) we iterate over each block across the sequence:
    K_block: (2, 2)
    V_block: (2, 2)
    S:       (2, 2)   # attention scores

THE LOOP:

for kv_idx in range(num_kv_blocks):
    K_blk, V_blk = load from HBM                    # (2, 2) each
    S = Q_block @ K_blk.T / sqrt(head_dim)          # (2, 2)
    m_new = max(m, S.max(dim=-1))                   # (2,)
    rescale = exp(m - m_new)                        # (2,)
    l = l * rescale                                 # (2,)
    o = o * rescale[:, None]                        # (2, 2)
    weights = exp(S - m_new[:, None])               # (2, 2)
    l = l + weights.sum(dim=-1)                     # (2,)
    o = o + weights @ V_blk                         # (2, 2)
    m = m_new

output = o / l[:, None]   # (2, 2) - final normalization

================================================================================
PART 3: WHY RESCALING WORKS
================================================================================

At any point: 
o = sum_j exp(s_j - m) * v_j
l = sum_j exp(s_j - m)

When max changes from m_old to m_new:
    exp(s_j - m_new) = exp(s_j - m_old) * exp(m_old - m_new)

So multiply o and l by exp(m_old - m_new) to convert to new max.
Division by l happens once at the end.

================================================================================
PART 4: OUTPUT
================================================================================

After all heads complete:
    out: (num_heads, seq, head_dim) -> transpose -> (seq, num_heads, head_dim)
    out: reshape -> (seq, embed_dim)
    out = out @ W_o   # final projection, mixes heads
'''


import torch
import torch.nn as nn
import torch.nn.functional as F
from ..encoding.RoPE import RoPE

class GQA(nn.Module):
    _printed_backend = False  # class var to print only once

    def __init__(self, embed_dim, num_q_heads, num_kv_heads, max_seq_len):
        super().__init__()
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        assert embed_dim % num_q_heads == 0, "embed_dim must be divisible by num_q_heads"
        self.head_dim = embed_dim // num_q_heads
        self.num_kv_groups = num_q_heads // num_kv_heads
        
        # Q: 32 heads * 96 = 3072
        # K: 8 heads * 96 = 768
        # V: 8 heads * 96 = 768
        # total: 4608
        qkv_size = num_q_heads * self.head_dim + 2 * num_kv_heads * self.head_dim
        self.qkv_proj = nn.Linear(embed_dim, qkv_size, bias=False)
        self.o_proj = nn.Linear(num_q_heads * self.head_dim, embed_dim, bias=False)
        
        self.rope = RoPE(self.head_dim, max_seq_len)

    def forward(self, x, positions):
        batch, seq, _ = x.shape

        qkv = self.qkv_proj(x)

        q_size = self.num_q_heads * self.head_dim
        kv_size = self.num_kv_heads * self.head_dim

        # q: (batch, num_q_heads, seq, head_dim)
        # k, v: (batch, num_kv_heads, seq, head_dim)
        q = qkv[..., :q_size].view(batch, seq, self.num_q_heads, self.head_dim).transpose(1, 2)
        k = qkv[..., q_size:q_size + kv_size].view(batch, seq, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = qkv[..., q_size + kv_size:].view(batch, seq, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q = self.rope(q, positions)
        k = self.rope(k, positions)

        # GQA: expand K, V to match Q heads for flash attention
        # k, v: (batch, num_kv_heads, seq, head_dim) -> (batch, num_q_heads, seq, head_dim)
        k = k.repeat_interleave(self.num_kv_groups, dim=1)
        v = v.repeat_interleave(self.num_kv_groups, dim=1)

        # flash attention: (batch, num_q_heads, seq, head_dim) -> (batch, num_q_heads, seq, head_dim)
        # is_causal=True handles the mask internally (more efficient than passing mask)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        # print SDPA backend info once
        if not GQA._printed_backend:
            GQA._printed_backend = True
            print(f"SDPA backends - Flash: {torch.backends.cuda.flash_sdp_enabled()}, "
                  f"MemEfficient: {torch.backends.cuda.mem_efficient_sdp_enabled()}, "
                  f"Math: {torch.backends.cuda.math_sdp_enabled()}")
        # (batch, seq, embed_dim)
        out = out.transpose(1, 2).reshape(batch, seq, -1)
        return self.o_proj(out) 
