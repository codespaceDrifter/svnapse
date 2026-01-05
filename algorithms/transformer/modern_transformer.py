import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from ..basic.RMSnorm import RMSNorm
from ..attention.GQA import GQA
from ..basic.gatedMLP import GatedMLP
import torch.nn.functional as F

class ModernTransformerDecoder(nn.Module):
    def __init__(self, embed_dim, num_q_heads, num_kv_heads, mlp_dim, max_seq_len):
        super().__init__()
        self.attn_norm = RMSNorm(embed_dim)
        self.attn = GQA(embed_dim, num_q_heads, num_kv_heads, max_seq_len)
        self.mlp_norm = RMSNorm(embed_dim)
        self.mlp = GatedMLP(embed_dim, mlp_dim)

    def forward(self, x, positions):
        x = x + self.attn(self.attn_norm(x), positions)
        x = x + self.mlp(self.mlp_norm(x))
        return x

class ModernCausalLM(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_layers, num_q_heads, num_kv_heads, mlp_dim, max_seq_len,
                 use_checkpoint=True):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.layers = nn.ModuleList([
            ModernTransformerDecoder(embed_dim, num_q_heads, num_kv_heads, mlp_dim, max_seq_len)
            for _ in range(num_layers)
        ])
        self.norm = RMSNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

        # weight tying
        self.lm_head.weight = self.embed.weight

    def forward(self, input_ids, labels=None):
        batch, seq = input_ids.shape

        # positions for RoPE
        positions = torch.arange(seq, device=input_ids.device)

        x = self.embed(input_ids)

        for layer in self.layers:
            if self.use_checkpoint and self.training:
                # checkpoint: recompute this layer's activations during backward
                # use_reentrant=False is newer and recommended
                x = checkpoint(layer, x, positions, use_reentrant=False)
            else:
                x = layer(x, positions)

        x = self.norm(x)
        logits = self.lm_head(x)

        if labels is not None:
            # cross entropy: L = (1/N) * Σ -log(softmax(logits)[target])
            # logits[:, :-1]: (batch, seq-1, vocab) -> prediction logits for positions 0..seq-2
            # labels[:, 1:]: (batch, seq-1) -> target indices at positions 1..seq-1 (next tokens)
            # reshape flattens for cross_entropy:
            # (batch, seq-1, vocab) -> (batch*(seq-1), vocab)
            # (batch, seq-1) -> (batch*(seq-1),)
            loss = F.cross_entropy(
                logits[:, :-1, :].reshape(-1, logits.size(-1)),
                labels[:, 1:].reshape(-1)
            )
            return logits, loss

        return logits
