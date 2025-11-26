import torch
import torch.nn as nn
from ..basic.RMSnorm import RMSNorm
from ..attention.GQA import GQA
from ..basic.gatedMLP import GatedMLP
import torch.nn.functional as F

class ModernTransformerDecoder(nn.Module):
    def __init__(self, embed_dim, num_q_heads, num_kv_heads, mlp_dim):
        super().__init__()
        self.attn_norm = RMSNorm(embed_dim)
        self.attn = GQA(embed_dim, num_q_heads, num_kv_heads)
        self.mlp_norm = RMSNorm(embed_dim)
        self.mlp = GatedMLP(embed_dim, mlp_dim)
    
    def forward(self, x, positions, mask):
        x = x + self.attn(self.attn_norm(x), positions, mask)
        x = x + self.mlp(self.mlp_norm(x))
        return x

class ModernCausalLM(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_layers, num_q_heads, num_kv_heads, mlp_dim, max_seq_len):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.layers = nn.ModuleList([
            ModernTransformerDecoder(embed_dim, num_q_heads, num_kv_heads, mlp_dim)
            for _ in range(num_layers)
        ])
        self.norm = RMSNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)
        
        # weight tying
        self.lm_head.weight = self.embed.weight
        
    def forward(self, input_ids, labels=None):
        batch, seq = input_ids.shape
        
        mask = torch.triu(torch.ones(seq, seq, device=input_ids.device), diagonal=1).bool()
        positions = torch.arange(seq, device=input_ids.device)
        
        x = self.embed(input_ids)
        
        for layer in self.layers:
            x = layer(x, positions, mask)
        
        x = self.norm(x)
        logits = self.lm_head(x)
        
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                labels[:, 1:].reshape(-1)
            )
            return logits, loss
        
        return logits