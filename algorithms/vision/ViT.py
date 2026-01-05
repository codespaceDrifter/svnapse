'''
Vision Transformer (ViT) with 2D RoPE

The image is divided into non-overlapping patches of fixed size (e.g. 14x14 pixels).
Each patch is flattened and linearly projected to the model's hidden dimension.

This projection is implemented as a Conv2d where kernel_size = stride = patch_size.
Mathematically equivalent to: for each patch, flatten(patch) @ W + b
The convolution executes this in parallel across all spatial positions.

2D RoPE encodes spatial position by splitting the head dimension in half:
    - first half: rotated by row position
    - second half: rotated by column position

Output: (batch, num_patches, visual_dim) contextualized visual tokens.
Projector maps visual_dim → embed_dim, then concatenated with text for the LLM.

Full VLM architecture:
    Image → ViT → Projector → concat with text embeddings → LLM → output
Two separate transformers: ViT processes vision, LLM processes the unified sequence.
'''
import torch
import torch.nn as nn
from ..attention.GQA2d import GQA2d
from ..basic.RMSnorm import RMSNorm
from ..basic.gatedMLP import GatedMLP


class PatchEmbedding(nn.Module):
    def __init__(self, img_size=448, patch_size=14, in_channels=3, visual_dim=1280):
        super().__init__()
        self.grid_size = img_size // patch_size  # 32
        self.num_patches = self.grid_size ** 2   # 1024
        self.proj = nn.Conv2d(in_channels, visual_dim, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x):
        # x: (batch, 3, 448, 448)
        
        x = self.proj(x)
        # (batch, 1280, 32, 32)
        
        x = x.flatten(2).transpose(1, 2)
        # (batch, 1024, 1280)
        
        return x


class ViTBlock(nn.Module):
    def __init__(self, visual_dim=1280, num_q_heads=16, num_kv_heads=16, mlp_dim=5120, grid_size=32):
        super().__init__()
        self.attn_norm = RMSNorm(visual_dim)
        self.attn = GQA2d(visual_dim, num_q_heads, num_kv_heads, grid_size)
        self.mlp_norm = RMSNorm(visual_dim)
        self.mlp = GatedMLP(visual_dim, mlp_dim)
    
    def forward(self, x, row_pos, col_pos):
        # x: (batch, 1024, 1280)
        # row_pos: (1024,) col_pos: (1024,)
        
        x = x + self.attn(self.attn_norm(x), row_pos, col_pos)
        x = x + self.mlp(self.mlp_norm(x))
        # (batch, 1024, 1280)
        
        return x


class ViT(nn.Module):
    def __init__(self, img_size=448, patch_size=14, in_channels=3, visual_dim=1280, 
                 num_layers=24, num_q_heads=16, num_kv_heads=16, mlp_dim=5120):
        super().__init__()
        grid_size = img_size // patch_size
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, visual_dim)
        self.blocks = nn.ModuleList([
            ViTBlock(visual_dim, num_q_heads, num_kv_heads, mlp_dim, grid_size) 
            for _ in range(num_layers)
        ])
        self.norm = RMSNorm(visual_dim)
        
        # precompute grid positions
        rows = torch.arange(grid_size).unsqueeze(1).expand(-1, grid_size).flatten()
        cols = torch.arange(grid_size).unsqueeze(0).expand(grid_size, -1).flatten()
        self.register_buffer('row_pos', rows)  # (1024,)
        self.register_buffer('col_pos', cols)  # (1024,)
    
    def forward(self, x):
        # x: (batch, 3, 448, 448)
        
        x = self.patch_embed(x)
        # (batch, 1024, 1280)
        
        for block in self.blocks:
            x = block(x, self.row_pos, self.col_pos)
        # (batch, 1024, 1280)
        
        x = self.norm(x)
        # (batch, 1024, 1280)
        
        return x


class VisionProjector(nn.Module):
    def __init__(self, visual_dim=1280, embed_dim=3584):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(visual_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )
    
    def forward(self, x):
        # x: (batch, 1024, 1280)
        
        x = self.proj(x)
        # (batch, 1024, 3584)
        
        return x


class ViT(nn.Module):
    def __init__(self, ViT, projector, llm):
        super().__init__()
        self.ViT = ViT
        self.projector = projector
        self.llm = llm
    
    def forward(self, image, text_embeds):
        # image: (batch, 3, 448, 448)
        # text_embeds: (batch, seq_len, embed_dim)
        
        visual_tokens = self.ViT(image)
        # (batch, 1024, visual_dim)
        
        visual_tokens = self.projector(visual_tokens)
        # (batch, 1024, embed_dim)
        
        combined = torch.cat([visual_tokens, text_embeds], dim=1)
        # (batch, 1024 + seq_len, embed_dim)
        
        output = self.llm(combined)
        
        return output
