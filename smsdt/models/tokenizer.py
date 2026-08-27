import torch
import torch.nn as nn

class PSIT(nn.Module):
    """Pyramidal Scale-Invariant Tokenizer: adaptive-pool each FPN level to 7x7, then merge."""
    def __init__(self, embed_dim=256, grid=7):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((grid, grid))
        self.merge = nn.Conv2d(embed_dim, embed_dim, 1)

    def forward(self, fpn_feats):
        pooled = [self.pool(f) for f in fpn_feats]     # each (B, D, 7, 7)
        summed = torch.stack(pooled, dim=0).sum(0)
        tokens = self.merge(summed).flatten(2).transpose(1, 2)   # (B, 49, D)
        return tokens

class FrequencyStem(nn.Module):
    def __init__(self, embed_dim=256, grid=7):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((grid, grid))

    def forward(self, wavelet_feat):
        pooled = self.pool(wavelet_feat)
        return pooled.flatten(2).transpose(1, 2)   # (B, 49, D)

class GatedFusion(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(embed_dim * 2, embed_dim), nn.Sigmoid())
        self.out = nn.Linear(embed_dim * 2, embed_dim)

    def forward(self, z_rgb, z_dwt):
        cat = torch.cat([z_rgb, z_dwt], dim=-1)
        g = self.gate(cat)
        fused = g * z_rgb + (1 - g) * z_dwt
        # Return fused. In original snippet it had `if False else fused`.
        return fused
