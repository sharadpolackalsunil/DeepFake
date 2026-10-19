"""Tokenizer components: DensePSIT, PSIT, FrequencyStem, GatedFusion.

Converts multi-scale spatial and frequency feature maps into token sequences
without lossy uniform average-pooling blur.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DensePSIT(nn.Module):
    """Multi-scale tokens on a 14x14 grid with learned strided convolutions.
    
    Replaces the 7x7 uniform box blur with learned feature downsampling and nearest upsampling:
    P2 (28x28) -> Conv 3x3 s=2 -> 14x14
    P3 (14x14) -> 14x14
    P4 (7x7)   -> Up 2x        -> 14x14
    Combined -> Conv 1x1       -> 196 tokens of dimension D
    """
    def __init__(self, embed_dim=256):
        super().__init__()
        self.down2 = nn.Conv2d(embed_dim, embed_dim, 3, stride=2, padding=1)  # 28 -> 14
        self.merge = nn.Conv2d(embed_dim, embed_dim, 1)

    def forward(self, fpn_feats):
        """
        Args:
            fpn_feats: list of [P2, P3, P4] from FPN (each with channels D)
        Returns:
            tokens: (B, 196, D)
        """
        p2, p3, p4 = fpn_feats
        up4 = F.interpolate(p4, size=p3.shape[-2:], mode="nearest")
        z = self.merge(self.down2(p2) + p3 + up4)              # (B, D, 14, 14)
        return z.flatten(2).transpose(1, 2)                    # (B, 196, D)


class PSIT(nn.Module):
    """Legacy 7x7 average-pooling tokenizer (kept for backward compatibility & ablations)."""
    def __init__(self, embed_dim=256, grid=7):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((grid, grid))
        self.merge = nn.Conv2d(embed_dim, embed_dim, 1)

    def forward(self, fpn_feats):
        pooled = [self.pool(f) for f in fpn_feats]
        summed = torch.stack(pooled, dim=0).sum(0)
        return self.merge(summed).flatten(2).transpose(1, 2)


class FrequencyStem(nn.Module):
    """Flattens 2D frequency feature maps (B, D, grid, grid) -> (B, grid*grid, D)."""
    def __init__(self, embed_dim=256, grid=14):
        super().__init__()
        self.grid = grid

    def forward(self, wavelet_feat):
        # wavelet_feat is (B, D, grid, grid) from WaveletBranch
        if wavelet_feat.shape[-1] != self.grid or wavelet_feat.shape[-2] != self.grid:
            wavelet_feat = F.interpolate(wavelet_feat, size=(self.grid, self.grid), mode="bilinear", align_corners=False)
        return wavelet_feat.flatten(2).transpose(1, 2)  # (B, grid*grid, D)


class GatedFusion(nn.Module):
    """Gated cross-domain fusion of RGB and wavelet token streams."""
    def __init__(self, embed_dim=256):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Sigmoid(),
        )
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, z_rgb, z_dwt):
        cat = torch.cat([z_rgb, z_dwt], dim=-1)  # (B, N, 2D)
        g = self.gate(cat)                         # (B, N, D) ∈ [0, 1]
        fused = g * z_rgb + (1.0 - g) * z_dwt     # (B, N, D)
        return self.out_proj(fused)                # (B, N, D)
