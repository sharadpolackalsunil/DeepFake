"""
Tokenizer components: PSIT, FrequencyStem, GatedFusion.
Converts spatial feature maps into token sequences for the ViT backbone.
"""
import torch
import torch.nn as nn


class PSIT(nn.Module):
    """Pyramidal Scale-Invariant Tokenizer.

    Adaptive-pools each FPN level to a fixed grid, sums across scales,
    and projects to a token sequence.
    """
    def __init__(self, embed_dim=256, grid=7):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((grid, grid))
        self.merge = nn.Conv2d(embed_dim, embed_dim, 1)

    def forward(self, fpn_feats):
        """
        Args:
            fpn_feats: list of (B, D, H_i, W_i) from FPN
        Returns:
            tokens: (B, grid*grid, D)
        """
        pooled = [self.pool(f) for f in fpn_feats]       # each (B, D, grid, grid)
        summed = torch.stack(pooled, dim=0).sum(0)        # (B, D, grid, grid)
        tokens = self.merge(summed).flatten(2).transpose(1, 2)  # (B, grid*grid, D)
        return tokens


class FrequencyStem(nn.Module):
    """Tokenizes wavelet branch output into a fixed-length token sequence."""
    def __init__(self, embed_dim=256, grid=7):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((grid, grid))

    def forward(self, wavelet_feat):
        """
        Args:
            wavelet_feat: (B, D, H/2, W/2) from WaveletBranch
        Returns:
            tokens: (B, grid*grid, D)
        """
        pooled = self.pool(wavelet_feat)
        return pooled.flatten(2).transpose(1, 2)  # (B, grid*grid, D)


class GatedFusion(nn.Module):
    """Gated cross-domain fusion of RGB and wavelet token streams.

    Uses a learned gate to blend the two modalities, then projects
    the fused representation through an output layer.
    """
    def __init__(self, embed_dim=256):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Sigmoid(),
        )
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, z_rgb, z_dwt):
        """
        Args:
            z_rgb: (B, N, D) RGB tokens
            z_dwt: (B, N, D) wavelet tokens
        Returns:
            fused: (B, N, D) fused tokens
        """
        cat = torch.cat([z_rgb, z_dwt], dim=-1)  # (B, N, 2D)
        g = self.gate(cat)                         # (B, N, D) ∈ [0, 1]
        fused = g * z_rgb + (1 - g) * z_dwt       # (B, N, D)
        return self.out_proj(fused)                # (B, N, D)
