"""
S-MSDT v3 Tokenizer Components
==============================
- PSIT: Adaptive-pools each FPN level to a fixed grid, sums across scales.
- FrequencyStem: Tokenizes wavelet branch output into fixed-length token sequence.
- GatedFusion: ρ-conditioned gated fusion of RGB and wavelet token streams.

V3 changes from v2:
  - GatedFusion: removed cross-attention (~2M params saved), kept ρ conditioning.
  - Default grid=7 (49 tokens) instead of 14 (196 tokens).
"""
import torch
import torch.nn as nn


class PSIT(nn.Module):
    """Pyramidal Scale-Invariant Tokenizer (PSIT).

    Projects each FPN level to grid x grid using anti-aliased / strided
    convolutions for higher-resolution feature maps (avoiding destructive box blur),
    sums across scales with learned weighting, and projects to a token sequence.
    """
    def __init__(self, embed_dim=256, grid=7):
        super().__init__()
        self.grid = grid
        self.embed_dim = embed_dim

        # Level 0 (28x28 -> 7x7): two 3x3 depthwise-separable convs with stride 2
        self.down_lvl0 = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 3, stride=2, padding=1, groups=embed_dim, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim, 1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim, 3, stride=2, padding=1, groups=embed_dim, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim, 1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
        )
        # Level 1 (14x14 -> 7x7): one 3x3 depthwise-separable conv with stride 2
        self.down_lvl1 = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 3, stride=2, padding=1, groups=embed_dim, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim, 1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
        )
        # Level 2 (7x7 -> 7x7): 1x1 conv
        self.down_lvl2 = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
        )
        self.fallback_pool = nn.AdaptiveAvgPool2d((grid, grid))
        self.scale_weights = nn.Parameter(torch.ones(3))
        self.merge = nn.Conv2d(embed_dim, embed_dim, 1)

    def forward(self, fpn_feats):
        """
        Args:
            fpn_feats: list of (B, D, H_i, W_i) from FPN
        Returns:
            tokens: (B, grid*grid, D)
        """
        weights = torch.softmax(self.scale_weights, dim=0)
        downsampled = []
        for i, f in enumerate(fpn_feats):
            h, w = f.shape[-2:]
            if self.grid == 7 and i == 0 and h == 28 and w == 28:
                d = self.down_lvl0(f)
            elif self.grid == 7 and i == 1 and h == 14 and w == 14:
                d = self.down_lvl1(f)
            elif self.grid == 7 and i == 2 and h == 7 and w == 7:
                d = self.down_lvl2(f)
            else:
                d = self.fallback_pool(f)
            w_idx = min(i, len(weights) - 1)
            downsampled.append(d * weights[w_idx])

        summed = torch.stack(downsampled, dim=0).sum(0)        # (B, D, grid, grid)
        tokens = self.merge(summed).flatten(2).transpose(1, 2)  # (B, grid*grid, D)
        return tokens


class FrequencyStem(nn.Module):
    """Tokenizes wavelet branch output into a fixed-length token sequence."""
    def __init__(self, embed_dim=256, grid=7):
        super().__init__()
        self.grid = grid
        self.pool = nn.AdaptiveAvgPool2d((grid, grid))

    def forward(self, wavelet_feat):
        """
        Args:
            wavelet_feat: (B, D, H, W) from WaveletBranch
        Returns:
            tokens: (B, grid*grid, D)
        """
        pooled = self.pool(wavelet_feat)
        return pooled.flatten(2).transpose(1, 2)  # (B, grid*grid, D)


class GatedFusion(nn.Module):
    """ρ-Conditioned Gated Fusion (v3).

    Blends RGB and wavelet tokens with a gate conditioned on spectral
    energy ratio ρ — heavier gating toward frequency features when
    compression is light (high ρ = more HF energy preserved).

    V3: removed cross-attention from v2 (saves ~2M params, same effectiveness).
    Kept the deeper 2-layer gate with GELU from v2.
    """
    def __init__(self, embed_dim=256):
        super().__init__()
        # Gate input: [z_rgb, z_dwt, rho] -> (D + D + 1) = 2D + 1
        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 2 + 1, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.Sigmoid(),
        )
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, z_rgb, z_dwt, rho=None):
        """
        Args:
            z_rgb: (B, N, D) RGB tokens
            z_dwt: (B, N, D) wavelet tokens
            rho:   (B, 1) optional spectral energy ratio
        Returns:
            fused: (B, N, D) fused tokens
        """
        B, N, D = z_rgb.shape

        if rho is not None:
            if rho.dim() == 2:
                rho_exp = rho.unsqueeze(1).expand(-1, N, -1)  # (B, N, 1)
            else:
                rho_exp = rho.expand(-1, N, -1)
            gate_in = torch.cat([z_rgb, z_dwt, rho_exp], dim=-1)
        else:
            dummy_rho = torch.zeros(B, N, 1, device=z_rgb.device, dtype=z_rgb.dtype)
            gate_in = torch.cat([z_rgb, z_dwt, dummy_rho], dim=-1)

        g = self.gate(gate_in)                         # (B, N, D) in [0, 1]
        fused = g * z_rgb + (1 - g) * z_dwt           # (B, N, D)
        return self.out_proj(fused)                    # (B, N, D)
