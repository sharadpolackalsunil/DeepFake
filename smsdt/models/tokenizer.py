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

    Adaptive-pools each FPN level to a fixed grid, sums across scales,
    and projects to a token sequence.
    """
    def __init__(self, embed_dim=256, grid=7):
        super().__init__()
        self.grid = grid
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
