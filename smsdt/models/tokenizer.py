"""
S-MSDT v2 High-Resolution Tokenizer Components
=============================================
- High-Res PSIT (14x14 = 196 tokens) from multi-scale FPN features.
- FrequencyStem to project wavelet features to (B, 196, D).
- Compression-Aware Spectral Gating (CASG) & Local Cross-Attention in GatedFusion.
"""
import torch
import torch.nn as nn


class PSIT(nn.Module):
    """Pyramidal Scale-Invariant Tokenizer (PSIT).

    Adaptive-pools each FPN level to a 14x14 grid (196 tokens), sums across scales,
    and projects to a high-resolution token sequence.
    """
    def __init__(self, embed_dim=256, grid=14):
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
    def __init__(self, embed_dim=256, grid=14):
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
    """Compression-Aware Spectral Gating (CASG) & Local Cross-Domain Fusion.

    Wavelet tokens query RGB tokens via local cross-attention, followed by a
    spectral ratio (rho_t) conditioned gating network to adaptively blend
    spatial and frequency features.
    """
    def __init__(self, embed_dim=256, use_cross_attn=True):
        super().__init__()
        self.use_cross_attn = use_cross_attn
        if use_cross_attn:
            self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads=4, batch_first=True)
            self.norm_dwt = nn.LayerNorm(embed_dim)
            self.norm_rgb = nn.LayerNorm(embed_dim)

        # Gate input: [z_rgb, z_dwt_refined, rho] -> (D + D + 1) = 2D + 1
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

        if self.use_cross_attn:
            # Local cross-attention: wavelet queries RGB
            q = self.norm_dwt(z_dwt)
            k = v = self.norm_rgb(z_rgb)
            attn_out, _ = self.cross_attn(q, k, v, need_weights=False)
            z_dwt_refined = z_dwt + attn_out
        else:
            z_dwt_refined = z_dwt

        # Prepare gate input with spectral energy ratio rho
        if rho is not None:
            if rho.dim() == 2:
                rho_exp = rho.unsqueeze(1).expand(-1, N, -1)  # (B, N, 1)
            else:
                rho_exp = rho.expand(-1, N, -1)
            gate_in = torch.cat([z_rgb, z_dwt_refined, rho_exp], dim=-1)
        else:
            dummy_rho = torch.zeros(B, N, 1, device=z_rgb.device, dtype=z_rgb.dtype)
            gate_in = torch.cat([z_rgb, z_dwt_refined, dummy_rho], dim=-1)

        g = self.gate(gate_in)                         # (B, N, D) in [0, 1]
        fused = g * z_rgb + (1 - g) * z_dwt_refined   # (B, N, D)
        return self.out_proj(fused)                    # (B, N, D)
