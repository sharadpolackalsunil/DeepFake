"""
S-MSDT: Scale-Invariance + Multi-Scale Feature Differences + Wavelet Frequency Guard.
Full assembled model combining RGB branch, Wavelet branch, tokenizers, fusion, and N-MSSS ViT.
"""
import torch
import torch.nn as nn
from einops import rearrange
from .rgb_branch import RGBBranch
from .wavelet_branch import WaveletBranch
from .tokenizer import PSIT, FrequencyStem, GatedFusion
from .nmsss_vit import NMSSSBlock


class SMSDT(nn.Module):
    def __init__(self, embed_dim=256, depth=4, heads=8, num_classes=1, grid=7):
        super().__init__()
        self.rgb_branch = RGBBranch(out_channels=embed_dim)
        self.wavelet_branch = WaveletBranch(embed_dim=embed_dim)
        self.psit = PSIT(embed_dim=embed_dim, grid=grid)
        self.freq_stem = FrequencyStem(embed_dim=embed_dim, grid=grid)
        self.fusion = GatedFusion(embed_dim=embed_dim)

        num_tokens = grid * grid
        self.pos_embed = nn.Parameter(torch.zeros(1, 1, num_tokens, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.blocks = nn.ModuleList([
            NMSSSBlock(embed_dim, heads) for _ in range(depth)
        ])
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, num_classes),
        )

    def forward(self, frames, conf, return_features=False):
        """
        Args:
            frames: (B, T, 3, 224, 224) face crop chunks
            conf:   (B, T) SCRFD landmark confidence per frame
            return_features: if True, also return intermediate z_rgb, z_dwt
                            (needed for auxiliary losses during training)
        Returns:
            logit: (B,) chunk-level logit
            (optional) z_rgb_pooled: (B, D) pooled RGB features
            (optional) z_dwt_pooled: (B, D) pooled wavelet features
        """
        B, T = frames.shape[:2]
        x = rearrange(frames, "b t c h w -> (b t) c h w")

        # --- Dual-stream feature extraction ---
        fpn = self.rgb_branch(x)                    # list of (B*T, D, h, w)
        z_rgb = self.psit(fpn)                      # (B*T, N, D)

        wv = self.wavelet_branch(x)                 # (B*T, D, H/2, W/2)
        z_dwt = self.freq_stem(wv)                  # (B*T, N, D)

        # --- Cross-domain fusion ---
        z = self.fusion(z_rgb, z_dwt)               # (B*T, N, D)
        z = rearrange(z, "(b t) n d -> b t n d", b=B, t=T)
        z = z + self.pos_embed

        # --- N-MSSS decomposed ViT ---
        for blk in self.blocks:
            z = blk(z, conf)

        # --- Pooling and classification ---
        pooled = z.mean(dim=2)          # (B, T, D) — collapse spatial tokens
        pooled = pooled.mean(dim=1)     # (B, D)   — collapse temporal

        logit = self.head(pooled).squeeze(-1)  # (B,)

        if return_features:
            # Pool intermediate features for auxiliary losses
            z_rgb_r = rearrange(z_rgb, "(b t) n d -> b t n d", b=B, t=T)
            z_dwt_r = rearrange(z_dwt, "(b t) n d -> b t n d", b=B, t=T)
            z_rgb_pooled = z_rgb_r.mean(dim=2).mean(dim=1)  # (B, D)
            z_dwt_pooled = z_dwt_r.mean(dim=2).mean(dim=1)  # (B, D)
            return logit, z_rgb_pooled, z_dwt_pooled

        return logit
