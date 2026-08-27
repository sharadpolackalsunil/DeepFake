import torch
import torch.nn as nn
from einops import rearrange
from .rgb_branch import RGBBranch
from .wavelet_branch import WaveletBranch
from .tokenizer import PSIT, FrequencyStem, GatedFusion
from .nmsss_vit import NMSSSBlock

class SMSDT(nn.Module):
    def __init__(self, embed_dim=256, depth=4, heads=8, num_classes=1):
        super().__init__()
        self.rgb_branch = RGBBranch(out_channels=embed_dim)
        self.wavelet_branch = WaveletBranch(embed_dim=embed_dim)
        self.psit = PSIT(embed_dim=embed_dim)
        self.freq_stem = FrequencyStem(embed_dim=embed_dim)
        self.fusion = GatedFusion(embed_dim=embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, 1, 49, embed_dim))
        self.blocks = nn.ModuleList([NMSSSBlock(embed_dim, heads) for _ in range(depth)])
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 1, embed_dim))
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim), nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(), nn.Linear(embed_dim // 2, num_classes),
        )

    def forward(self, frames, conf):
        """
        frames: (B, T, 3, 224, 224)
        conf:   (B, T)
        returns: (B,) chunk logit
        """
        B, T = frames.shape[:2]
        x = rearrange(frames, "b t c h w -> (b t) c h w")

        fpn = self.rgb_branch(x)                  # list of (B*T, D, h, w)
        z_rgb = self.psit(fpn)                     # (B*T, 49, D)

        wv = self.wavelet_branch(x)                # (B*T, D, H/2, W/2)
        z_dwt = self.freq_stem(wv)                  # (B*T, 49, D)

        z = self.fusion(z_rgb, z_dwt)               # (B*T, 49, D)
        z = rearrange(z, "(b t) n d -> b t n d", b=B, t=T)
        z = z + self.pos_embed

        for blk in self.blocks:
            z = blk(z, conf)

        pooled = z.mean(dim=2)          # (B, T, D) — collapse spatial tokens
        pooled = pooled.mean(dim=1)     # (B, D)   — collapse temporal (or use a learned temporal CLS token instead)
        logit = self.head(pooled).squeeze(-1)
        return logit
