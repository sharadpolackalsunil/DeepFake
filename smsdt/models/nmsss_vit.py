import torch
import torch.nn as nn
from einops import rearrange

class TransformerBlock(nn.Module):
    def __init__(self, dim=256, heads=8, mlp_ratio=4.0, drop=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=drop, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim), nn.Dropout(drop),
        )

    def forward(self, x):
        h = self.norm1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        x = x + self.mlp(self.norm2(x))
        return x

class NMSSSBlock(nn.Module):
    """One spatial-attn -> temporal-diff -> temporal-attn stage."""
    def __init__(self, dim=256, heads=8):
        super().__init__()
        self.spatial = TransformerBlock(dim, heads)
        self.diff_proj = nn.Conv1d(dim * 2, dim, 1)   # applied per-token across the concat([f_t, f_t - f_{t-1}])
        self.temporal = TransformerBlock(dim, heads)

    def forward(self, x, conf):
        """
        x:    (B, T, N, D)  fused tokens per frame
        conf: (B, T)        SCRFD landmark confidence per frame
        """
        B, T, N, D = x.shape

        # --- Spatial self-attention: intra-frame ---
        x = rearrange(x, "b t n d -> (b t) n d")
        x = self.spatial(x)
        x = rearrange(x, "(b t) n d -> b t n d", b=B, t=T)

        # --- L2-normalized temporal subtraction (N-MSSS) ---
        x_norm = nn.functional.normalize(x, dim=-1)
        prev = torch.cat([x_norm[:, :1], x_norm[:, :-1]], dim=1)   # shift; first frame diffs against itself
        delta = x_norm - prev                                       # (B, T, N, D)
        cat = torch.cat([x_norm, delta], dim=-1)                    # (B, T, N, 2D)
        cat = rearrange(cat, "b t n d2 -> (b n) d2 t")
        gated = self.diff_proj(cat)                                  # (B*N, D, T)
        gated = rearrange(gated, "(b n) d t -> b t n d", b=B, n=N)
        conf_w = conf.view(B, T, 1, 1)
        x = x + gated * conf_w   # residual, gated by detector confidence so shaky-track frames are down-weighted

        # --- Temporal self-attention: inter-frame, per spatial location ---
        x = rearrange(x, "b t n d -> (b n) t d")
        x = self.temporal(x)
        x = rearrange(x, "(b n) t d -> b t n d", b=B, n=N)
        return x
