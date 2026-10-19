import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from einops import rearrange


class TransformerBlock(nn.Module):
    def __init__(self, dim=256, heads=8, mlp_ratio=4.0, drop=0.1):
        super().__init__()
        self.gradient_checkpointing = False
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=drop, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(drop),
        )

    def _forward(self, x):
        h = self.norm1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        x = x + self.mlp(self.norm2(x))
        return x

    def forward(self, x):
        if self.gradient_checkpointing and self.training and x.requires_grad:
            return checkpoint(self._forward, x, use_reentrant=False)
        return self._forward(x)


class TemporalDiff(nn.Module):
    """x: (B, T, N, D) -> x + W([x_hat, delta]), delta_0 = 0, zero-init identity start.
    
    Replicate padding ensures frame 0 receives delta_0 = 0 instead of acting as an accidental
    first-frame marker. Linear projection is zero-initialized so the block starts as an identity.
    """
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Linear(2 * dim, dim)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        # x is (B, T, N, D)
        x_hat = F.normalize(self.norm(x), dim=-1)
        prev = torch.cat([x_hat[:, :1], x_hat[:, :-1]], dim=1)  # replicate pad -> delta_0 = 0
        delta = x_hat - prev                                     # (B, T, N, D)
        return x + self.proj(torch.cat([x_hat, delta], dim=-1))  # (B, T, N, D)


class NMSSSBlock(nn.Module):
    """One spatial-attn -> temporal-diff -> temporal-attn stage."""
    def __init__(self, dim=256, heads=8):
        super().__init__()
        self.gradient_checkpointing = False
        self.spatial = TransformerBlock(dim, heads)
        self.temporal_diff = TemporalDiff(dim)
        self.temporal = TransformerBlock(dim, heads)

    def set_gradient_checkpointing(self, enable=True):
        self.gradient_checkpointing = enable
        self.spatial.gradient_checkpointing = enable
        self.temporal.gradient_checkpointing = enable

    def forward(self, x, conf=None):
        """
        x:    (B, T, N, D)  fused tokens per frame
        conf: (B, T)        optional landmark confidence (kept for signature compatibility)
        """
        B, T, N, D = x.shape

        if self.gradient_checkpointing:
            self.spatial.gradient_checkpointing = True
            self.temporal.gradient_checkpointing = True

        # --- 1. Spatial self-attention: intra-frame ---
        x = rearrange(x, "b t n d -> (b t) n d")
        x = self.spatial(x)
        x = rearrange(x, "(b t) n d -> b t n d", b=B, t=T)

        # --- 2. Temporal difference update (replicate pad + identity init) ---
        x = self.temporal_diff(x)

        # --- 3. Temporal self-attention: inter-frame, per spatial location ---
        x = rearrange(x, "b t n d -> (b n) t d")
        x = self.temporal(x)
        x = rearrange(x, "(b n) t d -> b t n d", b=B, n=N)

        return x
