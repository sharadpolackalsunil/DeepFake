"""
S-MSDT v3 N-MSSS ViT Backbone
=============================
- Spatial-First Execution Order (Spatial MHSA -> Temporal Diff -> Temporal MHSA -> FFN).
- Hyperspherical L2 Temporal Differencing (Forward diff at t=0, Backward diff at t>=1).
- Learned Detector-Uncertainty Gated Residuals from [conf_t, conf_prev].
- Asymmetric Temporal Self-Attention: Queries and Keys from temporal residuals,
  Values from un-subtracted spatial tokens (preserves static facial texture).

V3 changes from v2:
  - Spatial attention runs FIRST so temporal diffs operate on spatially-refined features
    (v2 ran temporal first on raw noisy tokens → positional noise dominated diffs).
  - Keeps AsymmetricTemporalAttention (good design, just needed correct position).
  - Keeps learned confidence gating (upgrade over v1's scalar multiplication).
"""
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from einops import rearrange


class AsymmetricTemporalAttention(nn.Module):
    """Inter-frame temporal attention where Q and K are projected from
    gated temporal differences (detecting inter-frame jitter), while V is
    projected from un-subtracted spatial tokens (preserving facial textures)."""
    def __init__(self, dim=256, heads=8, drop=0.1):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(drop)
        self.norm_diff = nn.LayerNorm(dim)
        self.norm_val = nn.LayerNorm(dim)

    def forward(self, x, delta_gated):
        """
        x:           (B*N, T, D) un-subtracted tokens
        delta_gated: (B*N, T, D) uncertainty-gated temporal differences
        """
        BN, T, D = x.shape
        H = self.heads
        d_k = D // H

        # Queries & Keys from temporal difference
        d_norm = self.norm_diff(delta_gated)
        q = self.q_proj(d_norm).view(BN, T, H, d_k).transpose(1, 2)  # (BN, H, T, d_k)
        k = self.k_proj(d_norm).view(BN, T, H, d_k).transpose(1, 2)  # (BN, H, T, d_k)

        # Values from un-subtracted tokens
        v_norm = self.norm_val(x)
        v = self.v_proj(v_norm).view(BN, T, H, d_k).transpose(1, 2)  # (BN, H, T, d_k)

        # Scaled dot-product attention across time T
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)                                  # (BN, H, T, d_k)
        out = out.transpose(1, 2).contiguous().view(BN, T, D)       # (BN, T, D)
        return self.out_proj(out)


class SpatialTransformerBlock(nn.Module):
    """Intra-frame spatial self-attention block across patch tokens."""
    def __init__(self, dim=256, heads=8, mlp_ratio=4.0, drop=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=drop, batch_first=True)

    def forward(self, x):
        h = self.norm1(x)
        out, _ = self.attn(h, h, h, need_weights=False)
        return x + out


class FeedForward(nn.Module):
    """Standard MLP / FFN with LayerNorm and GELU."""
    def __init__(self, dim=256, mlp_ratio=4.0, drop=0.1):
        super().__init__()
        mid = int(dim * mlp_ratio)
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mid),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mid, dim),
            nn.Dropout(drop),
        )

    def forward(self, x):
        return x + self.mlp(self.norm(x))


class NMSSSBlock(nn.Module):
    """S-MSDT v3 Decomposed ViT Block:
    Spatial-First with Asymmetric Temporal Attention.

    Execution order:
      1. Spatial MHSA (intra-frame across N patches)
      2. L2-normalized temporal differencing + learned confidence gating
      3. Asymmetric Temporal MHSA (Q/K from diffs, V from originals)
      4. FFN

    V3 fix: spatial runs FIRST so temporal diffs operate on spatially-refined
    features, not raw noisy tokens (the root cause of v2's failure).
    """
    def __init__(self, dim=256, heads=8, mlp_ratio=4.0, drop=0.1):
        super().__init__()
        self.dim = dim
        self.gradient_checkpointing = False

        # 1. Spatial Self-Attention (intra-frame across patches N)
        self.spatial = SpatialTransformerBlock(dim, heads=heads, drop=drop)

        # 2. Learned Uncertainty Gating MLP for detector confidence [c_t, c_prev]
        self.conf_gate = nn.Sequential(
            nn.Linear(2, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        # 3. Asymmetric Temporal Self-Attention (inter-frame across T)
        self.temporal = AsymmetricTemporalAttention(dim, heads=heads, drop=drop)

        # 4. Feedforward Network
        self.ffn = FeedForward(dim, mlp_ratio=mlp_ratio, drop=drop)

    def set_gradient_checkpointing(self, enable=True):
        self.gradient_checkpointing = enable

    def _forward_block(self, x, conf):
        B, T, N, D = x.shape

        # --- Step 1: Spatial Self-Attention FIRST (intra-frame across N patches) ---
        x_s = rearrange(x, "b t n d -> (b t) n d")
        s_out = self.spatial(x_s)
        x = rearrange(s_out, "(b t) n d -> b t n d", b=B, t=T)

        # --- Step 2: Hyperspherical L2 Temporal Differencing ---
        x_norm = nn.functional.normalize(x, dim=-1)  # (B, T, N, D)

        if T > 1:
            # Forward difference at t=0 (X_1 - X_0); backward difference at t>=1
            delta_0 = x_norm[:, 1:2] - x_norm[:, 0:1]       # (B, 1, N, D)
            delta_rest = x_norm[:, 1:] - x_norm[:, :-1]     # (B, T-1, N, D)
            delta = torch.cat([delta_0, delta_rest], dim=1) # (B, T, N, D)

            # Learned Detector Uncertainty Gating
            conf_prev = torch.cat([conf[:, 1:2], conf[:, :-1]], dim=1)  # (B, T)
            conf_pair = torch.stack([conf, conf_prev], dim=-1)           # (B, T, 2)
            g_rel = self.conf_gate(conf_pair).unsqueeze(2)               # (B, T, 1, 1)
        else:
            delta = torch.zeros_like(x_norm)
            g_rel = torch.ones(B, 1, 1, 1, device=x.device, dtype=x.dtype)

        delta_gated = delta * g_rel  # (B, T, N, D)

        # --- Step 3: Asymmetric Temporal Attention (inter-frame, per patch) ---
        x_t = rearrange(x, "b t n d -> (b n) t d")
        d_t = rearrange(delta_gated, "b t n d -> (b n) t d")
        t_out = self.temporal(x_t, d_t)
        x = x + rearrange(t_out, "(b n) t d -> b t n d", b=B, n=N)

        # --- Step 4: FFN ---
        x_ffn = rearrange(x, "b t n d -> (b t) n d")
        f_out = self.ffn(x_ffn)
        x = rearrange(f_out, "(b t) n d -> b t n d", b=B, t=T)

        return x

    def forward(self, x, conf):
        """
        x:    (B, T, N, D)  tokens per frame
        conf: (B, T)        SCRFD landmark confidence per frame
        """
        if self.gradient_checkpointing and self.training and x.requires_grad:
            return checkpoint(self._forward_block, x, conf, use_reentrant=False)
        return self._forward_block(x, conf)
