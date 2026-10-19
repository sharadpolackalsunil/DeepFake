"""S-MSDT: Scale-Invariant Multi-Scale Deepfake Transformer.

Dual-stream RGB + Wavelet feature extraction with DensePSIT (14x14 grid),
learnable spatial & temporal position embeddings, N-MSSS spatio-temporal ViT,
and EvidenceReadout (gated attention pooling + top-k patch anomaly logits).
"""
import torch
import torch.nn as nn
from einops import rearrange

from .rgb_branch import RGBBranch
from .wavelet_branch import WaveletBranch
from .tokenizer import DensePSIT, PSIT, FrequencyStem, GatedFusion
from .nmsss_vit import NMSSSBlock


class EvidenceReadout(nn.Module):
    """z: (B, T, N, D) -> clip logit (B,), patch logits (B, T, N).

    Combines gated attention pooling (Ilse et al., ICML 2018) with top-k patch
    logits to detect subtle localized forgery anomalies (such as mouth/boundary seams)
    without global averaging dilution or max-pooling gradient saturation.
    """

    def __init__(self, dim=256, topk_frac=0.05):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.att_v = nn.Linear(dim, 128)
        self.att_u = nn.Linear(dim, 128)
        self.att_w = nn.Linear(128, 1)
        self.patch_head = nn.Linear(dim, 1)
        self.head = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim // 2, 1),
        )
        self.mix = nn.Parameter(torch.zeros(()))  # sigmoid(0) = 0.5 balanced start
        self.topk_frac = topk_frac

    def forward(self, z):
        B, T, N, D = z.shape
        h = self.norm(z).reshape(B, T * N, D)

        # Gated attention pooling weights
        a = self.att_w(torch.tanh(self.att_v(h)) * torch.sigmoid(self.att_u(h)))  # (B, TN, 1)
        a = torch.softmax(a.float(), dim=1).to(h.dtype)
        pooled = (a * h).sum(1)                                                   # (B, D)

        # Patch-level anomaly logits
        patch = self.patch_head(h).squeeze(-1)                                    # (B, TN)
        k = max(1, int(self.topk_frac * T * N))
        top = patch.topk(k, dim=1).values.mean(1)                                 # (B,)

        w = torch.sigmoid(self.mix)
        logit = w * self.head(pooled).squeeze(-1) + (1.0 - w) * top
        return logit, patch.view(B, T, N)


class SMSDT(nn.Module):
    def __init__(self, embed_dim=256, depth=4, heads=8, num_classes=1, grid=14,
                 stem_type="strided_patch", pool_energy=False, max_frames=16,
                 use_dense_psit=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.grid = grid
        num_tokens = grid * grid

        self.rgb_branch = RGBBranch(out_channels=embed_dim)
        self.wavelet_branch = WaveletBranch(embed_dim=embed_dim, stem_type=stem_type,
                                            pool_energy=pool_energy, grid=grid)
        self.psit = DensePSIT(embed_dim=embed_dim) if (use_dense_psit and grid == 14) else PSIT(embed_dim=embed_dim, grid=grid)
        self.freq_stem = FrequencyStem(embed_dim=embed_dim, grid=grid)
        self.fusion = GatedFusion(embed_dim=embed_dim)

        # Spatial position embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, 1, num_tokens, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Temporal position embedding (resolves Doubt 3)
        self.time_embed = nn.Parameter(torch.zeros(1, max_frames, 1, embed_dim))
        nn.init.trunc_normal_(self.time_embed, std=0.02)

        self.blocks = nn.ModuleList([
            NMSSSBlock(embed_dim, heads) for _ in range(depth)
        ])

        self.readout = EvidenceReadout(dim=embed_dim, topk_frac=0.05)

    def set_gradient_checkpointing(self, enable=True):
        for blk in self.blocks:
            blk.set_gradient_checkpointing(enable)

    def forward(self, frames, conf=None, return_features=False):
        """
        Args:
            frames: (B, T, 3, 224, 224) face crop chunks
            conf:   (B, T) landmark confidence per frame (optional)
            return_features: if True, also return intermediate representations
        Returns:
            logit: (B,) chunk-level binary logit
            patch_logits: (B, T, N) patch-level binary logits
        """
        B, T = frames.shape[:2]
        x = rearrange(frames, "b t c h w -> (b t) c h w")

        # --- Dual-stream feature extraction ---
        fpn = self.rgb_branch(x)                    # list of (B*T, D, h, w)
        z_rgb = self.psit(fpn)                      # (B*T, N, D)

        wv = self.wavelet_branch(x)                 # (B*T, D, grid, grid)
        z_dwt = self.freq_stem(wv)                  # (B*T, N, D)

        # --- Cross-domain fusion ---
        z = self.fusion(z_rgb, z_dwt)               # (B*T, N, D)
        z = rearrange(z, "(b t) n d -> b t n d", b=B, t=T)

        # Add spatial and temporal position encodings
        z = z + self.pos_embed + self.time_embed[:, :T]

        # --- N-MSSS spatio-temporal transformer ---
        for blk in self.blocks:
            z = blk(z, conf)

        # --- Evidence Readout (Attention + Top-k Patch Evidence) ---
        logit, patch_logits = self.readout(z)

        if return_features:
            pooled = z.mean(dim=(1, 2))
            return logit, patch_logits, pooled

        return logit, patch_logits
