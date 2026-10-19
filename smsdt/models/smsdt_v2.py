"""S-MSDTv2: Spatio-Temporal Video Transformer on Foundation CLIP ViT-L/14 Tokens.

Incorporates LayerNorm-only Parameter-Efficient Fine-Tuning (PEFT), dense 16x16 patch
tokens, factorized spatial-temporal attention with TemporalDiff, and dual-head readout
(EvidenceReadout patch anomaly top-k + normalized CLS token head).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .nmsss_vit import NMSSSBlock
from .smsdt import EvidenceReadout


class SMSDTv2(nn.Module):
    """S-MSDTv2 Spatio-Temporal Foundation Architecture.

    Args:
        encoder: CLIPFrameEncoder (outputs CLS (BT, 1024) and patches (BT, 256, 1024))
        dim: latent embedding dimension for spatio-temporal transformer (default 384)
        depth: number of NMSSSBlock layers (default 2)
        heads: attention heads (default 6, 64-dim per head)
        max_frames: maximum anticipated frames for temporal positional embedding (default 16)
        grid: patch spatial grid size (default 16 for 224/14)
        topk_frac: fraction of top-k patches for EvidenceReadout (default 0.05)
    """

    def __init__(
        self,
        encoder: nn.Module,
        dim: int = 384,
        depth: int = 2,
        heads: int = 6,
        max_frames: int = 16,
        grid: int = 16,
        topk_frac: float = 0.05,
    ):
        super().__init__()
        self.encoder = encoder
        self.dim = dim
        self.grid = grid
        self.max_frames = max_frames

        # Projection from CLIP 1024-dim hidden state to S-MSDT latent dim
        self.proj = nn.Sequential(
            nn.LayerNorm(1024),
            nn.Linear(1024, dim),
        )

        # Positional embeddings
        self.space = nn.Parameter(torch.zeros(1, 1, grid * grid, dim))
        self.time = nn.Parameter(torch.zeros(1, max_frames, 1, dim))
        nn.init.trunc_normal_(self.space, std=0.02)
        nn.init.trunc_normal_(self.time, std=0.02)

        # Spatio-temporal transformer blocks
        self.blocks = nn.ModuleList([NMSSSBlock(dim=dim, heads=heads) for _ in range(depth)])

        # Dual readout: patch EvidenceReadout + global CLS projection
        self.readout = EvidenceReadout(dim=dim, topk_frac=topk_frac)
        self.cls_head = nn.Linear(1024, 1)

        # Learnable balance between patch evidence and CLS head (sigmoid(0) = 0.5)
        self.beta = nn.Parameter(torch.zeros(()))

    def set_gradient_checkpointing(self, enable: bool = True):
        for blk in self.blocks:
            blk.set_gradient_checkpointing(enable)

    def forward(self, frames: torch.Tensor, conf: torch.Tensor | None = None):
        """
        Args:
            frames: (B, T, 3, 224, 224) RGB tensor in CLIP normalization
            conf: optional (B, T) confidence tensor (unused, kept for signature compatibility)
        Returns:
            logit: (B,) clip-level prediction logit
            patch_logits: (B, T, grid*grid) localized patch anomaly logits
        """
        B, T = frames.shape[:2]

        # 1. Foundation frame encoder
        cls, patches = self.encoder(rearrange(frames, "b t c h w -> (b t) c h w"))

        # 2. Project patches to transformer dimension and add position encodings
        z = rearrange(self.proj(patches), "(b t) n d -> b t n d", b=B, t=T)

        if T <= self.time.shape[1]:
            time_embed = self.time[:, :T]
        else:
            time_embed = F.interpolate(
                self.time.permute(0, 3, 1, 2), size=(T, 1), mode="linear", align_corners=False
            ).permute(0, 2, 3, 1)

        z = z + self.space + time_embed

        # 3. Spatio-temporal transformer blocks
        for blk in self.blocks:
            z = blk(z)

        # 4. Patch-level evidence readout
        logit_tok, patch_logits = self.readout(z)

        # 5. Global CLS head
        cls_norm = F.normalize(cls, dim=-1).view(B, T, -1).mean(dim=1)
        logit_cls = self.cls_head(cls_norm).squeeze(-1)

        # 6. Balanced combination
        w = torch.sigmoid(self.beta)
        logit = w * logit_tok + (1.0 - w) * logit_cls

        return logit, patch_logits

    def param_groups(
        self,
        lr_backbone_ln: float = 1e-4,
        lr_head: float = 5e-4,
        weight_decay: float = 0.05,
    ):
        """Build parameter groups distinguishing CLIP LayerNorms from newly initialized head."""
        groups = {
            ("ln", True): [],
            ("ln", False): [],
            ("head", True): [],
            ("head", False): [],
        }

        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            is_ln = "encoder" in n and "norm" in n.lower()
            no_decay = p.ndim == 1 or n.endswith(".bias") or "norm" in n.lower() or "embed" in n or "space" in n or "time" in n

            key = ("ln" if is_ln else "head", not no_decay)
            groups[key].append(p)

        out = []
        for (category, decay), ps in groups.items():
            if ps:
                out.append({
                    "params": ps,
                    "lr": lr_backbone_ln if category == "ln" else lr_head,
                    "weight_decay": weight_decay if decay else 0.0,
                })
        return out
