"""S-MSDTv3: Spatio-Temporal Video Transformer with Dual-Stream High-Frequency Residual Injection,
Landmark Perioral Prior, and Multi-Query Log-Sum-Exp Evidence Readout.

Architectural highlights:
1. Pretrained Foundation Backbone: OpenAI CLIP ViT-L/14 with LayerNorm-only PEFT (<0.03% params).
2. ResidualStream: Multi-scale high-frequency residual extraction (3 SRM kernels + Level-1/2 Haar DWT)
   injected via a zero-initialized sigmoid gate.
3. Spatio-Temporal Modeling: 2 NMSSSBlock layers with replicate-padded TemporalDiff (Delta_0 = 0).
4. EvidenceReadoutV3: K=4 learned queries, landmark-guided perioral prior bias, and LSE pooling.
5. Dual Readout Fusion: Dynamic blending of patch-level anomaly logits and global CLS semantics.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .nmsss_vit import NMSSSBlock
from .v3_modules import ResidualStream, EvidenceReadoutV3
from .clip_encoder import CLIP_MEAN, CLIP_STD


class SMSDTv3(nn.Module):
    """S-MSDTv3 Foundation Video Architecture with High-Frequency Residual Injection.

    Args:
        encoder:     CLIPFrameEncoder foundation model
        dim:         latent feature dimension (default 384)
        depth:       number of NMSSSBlock spatio-temporal layers (default 2)
        heads:       attention heads (default 6, 64-dim per head)
        max_frames:  maximum temporal frames for positional embedding (default 16)
        grid:        spatial patch grid size (default 16 for 224/14)
        n_queries:   learned query count for EvidenceReadoutV3 (default 4)
        r0:          initial temperature for Log-Sum-Exp pooling (default 3.0)
        topk_frac:   top-k patch fraction (default 0.05)
        init_bias:   initial gate bias for ResidualStream (default -4.0, g ~ 0.018)
        zero_gate:   if True, sets gate bias to -10.0 for near-zero initialization (<1e-4)
    """

    def __init__(
        self,
        encoder: nn.Module,
        dim: int = 384,
        depth: int = 2,
        heads: int = 6,
        max_frames: int = 16,
        grid: int = 16,
        n_queries: int = 4,
        r0: float = 3.0,
        topk_frac: float = 0.05,
        init_bias: float = -4.0,
        zero_gate: bool = False,
    ):
        super().__init__()
        self.encoder = encoder
        self.dim = dim
        self.grid = grid
        self.max_frames = max_frames

        # Projection from CLIP 1024-dim hidden state to S-MSDT latent dim (384)
        self.proj = nn.Sequential(
            nn.LayerNorm(1024),
            nn.Linear(1024, dim),
        )

        # 2D Spatial & 1D Chronological Temporal Positional Embeddings
        self.space = nn.Parameter(torch.zeros(1, 1, grid * grid, dim))
        self.time = nn.Parameter(torch.zeros(1, max_frames, 1, dim))
        nn.init.trunc_normal_(self.space, std=0.02)
        nn.init.trunc_normal_(self.time, std=0.02)

        # High-frequency residual stream (SRM + Haar DWT)
        self.res = ResidualStream(
            dim=dim,
            c_r=128,
            grid=grid,
            init_bias=init_bias,
            zero_gate=zero_gate,
        )

        # Spatio-temporal transformer blocks with replicate-padded TemporalDiff
        self.blocks = nn.ModuleList([NMSSSBlock(dim=dim, heads=heads) for _ in range(depth)])

        # EvidenceReadoutV3: K-query attention + mouth prior + LSE pooling
        self.readout = EvidenceReadoutV3(
            dim=dim,
            heads=heads,
            n_queries=n_queries,
            r0=r0,
            topk_frac=topk_frac,
        )

        # Global scene CLS head (1024 -> 1)
        self.cls_head = nn.Linear(1024, 1)

        # Learnable balance between patch evidence and CLS head (sigmoid(0) = 0.5)
        self.beta = nn.Parameter(torch.zeros(()))

        # Denormalization constants for mapping CLIP frames to [0, 1] for ResidualStream
        self.register_buffer("clip_mean", torch.tensor(CLIP_MEAN).view(1, 3, 1, 1))
        self.register_buffer("clip_std", torch.tensor(CLIP_STD).view(1, 3, 1, 1))

    def set_gradient_checkpointing(self, enable: bool = True):
        for blk in self.blocks:
            blk.set_gradient_checkpointing(enable)

    def forward(
        self,
        frames: torch.Tensor,
        conf: torch.Tensor | None = None,
        prior: torch.Tensor | None = None,
    ):
        """
        Args:
            frames: (B, T, 3, 224, 224) RGB tensor in CLIP normalization
            conf:   optional (B, T) confidence tensor (unused, kept for interface compatibility)
            prior:  optional (B, T, 256) soft perioral mouth prior
        Returns:
            logit:        (B,) clip-level manipulation logit
            patch_logits: (B, T, 256) localized patch anomaly logits
            aux_logit:    (B,) auxiliary frequency prediction for deep supervision
        """
        B, T = frames.shape[:2]

        # Flatten batch and temporal dimensions for frame-level feature extraction
        x = rearrange(frames, "b t c h w -> (b t) c h w")

        # 1. Foundation frame encoder (CLS token + 256 spatial patch tokens)
        cls, patches = self.encoder(x)

        # 2. Linear projection of patch tokens to latent dimension
        s = self.proj(patches)  # (BT, 256, dim)

        # 3. High-frequency residual stream injection
        # Denormalize CLIP frames back to [0, 1] range for SRM and Haar wavelet processing
        x01 = (x * self.clip_std + self.clip_mean).clamp(0.0, 1.0)
        s, aux_frame_logits = self.res(x01, s)

        # 4. Inject spatial and temporal positional embeddings
        z = rearrange(s, "(b t) n d -> b t n d", b=B, t=T)
        if T <= self.time.shape[1]:
            time_embed = self.time[:, :T]
        else:
            time_embed = F.interpolate(
                self.time.permute(0, 3, 1, 2), size=(T, 1), mode="linear", align_corners=False
            ).permute(0, 2, 3, 1)

        z = z + self.space + time_embed

        # 5. Spatio-temporal transformer blocks with TemporalDiff
        for blk in self.blocks:
            z = blk(z)

        # 6. EvidenceReadoutV3: LSE patch anomaly + multi-query attention
        logit_tok, patch_logits = self.readout(z, prior=prior)

        # 7. Global CLS head
        cls_norm = F.normalize(cls, dim=-1).view(B, T, -1).mean(dim=1)
        logit_cls = self.cls_head(cls_norm).squeeze(-1)

        # 8. Dual readout balance
        w = torch.sigmoid(self.beta)
        logit = w * logit_tok + (1.0 - w) * logit_cls

        # Video-level auxiliary frequency prediction
        aux_logit = rearrange(aux_frame_logits, "(b t) -> b t", b=B, t=T).mean(dim=1)

        return logit, patch_logits, aux_logit

    def param_groups(
        self,
        lr_backbone_ln: float = 5e-6,
        lr_head: float = 1e-4,
        lr_new: float = 3e-4,
        weight_decay: float = 0.05,
    ):
        """Construct three decoupled parameter groups according to Stage C curriculum:
        1. CLIP LayerNorms (lr = 5e-6, wd = 0.0)
        2. Existing Head (proj, blocks, cls_head, space, time, beta) (lr = 1e-4, wd = 0.05)
        3. New V3 Modules (res, readout) (lr = 3e-4, wd = 0.05)
        
        Excludes scalars and biases from weight decay.
        """
        groups = {
            ("ln", True): [],
            ("ln", False): [],
            ("head", True): [],
            ("head", False): [],
            ("new", True): [],
            ("new", False): [],
        }

        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue

            # Identify category
            is_ln = "encoder" in n and "norm" in n.lower()
            is_new = n.startswith("res.") or n.startswith("readout.")

            if is_ln:
                category = "ln"
            elif is_new:
                category = "new"
            else:
                category = "head"

            # Check if parameter should be excluded from weight decay
            no_decay = (
                p.ndim < 2
                or n.endswith(".bias")
                or "norm" in n.lower()
                or "embed" in n
                or "space" in n
                or "time" in n
                or n.endswith(("beta", "mix", "log_r", "prior_gain"))
            )

            key = (category, not no_decay)
            groups[key].append(p)

        lrs = {
            "ln": lr_backbone_ln,
            "head": lr_head,
            "new": lr_new,
        }

        out = []
        for (category, decay), ps in groups.items():
            if ps:
                out.append({
                    "params": ps,
                    "lr": lrs[category],
                    "weight_decay": weight_decay if decay else 0.0,
                    "name": f"{category}_{'decay' if decay else 'nodecay'}",
                })
        return out

    def load_v2_weights(self, state_dict: dict, strict: bool = False) -> tuple[list[str], list[str]]:
        """Load weights from Stage B v2 checkpoint (outputs/checkpoints_v2_fixed/best_ema.pt).

        Asserts that all foundation encoder, projection, NMSSS blocks, and CLS head weights
        are present, while new v3 modules (res.*, readout.*) are gracefully initialized.
        """
        # If checkpoint contains wrapper dict
        if "model_ema" in state_dict:
            state_dict = state_dict["model_ema"]
        elif "model" in state_dict:
            state_dict = state_dict["model"]

        # Filter out v2 linear readout head keys that changed in v3
        filtered_state = {}
        for k, v in state_dict.items():
            if k.startswith("readout."):
                continue  # readout was upgraded to EvidenceReadoutV3
            filtered_state[k] = v

        missing, unexpected = self.load_state_dict(filtered_state, strict=False)

        # Expected missing keys must belong to new modules
        non_new_missing = [
            k for k in missing
            if not (k.startswith("res.") or k.startswith("readout."))
        ]
        if non_new_missing:
            raise RuntimeError(f"Unexpected missing keys from Stage B checkpoint: {non_new_missing}")

        return missing, unexpected
