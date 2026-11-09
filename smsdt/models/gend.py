"""GenD / LNCLIP-DF Reference Baseline Implementation.

Based on official GenD codebase (arXiv:2508.06248 v3):
1. Pretrained CLIP ViT-L/14 with LayerNorm-only parameter-efficient fine-tuning
2. L2-normalized feature embeddings
3. Linear probe classification head
4. Multi-task loss: Cross-Entropy + Label-Aware Alignment + Uniformity losses
5. Spherical linear interpolation (Slerp) latent augmentation for paired training
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional

from .clip_encoder import CLIPFrameEncoder


def slerp(p0: torch.Tensor, p1: torch.Tensor, t: float, eps: float = 1e-7) -> torch.Tensor:
    """Spherical linear interpolation between two batches of unit vectors."""
    dot = (p0 * p1).sum(dim=-1, keepdim=True).clamp(-1.0 + eps, 1.0 - eps)
    omega = torch.acos(dot)
    sin_omega = torch.sin(omega)
    so = torch.sin((1.0 - t) * omega) / sin_omega
    s1 = torch.sin(t * omega) / sin_omega
    return so * p0 + s1 * p1


def alignment_loss(embeddings: torch.Tensor, labels: torch.Tensor, alpha: float = 2.0) -> torch.Tensor:
    """Label-aware alignment loss on unit-normalized embeddings."""
    if embeddings.size(0) < 2:
        return torch.tensor(0.0, device=embeddings.device)
    labels_equal = (labels[:, None] == labels[None, :]).triu(diagonal=1)
    pos_idx = torch.nonzero(labels_equal, as_tuple=False)
    if pos_idx.numel() == 0:
        return torch.tensor(0.0, device=embeddings.device)
    x = embeddings[pos_idx[:, 0]]
    y = embeddings[pos_idx[:, 1]]
    return (x - y).norm(p=2, dim=1).pow(alpha).mean()


def uniformity_loss(embeddings: torch.Tensor, t: float = 2.0, clip_val: float = 1e-6) -> torch.Tensor:
    """Uniformity loss on unit-normalized embeddings."""
    return torch.pdist(embeddings, p=2).pow(2).mul(-t).exp().mean().clamp(min=clip_val).log()


class GenDModel(nn.Module):
    """GenD / LNCLIP-DF baseline detector.
    
    Args:
        name: Hugging Face CLIP model identifier (default: openai/clip-vit-large-patch14)
        tune: 'layernorm' for LN-only PEFT
        normalize_inputs: Whether linear probe takes L2-normalized features
    """

    def __init__(
        self,
        name: str = "openai/clip-vit-large-patch14",
        tune: str = "layernorm",
        normalize_inputs: bool = True,
    ):
        super().__init__()
        self.encoder = CLIPFrameEncoder(name=name, tune=tune, grad_ckpt=True)
        self.features_dim = 1024
        self.normalize_inputs = normalize_inputs
        self.head = nn.Linear(self.features_dim, 1)

    def extract_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract raw CLS token and L2-normalized embedding.
        
        Args:
            x: (B, 3, 224, 224) input tensor in CLIP normalization
        Returns:
            raw_features: (B, 1024)
            l2_embeddings: (B, 1024) unit-norm
        """
        cls, _ = self.encoder(x)
        l2_norm = F.normalize(cls, p=2, dim=1)
        return cls, l2_norm

    def forward(
        self,
        x: torch.Tensor,
        slerp_t: Optional[float] = None,
        pair_x: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Args:
            x: (B, 3, 224, 224) input frames
            slerp_t: Optional interpolation factor for latent Slerp
            pair_x: Matching real/fake pair frames for Slerp
            
        Returns:
            logits: (B,) logit predictions
            l2_embeddings: (B, 1024) unit-normalized feature embeddings
        """
        raw_feat, l2_feat = self.extract_features(x)

        if slerp_t is not None and pair_x is not None and self.training:
            _, pair_l2 = self.extract_features(pair_x)
            l2_feat = slerp(l2_feat, pair_l2, slerp_t)
            raw_feat = l2_feat

        inp = l2_feat if self.normalize_inputs else raw_feat
        logit = self.head(inp).squeeze(-1)
        return logit, l2_feat

    def compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        embeddings: torch.Tensor,
        lambda_align: float = 0.5,
        lambda_unif: float = 0.5
    ) -> Dict[str, torch.Tensor]:
        """Compute GenD multi-task training loss."""
        bce = F.binary_cross_entropy_with_logits(logits, labels.float())
        align = alignment_loss(embeddings, labels)
        unif = uniformity_loss(embeddings)
        total = bce + lambda_align * align + lambda_unif * unif
        return {
            "loss": total,
            "loss_bce": bce,
            "loss_align": align,
            "loss_unif": unif
        }

