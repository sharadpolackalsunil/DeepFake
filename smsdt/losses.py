"""S-MSDT Loss Functions.

Primary: Binary cross-entropy with label smoothing (0.05) + patch-level mask loss (lambda=0.5).
Auxiliary / Ablation: Focal BCE, Supervised Contrastive, Cross-domain alignment.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def detection_loss(logit, label, patch_logits=None, mask=None, has_mask=None,
                   smoothing=0.05, lam_mask=0.5):
    """Primary S-MSDT detection loss with localized patch-mask supervision.

    Args:
        logit: (B,) raw model output logit
        label: (B,) binary targets (0=real, 1=fake)
        patch_logits: (B, T, N) patch-level binary logits from EvidenceReadout
        mask: (B, T, g, g) ground-truth blend/manipulation mask (0 for real)
        has_mask: (B,) binary flags indicating if mask supervision is valid
        smoothing: label smoothing factor
        lam_mask: weight for patch-mask BCE loss
    """
    target = label * (1.0 - smoothing) + 0.5 * smoothing
    cls_loss = F.binary_cross_entropy_with_logits(logit.float(), target.float())
    total = cls_loss
    logs = {"cls": float(cls_loss.detach())}

    if patch_logits is not None and mask is not None:
        sel = has_mask.bool() if has_mask is not None else torch.ones_like(label, dtype=torch.bool)
        if sel.any():
            m = (mask[sel].flatten(2) > 0.1).float()  # (B', T, g*g)
            lm = F.binary_cross_entropy_with_logits(patch_logits[sel].float(), m)
            total = total + lam_mask * lm
            logs["mask"] = float(lm.detach())

    return total, logs


def focal_bce(logits, targets, alpha=0.2, gamma=2.0, label_smoothing=0.0):
    """Focal binary cross-entropy loss (legacy / ablation)."""
    if label_smoothing > 0.0:
        targets = targets * (1.0 - label_smoothing) + 0.5 * label_smoothing

    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    pt = p * targets + (1 - p) * (1 - targets)
    focal = ((1 - pt) ** gamma) * ce
    alpha_w = alpha * targets + (1 - alpha) * (1 - targets)
    return (alpha_w * focal).mean()


def supervised_contrastive_loss(features, labels, manipulation_types=None, temperature=0.07):
    """Supervised contrastive loss (legacy / ablation)."""
    features = F.normalize(features, dim=-1)
    B = features.shape[0]
    if B < 2:
        return torch.tensor(0.0, device=features.device, requires_grad=True)

    sim = features @ features.T / temperature
    cand_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    cand_mask.fill_diagonal_(0)
    pos_count = cand_mask.sum(1)
    valid = (pos_count > 0)
    if not valid.any():
        return torch.tensor(0.0, device=features.device, requires_grad=True)

    self_mask = torch.eye(B, device=features.device, dtype=torch.bool)
    sim = sim.masked_fill(self_mask, -1e9)
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    loss = -(cand_mask * log_prob).sum(1) / pos_count.clamp(min=1)
    return loss[valid].mean()


def alignment_loss(z_rgb, z_dwt, mode="decorrelate"):
    """Cross-domain feature alignment / decorrelation loss (legacy / ablation)."""
    cos = F.cosine_similarity(z_rgb, z_dwt, dim=-1)
    if mode in ("decorrelate", "orthogonality"):
        return torch.mean(torch.abs(cos))
    return torch.mean(2.0 - 2.0 * cos)


def total_loss(logits, labels, z_rgb=None, z_dwt=None, pooled=None, manipulation_types=None, **kwargs):
    """Backward-compatible combined loss wrapper."""
    return focal_bce(logits, labels)
