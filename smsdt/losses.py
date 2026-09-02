"""
S-MSDT Loss Functions
====================
Focal BCE (primary) + Supervised Contrastive + Cross-Domain Alignment (auxiliary).
Designed for FF++'s 1:4 real:fake class imbalance.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def focal_bce(logits, targets, alpha=0.8, gamma=2.0):
    """Focal binary cross-entropy loss.

    Args:
        logits: (B,) raw model output (before sigmoid)
        targets: (B,) binary labels (0=real, 1=fake)
        alpha: weight for the positive (fake) class. Set to 0.8 for FF++'s 1:4
               real:fake ratio so effective per-sample weight is balanced:
               fake gets alpha/4 ≈ 0.2, real gets (1-alpha)/1 = 0.2
        gamma: focusing parameter — downweights easy/well-classified examples
    """
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    pt = p * targets + (1 - p) * (1 - targets)
    focal = ((1 - pt) ** gamma) * ce
    alpha_w = alpha * targets + (1 - alpha) * (1 - targets)
    return (alpha_w * focal).mean()


def supervised_contrastive_loss(features, labels, manipulation_types=None,
                                temperature=0.07):
    """Supervised contrastive loss on wavelet tokens.

    Pulls same-manipulation-type fakes together and pushes real/fake apart.
    Helps the frequency branch specialize instead of being drowned out
    by the (usually stronger) RGB gradient.

    Args:
        features: (B, D) pooled wavelet features
        labels: (B,) binary labels
        manipulation_types: (B,) integer manipulation type IDs (optional).
                           If provided, positive pairs require same manipulation type.
                           If None, falls back to same binary label.
        temperature: softmax temperature
    """
    features = F.normalize(features, dim=-1)
    B = features.shape[0]
    if B < 2:
        return torch.tensor(0.0, device=features.device, requires_grad=True)

    sim = features @ features.T / temperature  # (B, B)

    # Build positive pair mask
    if manipulation_types is not None:
        # Same manipulation type (or both real) = positive pair
        mask = (manipulation_types.unsqueeze(0) == manipulation_types.unsqueeze(1)).float()
    else:
        # Fallback: same binary label = positive pair
        mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()

    mask.fill_diagonal_(0)  # exclude self-pairs

    # Check if there are any positive pairs
    pos_count = mask.sum(1)
    valid = (pos_count > 0)
    if not valid.any():
        return torch.tensor(0.0, device=features.device, requires_grad=True)

    # Log-softmax over all pairs (excluding self)
    self_mask = torch.eye(B, device=features.device, dtype=torch.bool)
    # Set self-similarity to large negative so it's excluded from softmax
    sim = sim.masked_fill(self_mask, -1e9)
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)

    # Mean of log-prob over positive pairs (only for samples with valid positives)
    loss = -(mask * log_prob).sum(1) / pos_count.clamp(min=1)
    loss = loss[valid]  # only backprop through samples with positive pairs
    return loss.mean()


def alignment_loss(z_rgb, z_dwt):
    """Cross-domain alignment loss (CAL).

    Encourages RGB and wavelet features to learn complementary representations
    while maintaining shared structure for deepfake artifact localization.

    Args:
        z_rgb: (B, D) pooled RGB features
        z_dwt: (B, D) pooled wavelet features
    """
    z_rgb_n = F.normalize(z_rgb, dim=-1)
    z_dwt_n = F.normalize(z_dwt, dim=-1)
    # Cosine similarity → alignment = 2 - 2*cos_sim
    return 2 - 2 * (z_rgb_n * z_dwt_n).sum(dim=-1).mean()


def total_loss(logits, targets, z_rgb=None, z_dwt=None,
               manipulation_types=None, alpha=0.8, gamma=2.0,
               lambda_contrastive=0.1, lambda_align=0.05):
    """Combined loss: focal BCE + supervised contrastive + alignment.

    Args:
        logits: (B,) raw model output
        targets: (B,) binary labels
        z_rgb: (B, D) pooled RGB features (optional, for alignment loss)
        z_dwt: (B, D) pooled wavelet features (optional, for contrastive + alignment)
        manipulation_types: (B,) manipulation type IDs (optional)
        alpha, gamma: focal loss hyperparameters
        lambda_contrastive: weight for contrastive loss
        lambda_align: weight for alignment loss
    """
    L_focal = focal_bce(logits, targets, alpha=alpha, gamma=gamma)

    L_total = L_focal
    loss_dict = {"focal": L_focal.item()}

    if z_dwt is not None and lambda_contrastive > 0:
        L_con = supervised_contrastive_loss(z_dwt, targets, manipulation_types)
        L_total = L_total + lambda_contrastive * L_con
        loss_dict["contrastive"] = L_con.item()

    if z_rgb is not None and z_dwt is not None and lambda_align > 0:
        L_align = alignment_loss(z_rgb, z_dwt)
        L_total = L_total + lambda_align * L_align
        loss_dict["alignment"] = L_align.item()

    loss_dict["total"] = L_total.item()
    return L_total, loss_dict
