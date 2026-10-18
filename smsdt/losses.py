"""
S-MSDT v2 Loss Functions
=======================
- Class-Balanced Focal BCE (alpha=0.2, gamma=2.0) for 1:4 real:fake ratio.
- Asymmetric Real-Center Contrastive Loss: compacts genuine faces around a prototype
  and pushes fakes outside a hyperspherical margin m=0.25.
- Cross-Domain Decorrelation / Orthogonality Alignment Loss.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def focal_bce(logits, targets, alpha=0.2, gamma=2.0, label_smoothing=0.0):
    """Focal binary cross-entropy loss with class-balancing and optional label smoothing.

    Args:
        logits: (B,) raw model output (before sigmoid)
        targets: (B,) binary labels (0=real, 1=fake)
        alpha: weight for the positive (fake) class. Set to 0.2 for FF++'s 1:4
               real:fake ratio so effective per-sample weight is balanced:
               4 fakes * 0.2 = 0.8 effective weight vs 1 real * (1 - 0.2) = 0.8 effective weight.
        gamma: focusing parameter — downweights easy/well-classified examples
        label_smoothing: fraction of label smoothing to apply
    """
    if label_smoothing > 0.0:
        targets = targets * (1.0 - label_smoothing) + 0.5 * label_smoothing

    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    pt = p * targets + (1 - p) * (1 - targets)
    focal = ((1 - pt) ** gamma) * ce
    alpha_w = alpha * targets + (1 - alpha) * (1 - targets)
    return (alpha_w * focal).mean()


def asymmetric_real_compactness_loss(features, targets, margin=0.25):
    """Asymmetric Real-Center Contrastive Loss:
    Compacts authentic face representations around a real-domain centroid while
    pushing fake representations outside a margin m. Prevents heterogeneous
    fake methods from being forced into an unnatural single cluster.

    Args:
        features: (B, D) post-ViT learned representations
        targets:  (B,) binary labels (0=real, 1=fake)
        margin:   cosine margin for fakes (default 0.25)
    """
    features_norm = F.normalize(features, dim=-1)
    real_mask = (targets == 0)
    fake_mask = (targets == 1)

    if not real_mask.any():
        return torch.tensor(0.0, device=features.device, requires_grad=True)

    # Compute centroid of real samples in batch
    real_feats = features_norm[real_mask]
    c_real = F.normalize(real_feats.mean(dim=0, keepdim=True), dim=-1)  # (1, D)

    # 1. Compactness penalty for real samples: pull toward centroid
    sim_real = (real_feats * c_real).sum(dim=-1)
    loss_real = (1.0 - sim_real).mean()

    # 2. Margin penalty for fake samples: push cosine similarity below margin
    if fake_mask.any():
        fake_feats = features_norm[fake_mask]
        sim_fake = (fake_feats * c_real).sum(dim=-1)
        loss_fake = F.relu(sim_fake - margin).pow(2).mean()
    else:
        loss_fake = torch.tensor(0.0, device=features.device)

    return loss_real + loss_fake


def supervised_contrastive_loss(features, labels, manipulation_types=None,
                                temperature=0.07):
    """Supervised contrastive loss with same-manipulation-type clustering."""
    features = F.normalize(features, dim=-1)
    B = features.shape[0]
    if B < 2:
        return torch.tensor(0.0, device=features.device, requires_grad=True)

    sim = features @ features.T / temperature

    mask = None
    if manipulation_types is not None:
        cand_mask = (manipulation_types.unsqueeze(0) == manipulation_types.unsqueeze(1)).float()
        cand_mask.fill_diagonal_(0)
        if cand_mask.sum(1).gt(0).any():
            mask = cand_mask

    if mask is None:
        cand_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        cand_mask.fill_diagonal_(0)
        if cand_mask.sum(1).gt(0).any():
            mask = cand_mask

    if mask is None:
        return torch.tensor(0.0, device=features.device, requires_grad=True)

    pos_count = mask.sum(1)
    valid = (pos_count > 0)
    if not valid.any():
        return torch.tensor(0.0, device=features.device, requires_grad=True)

    self_mask = torch.eye(B, device=features.device, dtype=torch.bool)
    sim = sim.masked_fill(self_mask, -1e9)
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)

    loss = -(mask * log_prob).sum(1) / pos_count.clamp(min=1)
    loss = loss[valid]
    return loss.mean()


def alignment_loss(z_rgb, z_dwt, mode="decorrelate"):
    """Cross-domain feature alignment / decorrelation loss."""
    z_rgb_n = F.normalize(z_rgb, dim=-1)
    z_dwt_n = F.normalize(z_dwt, dim=-1)

    if mode in ("decorrelate", "orthogonality"):
        return (z_rgb_n * z_dwt_n).sum(dim=-1).abs().mean()
    else:
        return 2 - 2 * (z_rgb_n * z_dwt_n).sum(dim=-1).mean()


def total_loss(logits, targets, z_rgb=None, z_dwt=None, z_post_vit=None,
               manipulation_types=None, alpha=0.2, gamma=2.0,
               label_smoothing=0.0, lambda_contrastive=0.1, lambda_align=0.05,
               align_mode="decorrelate", contrastive_mode="asymmetric", real_margin=0.25):
    """Combined loss: balanced focal BCE + contrastive + decorrelation alignment.

    Args:
        logits: (B,) raw model output
        targets: (B,) binary labels
        z_rgb: (B, D) pooled RGB features (for alignment loss)
        z_dwt: (B, D) pooled wavelet features (for alignment loss)
        z_post_vit: (B, D) post-ViT pooled features for contrastive loss
        manipulation_types: (B,) manipulation type IDs (optional)
        alpha, gamma: focal loss hyperparameters (default alpha=0.2 for 1:4 real:fake ratio)
        label_smoothing: label smoothing factor
        lambda_contrastive: weight for contrastive loss
        lambda_align: weight for alignment loss
        align_mode: "decorrelate" or "cosine"
        contrastive_mode: "asymmetric" (default in v2) or "supervised"
        real_margin: margin for asymmetric real-center loss (default 0.25)
    """
    L_focal = focal_bce(logits, targets, alpha=alpha, gamma=gamma,
                        label_smoothing=label_smoothing)

    L_total = L_focal
    loss_dict = {"focal": L_focal.item()}

    feat_con = z_post_vit if z_post_vit is not None else z_dwt
    if feat_con is not None and lambda_contrastive > 0:
        if contrastive_mode == "asymmetric":
            L_con = asymmetric_real_compactness_loss(feat_con, targets, margin=real_margin)
        else:
            L_con = supervised_contrastive_loss(feat_con, targets, manipulation_types)
        L_total = L_total + lambda_contrastive * L_con
        loss_dict["contrastive"] = L_con.item()

    if z_rgb is not None and z_dwt is not None and lambda_align > 0:
        L_align = alignment_loss(z_rgb, z_dwt, mode=align_mode)
        L_total = L_total + lambda_align * L_align
        loss_dict["alignment"] = L_align.item()

    loss_dict["total"] = L_total.item()
    return L_total, loss_dict
