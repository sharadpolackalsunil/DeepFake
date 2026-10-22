"""S-MSDTv3 Multi-Task Loss Formulation.

Includes:
1. Asymmetric Focal BCE Loss: Eliminates the rigid label smoothing floor while
   dynamically modulating hard synthetic examples.
2. Pairwise Margin Ranking Loss: Exploits matched real-fake identity pairs in the batch
   to optimize the Area Under the ROC Curve (AUC) directly.
3. Soft Difference-Mask Supervision: Weighted BCE over localized 16x16 patch logits
   with positive class weighting (w_pos = 3.0).
4. Auxiliary Frequency Supervision: Deep supervision on ResidualStream high-frequency features.
5. Online Hard Example Mining (OHEM): Preserves authentic specificity by retaining all reals
   and the top 50% highest-loss fakes per batch.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def focal_bce(
    logit: torch.Tensor,
    y: torch.Tensor,
    alpha: float = 0.5,
    g_pos: float = 2.0,
    g_neg: float = 1.0,
) -> torch.Tensor:
    """Asymmetric focal binary cross-entropy loss.
    
    Args:
        logit: (B,) float logits
        y:     (B,) targets in {0.0, 1.0}
        alpha: balance factor for positive samples (default 0.5)
        g_pos: focusing parameter gamma for positive fakes (default 2.0)
        g_neg: focusing parameter gamma for authentic reals (default 1.0)
    Returns:
        Per-sample focal loss tensor of shape (B,)
    """
    logit = logit.float()
    y = y.float()
    p = torch.sigmoid(logit)
    lp = F.logsigmoid(logit)
    ln = F.logsigmoid(-logit)

    pos = -alpha * (1.0 - p).pow(g_pos) * lp
    neg = -(1.0 - alpha) * p.pow(g_neg) * ln
    return y * pos + (1.0 - y) * neg


def detection_loss_v3(
    logit: torch.Tensor,
    label: torch.Tensor,
    patch_logits: torch.Tensor,
    soft_mask: torch.Tensor | None = None,
    aux_logit: torch.Tensor | None = None,
    method_w: torch.Tensor | None = None,
    pair_idx: torch.Tensor | None = None,
    lam_rank: float = 0.5,
    lam_mask: float = 0.5,
    lam_freq: float = 0.3,
    margin: float = 1.0,
    ohem_frac: float = 0.5,
    pos_weight: float = 3.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Multi-task objective combining focal BCE, pairwise ranking, soft patch-masks,
    and frequency auxiliary supervision.

    Args:
        logit:        (B,) clip-level manipulation logits
        label:        (B,) binary targets (0.0=real, 1.0=fake)
        patch_logits: (B, T, N) or (B, N) localized patch anomaly logits
        soft_mask:    (B, T, N) or (B, N) ground-truth difference masks in [0, 1]
        aux_logit:    (B,) or (BT,) auxiliary frequency predictions from ResidualStream
        method_w:     optional (B,) dynamic per-method sample weights
        pair_idx:     optional (B,) indices of matched real<->fake identity partners
        lam_rank:     pairwise margin ranking loss weight (default 0.5)
        lam_mask:     soft patch mask loss weight (default 0.5)
        lam_freq:     frequency auxiliary loss weight (default 0.3)
        margin:       pairwise ranking margin (default 1.0)
        ohem_frac:    fraction of highest-loss fakes to keep (default 0.5; 1.0 disables OHEM)
        pos_weight:   weight multiplier on positive manipulated patch pixels (default 3.0)

    Returns:
        total_loss: scalar tensor for backward()
        logs:       dict of detached component losses for telemetry
    """
    b = label.shape[0]
    logit_f = logit.float()
    label_f = label.float()

    # 1. Focal classification loss with optional method reweighting
    cls_per_item = focal_bce(logit_f, label_f)
    if method_w is not None:
        cls_per_item = cls_per_item * method_w.float()

    # Online Hard Example Mining (OHEM)
    if ohem_frac < 1.0:
        is_fake = label_f > 0.5
        n_fake = int(is_fake.sum().item())
        if n_fake > 1:
            n_keep = max(1, int(ohem_frac * n_fake))
            fake_losses = cls_per_item.masked_fill(~is_fake, float("-inf"))
            thr = torch.topk(fake_losses, n_keep).values.min()
            keep_mask = (~is_fake) | (cls_per_item >= thr)
            loss_cls = (cls_per_item * keep_mask).sum() / keep_mask.sum().clamp(min=1)
        else:
            loss_cls = cls_per_item.mean()
    else:
        loss_cls = cls_per_item.mean()

    # 2. Pairwise Margin Ranking Loss
    # If pair_idx is provided (or if batch is structured as pairs: [r_0, f_0, r_1, f_1...])
    if pair_idx is None and b >= 2 and (b % 2 == 0):
        # Default paired interleaved layout: pair_idx[2k] = 2k+1, pair_idx[2k+1] = 2k
        pair_idx = torch.arange(b, device=label.device)
        pair_idx[0::2] = torch.arange(1, b, 2, device=label.device)
        pair_idx[1::2] = torch.arange(0, b, 2, device=label.device)

    is_fake = label_f > 0.5
    fake_indices = torch.nonzero(is_fake).squeeze(1)

    if lam_rank > 0.0 and pair_idx is not None and fake_indices.numel() > 0:
        real_partners = pair_idx[fake_indices]
        # Diff: fake_logit - real_logit
        diff = logit_f[fake_indices] - logit_f[real_partners]
        # Maximize margin: softplus(-(diff - margin))
        loss_rank = F.softplus(-(diff - margin)).mean()
    else:
        loss_rank = logit_f.new_zeros(())

    # 3. Soft Patch-Mask BCE Loss
    if lam_mask > 0.0 and soft_mask is not None:
        pl = patch_logits.reshape(b, -1).float()
        sm = soft_mask.reshape(b, -1).float().clamp(0.0, 1.0)
        # Apply pos_weight to positive mask pixels
        w = 1.0 + (pos_weight - 1.0) * sm
        loss_mask = (F.binary_cross_entropy_with_logits(pl, sm, reduction="none") * w).mean()
    else:
        loss_mask = logit_f.new_zeros(())

    # 4. Auxiliary Frequency Supervision
    if lam_freq > 0.0 and aux_logit is not None:
        aux_f = aux_logit.float()
        if aux_f.numel() != b:
            aux_f = aux_f.view(b, -1).mean(dim=1)
        loss_freq = F.binary_cross_entropy_with_logits(aux_f, label_f)
    else:
        loss_freq = logit_f.new_zeros(())

    # Total Multi-Task Loss
    total_loss = loss_cls + lam_rank * loss_rank + lam_mask * loss_mask + lam_freq * loss_freq

    logs = {
        "loss": total_loss.detach(),
        "cls": loss_cls.detach(),
        "rank": loss_rank.detach(),
        "mask": loss_mask.detach(),
        "freq": loss_freq.detach(),
    }
    return total_loss, logs
