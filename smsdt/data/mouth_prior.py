"""Landmark-Guided Perioral Prior Generation.

Extracts mouth/lip landmarks from InsightFace 2D-106 facial landmark arrays
and generates a soft Gaussian spatial prior on the 16x16 patch grid for EvidenceReadoutV3.
"""
import torch
import torch.nn.functional as F
import numpy as np

# In InsightFace 2D-106: landmarks 52..71 encompass outer and inner lip contours
MOUTH_INDICES_106 = list(range(52, 72))


def compute_perioral_prior(
    lmk: torch.Tensor | np.ndarray,
    grid_size: int = 16,
    crop_size: float = 224.0,
    sigma_patches: float = 1.2,
) -> torch.Tensor:
    """Compute 2D soft perioral Gaussian prior on a grid of spatial patches.

    Args:
        lmk: (..., 106, 2) facial landmarks in pixel space [0, crop_size] or normalized [0, 1].
        grid_size: patch grid dimension (default 16 for 224/14).
        crop_size: image pixel size (default 224.0).
        sigma_patches: Gaussian standard deviation in patch units (default 1.2).

    Returns:
        prior: (..., grid_size * grid_size) soft Gaussian attention prior in [0, 1].
    """
    if isinstance(lmk, np.ndarray):
        lmk = torch.from_numpy(lmk)

    orig_shape = lmk.shape[:-2]  # e.g., (B, T) or (B,)
    lmk_flat = lmk.view(-1, 106, 2).float()  # (M, 106, 2)
    device = lmk_flat.device

    # Normalize landmarks to [0, 1] if not already normalized
    if lmk_flat.max() > 1.5:
        lmk_norm = lmk_flat / crop_size
    else:
        lmk_norm = lmk_flat

    # Extract mouth landmarks: (M, 20, 2)
    mouth_pts = lmk_norm[:, MOUTH_INDICES_106, :]  # (M, 20, 2)
    mouth_center = mouth_pts.mean(dim=1, keepdim=True)  # (M, 1, 2)
    mouth_radius = (mouth_pts - mouth_center).norm(dim=-1).max(dim=1).values.clamp(min=0.04)  # (M,)

    # 16x16 patch centers in [0, 1] coordinate space
    patch_coords = (torch.arange(grid_size, device=device, dtype=torch.float32) + 0.5) / float(grid_size)
    gy, gx = torch.meshgrid(patch_coords, patch_coords, indexing="ij")
    grid_pts = torch.stack([gx, gy], dim=-1).view(1, grid_size * grid_size, 2)  # (1, 256, 2)

    # Compute Euclidean distance from each patch center to mouth center in normalized space
    dist = (grid_pts - mouth_center).norm(dim=-1)  # (M, 256)

    # Convert patch sigma to normalized coordinate space
    patch_sigma = (sigma_patches / float(grid_size)) * (mouth_radius.unsqueeze(-1) / 0.15).clamp(0.8, 1.5)
    prior = torch.exp(-0.5 * (dist / patch_sigma).pow(2))  # (M, 256)

    # Reshape back to original batch dimensions: (..., 256)
    return prior.view(*orig_shape, grid_size * grid_size)
