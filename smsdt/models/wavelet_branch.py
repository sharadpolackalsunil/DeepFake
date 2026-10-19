import torch
import torch.nn as nn
import torch.nn.functional as F


class HaarDWT2D(nn.Module):
    """Single-level 2D Haar DWT. Input (B, 3, H, W) -> LL, LH, HL, HH each (B, 3, H/2, W/2)."""
    def __init__(self):
        super().__init__()
        ll = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
        lh = torch.tensor([[0.5, 0.5], [-0.5, -0.5]])
        hl = torch.tensor([[0.5, -0.5], [0.5, -0.5]])
        hh = torch.tensor([[0.5, -0.5], [-0.5, 0.5]])
        filt = torch.stack([ll, lh, hl, hh]).unsqueeze(1)   # (4, 1, 2, 2)
        filt = filt.repeat(3, 1, 1, 1)                       # apply per RGB channel -> (12, 1, 2, 2)
        self.register_buffer("filt", filt)

    def forward(self, x):
        B, C, H, W = x.shape
        out = F.conv2d(x, self.filt, stride=2, groups=C)     # (B, C*4, H/2, W/2)
        out = out.view(B, C, 4, H // 2, W // 2)
        ll, lh, hl, hh = out.unbind(dim=2)
        return ll, lh, hl, hh


class WaveletBranch(nn.Module):
    """Wavelet frequency branch with signed Haar subband projection onto a 14x14 or 7x7 token grid."""
    def __init__(self, embed_dim=256, stem_type="strided_patch", pool_energy=False, grid=14):
        super().__init__()
        self.dwt = HaarDWT2D()
        self.stem_type = stem_type
        self.pool_energy = pool_energy
        self.grid = grid

        # For 224x224 input: DWT produces 112x112 subbands.
        # stride=8 maps 112 -> 14x14; stride=16 maps 112 -> 7x7
        patch_size = 8 if grid == 14 else 16

        if stem_type == "strided_patch":
            self.proj = nn.Sequential(
                nn.Conv2d(9, embed_dim, kernel_size=patch_size, stride=patch_size),
                nn.BatchNorm2d(embed_dim),
                nn.GELU(),
            )
        else:
            self.proj = nn.Sequential(
                nn.Conv2d(9, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.GELU(),
                nn.Conv2d(64, embed_dim, 3, padding=1),
                nn.BatchNorm2d(embed_dim),
                nn.GELU(),
            )

    def forward(self, x):
        _, lh, hl, hh = self.dwt(x)          # discard LL spatial approximation
        if self.pool_energy:
            # Optional magnitude energy
            lh, hl, hh = torch.abs(lh), torch.abs(hl), torch.abs(hh)
        hf = torch.cat([lh, hl, hh], dim=1)  # (B, 9, H/2, W/2)
        return self.proj(hf)                 # (B, embed_dim, grid, grid)
