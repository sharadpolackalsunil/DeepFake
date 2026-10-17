import torch
import torch.nn as nn

class HaarDWT2D(nn.Module):
    """Single-level 2D Haar DWT. Input (B,3,H,W) -> LL,LH,HL,HH each (B,3,H/2,W/2)."""
    def __init__(self):
        super().__init__()
        ll = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
        lh = torch.tensor([[0.5, 0.5], [-0.5, -0.5]])
        hl = torch.tensor([[0.5, -0.5], [0.5, -0.5]])
        hh = torch.tensor([[0.5, -0.5], [-0.5, 0.5]])
        filt = torch.stack([ll, lh, hl, hh]).unsqueeze(1)   # (4,1,2,2)
        filt = filt.repeat(3, 1, 1, 1)                       # apply per RGB channel -> (12,1,2,2)
        self.register_buffer("filt", filt)

    def forward(self, x):
        B, C, H, W = x.shape
        out = nn.functional.conv2d(x, self.filt, stride=2, groups=C)  # (B, C*4, H/2, W/2)
        out = out.view(B, C, 4, H // 2, W // 2)
        ll, lh, hl, hh = out.unbind(dim=2)
        return ll, lh, hl, hh

class WaveletBranch(nn.Module):
    def __init__(self, embed_dim=256, stem_type="strided_patch", pool_energy=True):
        super().__init__()
        self.dwt = HaarDWT2D()
        self.stem_type = stem_type
        self.pool_energy = pool_energy
        if stem_type == "strided_patch":
            # Strided patch stem: 9 channels -> embed_dim with 16x16 kernel, stride 16 (for 112 -> 7x7)
            # ~60x cheaper (~0.03 GMAC vs 1.92 GMAC) and directly aligns with 7x7 token grid
            self.proj = nn.Sequential(
                nn.Conv2d(9, embed_dim, kernel_size=16, stride=16),
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
        _, lh, hl, hh = self.dwt(x)          # discard LL per spec
        if self.pool_energy:
            # Take absolute magnitude / energy to avoid signed cancellation across pooling
            lh, hl, hh = torch.abs(lh), torch.abs(hl), torch.abs(hh)
        hf = torch.cat([lh, hl, hh], dim=1)  # (B, 9, H/2, W/2)
        return self.proj(hf)                 # (B, embed_dim, 7, 7) or (B, embed_dim, H/2, W/2)
