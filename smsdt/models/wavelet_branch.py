"""
S-MSDT v3 Spatio-Temporal Wavelet Frequency Guard
================================================
- Single-level 2D Haar DWT (LH, HL, HH subbands, discarding LL for spatial path).
- High-to-Low Spectral Energy Ratio (rho_t) as parameter-free compression sensor.
- 3-Stage Depthwise-Separable + Squeeze-and-Excitation (SE) Stem: 9ch @ 112×112 → 7×7.

V3 changes from v2:
  - Removed temporal LL-diff channels (let ViT handle temporal reasoning)
  - 9ch input instead of 12ch (no BatchNorm distribution mismatch at t=0)
  - Output 7×7 instead of 14×14 (matched to ViT capacity)
"""
import torch
import torch.nn as nn


class SqueezeExcitation(nn.Module):
    """Squeeze-and-Excitation channel attention block."""
    def __init__(self, channels, reduction=4):
        super().__init__()
        mid = max(8, channels // reduction)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, mid, 1),
            nn.GELU(),
            nn.Conv2d(mid, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.fc(x)


class HaarDWT2D(nn.Module):
    """Single-level 2D Haar DWT. Input (B, 3, H, W) -> LL, LH, HL, HH each (B, 3, H/2, W/2)."""
    def __init__(self):
        super().__init__()
        ll = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
        lh = torch.tensor([[0.5, 0.5], [-0.5, -0.5]])
        hl = torch.tensor([[0.5, -0.5], [0.5, -0.5]])
        hh = torch.tensor([[0.5, -0.5], [-0.5, 0.5]])
        filt = torch.stack([ll, lh, hl, hh]).unsqueeze(1)   # (4, 1, 2, 2)
        filt = filt.repeat(3, 1, 1, 1)                       # per RGB channel -> (12, 1, 2, 2)
        self.register_buffer("filt", filt)

    def forward(self, x):
        B, C, H, W = x.shape
        out = nn.functional.conv2d(x, self.filt, stride=2, groups=C)  # (B, C*4, H/2, W/2)
        out = out.view(B, C, 4, H // 2, W // 2)
        ll, lh, hl, hh = out.unbind(dim=2)
        return ll, lh, hl, hh


class DepthwiseSeparableBlock(nn.Module):
    """Depthwise-Separable Convolution with BatchNorm, GELU, and Squeeze-and-Excitation."""
    def __init__(self, in_ch, out_ch, stride=2):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=stride, padding=1, groups=in_ch, bias=False)
        self.bn_dw = nn.BatchNorm2d(in_ch)
        self.pw = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn_pw = nn.BatchNorm2d(out_ch)
        self.act = nn.GELU()
        self.se = SqueezeExcitation(out_ch)

    def forward(self, x):
        x = self.act(self.bn_dw(self.dw(x)))
        x = self.act(self.bn_pw(self.pw(x)))
        return self.se(x)


class WaveletBranch(nn.Module):
    """Spatio-Temporal Wavelet Frequency Guard (v3):
    Per-frame 2D Haar DWT -> 9ch spatial HF -> 3-stage DW-SE stem -> 7x7 feature maps.
    Also computes spectral energy ratio rho as a parameter-free compression sensor.

    The temporal LL-diff channels from v2 are removed — all temporal reasoning is
    delegated to the ViT backbone, keeping the wavelet branch purely spatial.
    """
    def __init__(self, embed_dim=256, stem_type="dw_separable_se", pool_energy=True):
        super().__init__()
        self.dwt = HaarDWT2D()
        self.stem_type = stem_type
        self.pool_energy = pool_energy

        if stem_type == "strided_patch":
            # Strided patch stem: 9ch -> embed_dim with 16x16 stride 16 (112 -> 7x7)
            self.proj = nn.Sequential(
                nn.Conv2d(9, embed_dim, kernel_size=16, stride=16),
                nn.BatchNorm2d(embed_dim),
                nn.GELU(),
            )
        else:
            # 3-stage DW-SE stem: 9ch @ 112x112 -> 7x7
            # Stage 1: stride=4 downsamples 112 -> 28
            self.stage1 = nn.Sequential(
                nn.Conv2d(9, 64, kernel_size=3, stride=4, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.GELU(),
                SqueezeExcitation(64),
            )
            # Stage 2: stride=2 downsamples 28 -> 14
            self.stage2 = DepthwiseSeparableBlock(64, 128, stride=2)
            # Stage 3: stride=2 downsamples 14 -> 7
            self.stage3 = DepthwiseSeparableBlock(128, embed_dim, stride=2)

    def forward(self, x, return_rho=False):
        """
        Args:
            x: (B*T, 3, 224, 224) flattened frames
            return_rho: if True, returns (out, rho) where rho is spectral energy ratio
        Returns:
            out: (B*T, embed_dim, 7, 7) frequency feature maps
            (optional) rho: (B*T, 1) spectral energy ratio
        """
        ll, lh, hl, hh = self.dwt(x)

        # Spectral Energy Ratio: rho = HF_energy / LF_energy (compression sensor)
        ll_energy = torch.mean(torch.abs(ll), dim=[1, 2, 3], keepdim=True)
        hf_energy = torch.mean(torch.abs(lh) + torch.abs(hl) + torch.abs(hh), dim=[1, 2, 3], keepdim=True)
        rho = (hf_energy / (ll_energy + 1e-6)).view(x.size(0), 1)  # (B*T, 1)

        if self.pool_energy:
            lh, hl, hh = torch.abs(lh), torch.abs(hl), torch.abs(hh)

        hf = torch.cat([lh, hl, hh], dim=1)  # (B*T, 9, 112, 112)

        if self.stem_type == "strided_patch":
            out = self.proj(hf)       # (B*T, embed_dim, 7, 7)
        else:
            out = self.stage1(hf)     # (B*T, 64, 28, 28)
            out = self.stage2(out)    # (B*T, 128, 14, 14)
            out = self.stage3(out)    # (B*T, embed_dim, 7, 7)

        if return_rho:
            return out, rho
        return out
