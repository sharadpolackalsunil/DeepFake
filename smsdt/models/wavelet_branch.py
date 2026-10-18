"""
S-MSDT v2 Spatio-Temporal Wavelet Frequency Guard
================================================
- Single-level 2D Haar DWT (LH, HL, HH subbands, discarding LL for spatial path).
- High-to-Low Spectral Energy Ratio (rho_t) as parameter-free compression sensor.
- 1D Temporal High-Pass filtering along time dimension T to capture inter-frame flicker.
- 3-Stage Depthwise-Separable + Squeeze-and-Excitation (SE) Stem mapping (112x112) -> (14x14).
"""
import torch
import torch.nn as nn
from einops import rearrange


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
    """Depthwise-Separable Convolution with BatchNorm, GELU, and Squeeze-Excitation."""
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
    """
    Spatio-Temporal Wavelet Frequency Guard:
    Processes 2D Haar spatial subbands + 1D temporal difference through a
    3-stage depthwise-separable stem to produce (B*T, embed_dim, 14, 14) frequency tokens.
    """
    def __init__(self, embed_dim=256, stem_type="dw_separable_se", pool_energy=True):
        super().__init__()
        self.dwt = HaarDWT2D()
        self.stem_type = stem_type
        self.pool_energy = pool_energy
        self.embed_dim = embed_dim

        if self.stem_type == "strided_patch":
            # Strided patch stem (9 channels -> embed_dim with 16x16 stride 16 for 112 -> 7x7)
            self.proj = nn.Sequential(
                nn.Conv2d(9, embed_dim, kernel_size=16, stride=16),
                nn.BatchNorm2d(embed_dim),
                nn.GELU(),
            )
        else:
            # 3-stage DW-SE stem: (112x112) -> (56x56) -> (28x28) -> (14x14)
            # Input has 12 channels (9 spatial LH,HL,HH + 3 temporal high-pass)
            self.stage1 = nn.Sequential(
                nn.Conv2d(12, 64, kernel_size=3, stride=2, padding=1, bias=False),  # 112 -> 56
                nn.BatchNorm2d(64),
                nn.GELU(),
                SqueezeExcitation(64),
            )
            self.stage2 = DepthwiseSeparableBlock(64, 128, stride=2)                 # 56 -> 28
            self.stage3 = DepthwiseSeparableBlock(128, embed_dim, stride=2)           # 28 -> 14

    def forward(self, x, num_frames=None, return_rho=False):
        """
        Args:
            x: (B*T, 3, 224, 224) or (B, T, 3, 224, 224)
            num_frames: optional T if x is already flattened to (B*T, ...)
            return_rho: if True, returns (out, rho) where rho is spectral energy ratio
        Returns:
            out: (B*T, embed_dim, 14, 14) frequency feature maps
            (optional) rho: (B*T, 1) spectral energy ratio
        """
        is_5d = (x.dim() == 5)
        if is_5d:
            B, T, C, H, W = x.shape
            x_flat = rearrange(x, "b t c h w -> (b t) c h w")
        else:
            x_flat = x
            BT = x.size(0)
            T = num_frames if num_frames is not None else 1
            B = BT // T

        ll, lh, hl, hh = self.dwt(x_flat)

        # Compute Spectral Energy Ratio rho_t = (||LH||_1 + ||HL||_1 + ||HH||_1) / (||LL||_1 + eps)
        ll_energy = torch.mean(torch.abs(ll), dim=[1, 2, 3], keepdim=True)  # (B*T, 1, 1, 1)
        hf_energy = torch.mean(torch.abs(lh) + torch.abs(hl) + torch.abs(hh), dim=[1, 2, 3], keepdim=True)
        rho = (hf_energy / (ll_energy + 1e-6)).view(x_flat.size(0), 1)     # (B*T, 1)

        if self.pool_energy:
            lh, hl, hh = torch.abs(lh), torch.abs(hl), torch.abs(hh)

        spatial_hf = torch.cat([lh, hl, hh], dim=1)  # (B*T, 9, 112, 112)

        # 1D Temporal High-Pass across T frames on LL subband
        if T > 1 and (B * T == x_flat.size(0)):
            ll_r = rearrange(ll, "(b t) c h w -> b t c h w", b=B, t=T)
            # Forward diff at t=0, backward diff at t >= 1
            f_temp_0 = (ll_r[:, 1:2] - ll_r[:, 0:1])  # (B, 1, C, H/2, W/2)
            f_temp_rest = (ll_r[:, 1:] - ll_r[:, :-1]) # (B, T-1, C, H/2, W/2)
            f_temp = torch.cat([f_temp_0, f_temp_rest], dim=1)  # (B, T, C, H/2, W/2)
            f_temp = torch.abs(rearrange(f_temp, "b t c h w -> (b t) c h w"))
        else:
            # Fallback for single frames
            f_temp = torch.zeros_like(ll)

        # Combine 9 spatial + 3 temporal high-frequency channels = 12 channels
        hf_12 = torch.cat([spatial_hf, f_temp], dim=1)  # (B*T, 12, 112, 112)

        if self.stem_type == "strided_patch":
            out = self.proj(spatial_hf)  # (B*T, embed_dim, 7, 7)
        else:
            # 3-stage DW-SE stem: (112x112) -> (14x14)
            x_stem = self.stage1(hf_12)
            x_stem = self.stage2(x_stem)
            out = self.stage3(x_stem)    # (B*T, embed_dim, 14, 14)

        if return_rho:
            return out, rho
        return out
