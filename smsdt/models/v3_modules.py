"""S-MSDTv3 Architectural Modules.

Includes:
1. 2D Haar Discrete Wavelet Transform (haar_dwt) and Spatial Rich Model (SRM) kernels.
2. ResidualStream: High-frequency residual feature extractor with zero-init sigmoid gate
   and frequency auxiliary supervision.
3. EvidenceReadoutV3: K-query cross-attention, landmark-guided perioral prior bias,
   and Log-Sum-Exp (LSE) pooling over spatial patches and temporal frames.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def haar_dwt(x: torch.Tensor):
    """Single-level signed 2D Haar DWT.
    
    Args:
        x: (B, C, H, W) tensor
    Returns:
        ll, lh, hl, hh each of shape (B, C, H // 2, W // 2)
    """
    x00 = x[:, :, 0::2, 0::2]
    x01 = x[:, :, 0::2, 1::2]
    x10 = x[:, :, 1::2, 0::2]
    x11 = x[:, :, 1::2, 1::2]
    ll = (x00 + x01 + x10 + x11) * 0.5
    lh = (x00 + x01 - x10 - x11) * 0.5  # vertical high frequencies
    hl = (x00 - x01 + x10 - x11) * 0.5  # horizontal high frequencies
    hh = (x00 - x01 - x10 + x11) * 0.5  # diagonal high frequencies
    return ll, lh, hl, hh


def _srm_kernels() -> torch.Tensor:
    """Standard 5x5 Spatial Rich Model (SRM) high-pass filter kernels.
    
    Returns:
        Tensor of shape (3, 5, 5) with float32 weights.
    """
    k1 = torch.tensor([
        [0, 0, 0, 0, 0],
        [0, -1, 2, -1, 0],
        [0, 2, -4, 2, 0],
        [0, -1, 2, -1, 0],
        [0, 0, 0, 0, 0],
    ], dtype=torch.float32) / 4.0

    k2 = torch.tensor([
        [-1, 2, -2, 2, -1],
        [2, -6, 8, -6, 2],
        [-2, 8, -12, 8, -2],
        [2, -6, 8, -6, 2],
        [-1, 2, -2, 2, -1],
    ], dtype=torch.float32) / 12.0

    k3 = torch.tensor([
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 1, -2, 1, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ], dtype=torch.float32) / 2.0

    return torch.stack([k1, k2, k3])  # (3, 5, 5)


class ResidualStream(nn.Module):
    """High-frequency (SRM + Haar DWT) residual stream on the 16x16 patch grid.
    
    Injects high-frequency spatial noise and boundary residuals into projected CLIP tokens
    via a zero-initialized sigmoid gate, supervised with an auxiliary frequency loss.
    """

    def __init__(
        self,
        dim: int = 384,
        c_r: int = 128,
        grid: int = 16,
        init_bias: float = -4.0,
        zero_gate: bool = False,
    ):
        super().__init__()
        self.grid = grid
        self.dim = dim

        # 3 SRM filters applied per channel (3 RGB channels -> 9 channels)
        srm_3 = _srm_kernels().unsqueeze(1)  # (3, 1, 5, 5)
        # Repeat along dim 0 so channels [0..2] apply to R, [3..5] to G, [6..8] to B with groups=3
        srm_weights = torch.cat([srm_3, srm_3, srm_3], dim=0)  # (9, 1, 5, 5)
        self.register_buffer("srm", srm_weights)

        # Lightweight CNN stem mapping 21 high-pass channels to c_r=128
        # Input resolution is H/2, W/2 (112x112 for 224x224 input)
        self.cnn = nn.Sequential(
            nn.Conv2d(21, 64, kernel_size=3, stride=2, padding=1),   # 112 -> 56
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv2d(64, c_r, kernel_size=3, stride=2, padding=1),  # 56 -> 28
            nn.GroupNorm(8, c_r),
            nn.GELU(),
        )
        self.to_dim = nn.Conv2d(c_r, dim, kernel_size=1)
        self.ln_r = nn.LayerNorm(dim)

        # Gate combining projected CLIP token s and residual token r
        self.gate = nn.Linear(2 * dim, dim)
        nn.init.zeros_(self.gate.weight)
        bias_val = -12.0 if zero_gate else init_bias
        nn.init.constant_(self.gate.bias, bias_val)

        # Frequency-only auxiliary supervision head
        self.aux = nn.Linear(dim, 1)

    def residual_input(self, x01: torch.Tensor) -> torch.Tensor:
        """Extract multi-scale high-frequency subbands from denormalized [0, 1] frames.
        
        Args:
            x01: (N, 3, H, W) float tensor with values in [0, 1]
        Returns:
            Concatenated high-frequency map of shape (N, 21, H // 2, W // 2)
        """
        N, C, H, W = x01.shape
        # 1. 3 SRM high-pass filters per RGB channel (9 channels)
        srm = F.conv2d(x01, self.srm, padding=2, groups=C).clamp(-2.0, 2.0)  # (N, 9, H, W)
        srm_sub = F.avg_pool2d(srm, kernel_size=2, stride=2)                 # (N, 9, H/2, W/2)

        # 2. Level-1 2D Haar DWT high subbands (LH, HL, HH -> 9 channels)
        ll1, lh1, hl1, hh1 = haar_dwt(x01)
        l1 = torch.cat([lh1, hl1, hh1], dim=1)                               # (N, 9, H/2, W/2)

        # 3. Level-2 2D Haar DWT diagonal subband (HH2 -> 3 channels)
        _, _, _, hh2 = haar_dwt(ll1)                                         # (N, 3, H/4, W/4)
        hh2_up = F.interpolate(hh2, size=(H // 2, W // 2), mode="nearest")   # (N, 3, H/2, W/2)

        # Concatenate: 9 + 9 + 3 = 21 channels at 112x112
        return torch.cat([srm_sub, l1, hh2_up], dim=1)                       # (N, 21, H/2, W/2)

    def forward(self, x01: torch.Tensor, s: torch.Tensor):
        """
        Args:
            x01: (BT, 3, 224, 224) denormalized frames in [0, 1]
            s:   (BT, 256, dim) projected foundation tokens
        Returns:
            fused_s:   (BT, 256, dim) tokens with gated high-frequency injection
            aux_logit: (BT,) frame-level frequency auxiliary prediction
        """
        feats = self.residual_input(x01)
        r = self.cnn(feats)
        r = F.adaptive_avg_pool2d(r, (self.grid, self.grid))
        r = self.to_dim(r).flatten(2).transpose(1, 2)  # (BT, 256, dim)
        r = self.ln_r(r)

        # Gated fusion
        g = torch.sigmoid(self.gate(torch.cat([s, r], dim=-1)))
        fused_s = s + g * r

        # Auxiliary deep supervision on frequency features
        aux_logit = self.aux(r.mean(dim=1)).squeeze(-1)  # (BT,)
        return fused_s, aux_logit


class EvidenceReadoutV3(nn.Module):
    """K-query cross-attention with landmark-guided mouth prior bias and Log-Sum-Exp pooling.
    
    Blends localized Log-Sum-Exp (LSE) patch anomaly logits with global multi-query
    cross-attention logits, enabling simultaneous sensitivity to micro-boundary seams
    and diffuse expression anomalies.
    """

    def __init__(
        self,
        dim: int = 384,
        heads: int = 6,
        n_queries: int = 4,
        r0: float = 3.0,
        topk_frac: float = 0.05,
    ):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.n_queries = n_queries
        self.topk_frac = topk_frac

        # K learned query vectors for multi-view video attention
        self.q = nn.Parameter(torch.randn(n_queries, dim) * 0.02)
        self.kv = nn.Linear(dim, 2 * dim)

        # Multi-layer patch anomaly classifier
        self.patch_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, 1),
        )

        # Landmark perioral prior gain (starts at 0.0, completely neutral at init)
        self.prior_gain = nn.Parameter(torch.zeros(()))

        # Learnable LSE temperature: r = exp(log_r), r0=3.0
        self.log_r = nn.Parameter(torch.tensor(math.log(r0)))

        # Cross-attention output projection
        self.out = nn.Sequential(
            nn.LayerNorm(dim * n_queries),
            nn.Linear(dim * n_queries, 1),
        )

        # Learnable balance between LSE pooling logit and query attention logit
        self.mix = nn.Parameter(torch.zeros(()))

    @staticmethod
    def lse(x: torch.Tensor, r: torch.Tensor, dim: int) -> torch.Tensor:
        """Numerically stable Log-Sum-Exp pooling: (1/r) * (logsumexp(r * x) - log(n))."""
        n = x.shape[dim]
        return (torch.logsumexp(r * x, dim=dim) - math.log(n)) / r

    def forward(self, z: torch.Tensor, prior: torch.Tensor | None = None):
        """
        Args:
            z:     (B, T, N, D) spatio-temporal tokens (N=256, D=384)
            prior: optional (B, T, N) soft perioral prior in [0, 1]
        Returns:
            logit:        (B,) video-level manipulation prediction logit
            patch_logits: (B, T, N) localized patch anomaly logits
        """
        B, T, N, D = z.shape
        TN = T * N

        # 1. Patch-level anomaly logits
        patch_logits = self.patch_head(z).squeeze(-1)  # (B, T, N)

        # 2. Multi-query cross-attention over spatio-temporal tokens
        k, v = self.kv(z.reshape(B, TN, D)).chunk(2, dim=-1)
        H = self.heads
        dh = D // H

        q = self.q.unsqueeze(0).expand(B, -1, -1)                  # (B, K, D)
        qh = q.view(B, -1, H, dh).transpose(1, 2)                  # (B, H, K, dh)
        kh = k.view(B, TN, H, dh).transpose(1, 2)                  # (B, H, TN, dh)
        vh = v.view(B, TN, H, dh).transpose(1, 2)                  # (B, H, TN, dh)

        att = (qh @ kh.transpose(-1, -2)) / math.sqrt(dh)          # (B, H, K, TN)

        # Inject soft landmark perioral prior as additive attention log-bias
        if prior is not None:
            prior_flat = prior.reshape(B, 1, 1, TN).clamp(min=1e-5)
            att = att + self.prior_gain * torch.log(prior_flat)

        att = att.softmax(dim=-1)                                  # (B, H, K, TN)
        pooled = (att @ vh).transpose(1, 2).reshape(B, -1)         # (B, K * D)
        logit_att = self.out(pooled).squeeze(-1)                   # (B,)

        # 3. Log-Sum-Exp pooling across spatial patches then temporal frames
        r = self.log_r.exp().clamp(min=0.1, max=20.0)
        spatial_lse = self.lse(patch_logits, r, dim=2)             # (B, T)
        logit_lse = self.lse(spatial_lse, r, dim=1)                # (B,)

        # 4. Learned dynamic blend between LSE pooling and attention readout
        w = torch.sigmoid(self.mix)
        logit = w * logit_lse + (1.0 - w) * logit_att
        return logit, patch_logits
