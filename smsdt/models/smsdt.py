"""
S-MSDT v2: Scale-Invariance + Multi-Scale Feature Differences + Spatio-Temporal Wavelet Guard.
Full assembled model combining:
- EfficientNet-B4 FPN RGB branch with High-Res PSIT (14x14 = 196 tokens)
- Spatio-temporal Wavelet branch with 3-stage DW-SE stem & temporal high-pass
- Compression-Aware Spectral Gating (CASG) with local cross-attention
- Temporal-First N-MSSS ViT with asymmetric Q/K temporal attention
- Sharp Attentional Readout (temperature-scaled spatial-temporal pooling)
"""
import torch
import torch.nn as nn
from einops import rearrange
from .rgb_branch import RGBBranch
from .wavelet_branch import WaveletBranch
from .tokenizer import PSIT, FrequencyStem, GatedFusion
from .nmsss_vit import NMSSSBlock


class SharpAttentionalReadout(nn.Module):
    """Temperature-scaled Softmax Attentional Pooling:
    Replaces uniform mean-pooling with learned attention weights across patches
    and frames, preventing localized forgeries (e.g. mouth in NeuralTextures)
    from being diluted by authentic facial skin.
    """
    def __init__(self, embed_dim=256, temperature=0.2):
        super().__init__()
        self.temperature = temperature
        self.spatial_attn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4),
            nn.Tanh(),
            nn.Linear(embed_dim // 4, 1),
        )
        self.temporal_attn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4),
            nn.Tanh(),
            nn.Linear(embed_dim // 4, 1),
        )

    def forward(self, z):
        """
        Args:
            z: (B, T, N, D) transformer output tokens
        Returns:
            z_video: (B, D) aggregated video representation
        """
        # 1. Spatial sharp attention across N patches
        s_scores = self.spatial_attn(z) / self.temperature      # (B, T, N, 1)
        s_weights = torch.softmax(s_scores, dim=2)               # (B, T, N, 1)
        z_frame = (z * s_weights).sum(dim=2)                     # (B, T, D)

        # 2. Temporal sharp attention across T frames
        t_scores = self.temporal_attn(z_frame) / self.temperature # (B, T, 1)
        t_weights = torch.softmax(t_scores, dim=1)                 # (B, T, 1)
        z_video = (z_frame * t_weights).sum(dim=1)                 # (B, D)

        return z_video


class SMSDT(nn.Module):
    def __init__(self, embed_dim=256, depth=4, heads=8, num_classes=1, grid=14,
                 stem_type="dw_separable_se", pool_energy=True, pool_temperature=0.2):
        super().__init__()
        self.grid = grid
        self.embed_dim = embed_dim

        # Dual-stream feature extractors
        self.rgb_branch = RGBBranch(out_channels=embed_dim)
        self.wavelet_branch = WaveletBranch(embed_dim=embed_dim, stem_type=stem_type, pool_energy=pool_energy)

        # High-res tokenizers (14x14 = 196 tokens)
        self.psit = PSIT(embed_dim=embed_dim, grid=grid)
        self.freq_stem = FrequencyStem(embed_dim=embed_dim, grid=grid)
        self.fusion = GatedFusion(embed_dim=embed_dim, use_cross_attn=True)

        num_tokens = grid * grid
        self.pos_embed = nn.Parameter(torch.zeros(1, 1, num_tokens, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Temporal-First Decomposed ViT Blocks
        self.blocks = nn.ModuleList([
            NMSSSBlock(dim=embed_dim, heads=heads) for _ in range(depth)
        ])

        # Sharp Attentional Readout & Head
        self.readout = SharpAttentionalReadout(embed_dim=embed_dim, temperature=pool_temperature)
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim // 2, num_classes),
        )

    def set_gradient_checkpointing(self, enable=True):
        for blk in self.blocks:
            blk.set_gradient_checkpointing(enable)

    def forward(self, frames, conf, return_features=False):
        """
        Args:
            frames: (B, T, 3, 224, 224) face crop chunks
            conf:   (B, T) SCRFD landmark confidence per frame
            return_features: if True, also return intermediate z_rgb, z_dwt, z_video
        Returns:
            logit: (B,) chunk-level logit
            (optional) z_rgb_pooled: (B, D) pooled pre-fusion RGB features
            (optional) z_dwt_pooled: (B, D) pooled pre-fusion wavelet features
            (optional) z_video: (B, D) post-ViT learned representations
        """
        B, T = frames.shape[:2]
        x = rearrange(frames, "b t c h w -> (b t) c h w")

        # --- Dual-stream feature extraction ---
        fpn = self.rgb_branch(x)                    # list of (B*T, D, h, w)
        z_rgb = self.psit(fpn)                      # (B*T, N=196, D)

        # Spatio-Temporal Wavelet Frequency Guard with spectral ratio rho
        wv, rho = self.wavelet_branch(frames, return_rho=True)  # (B*T, D, 14, 14), (B*T, 1)
        z_dwt = self.freq_stem(wv)                              # (B*T, N=196, D)

        # --- Compression-Aware Cross-Domain Fusion ---
        z = self.fusion(z_rgb, z_dwt, rho=rho)                  # (B*T, N=196, D)
        z = rearrange(z, "(b t) n d -> b t n d", b=B, t=T)
        z = z + self.pos_embed

        # --- Temporal-First N-MSSS ViT Backbone ---
        for blk in self.blocks:
            z = blk(z, conf)

        # --- Sharp Attentional Readout ---
        z_video = self.readout(z)                   # (B, D)
        logit = self.head(z_video).squeeze(-1)      # (B,)

        if return_features:
            # Pre-fusion features for cross-domain orthogonality / decorrelation loss
            z_rgb_r = rearrange(z_rgb, "(b t) n d -> b t n d", b=B, t=T)
            z_dwt_r = rearrange(z_dwt, "(b t) n d -> b t n d", b=B, t=T)
            z_rgb_pooled = z_rgb_r.mean(dim=2).mean(dim=1)  # (B, D)
            z_dwt_pooled = z_dwt_r.mean(dim=2).mean(dim=1)  # (B, D)
            return logit, z_rgb_pooled, z_dwt_pooled, z_video

        return logit
