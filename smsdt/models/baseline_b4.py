"""
Stripped EfficientNet-B4 Baseline Classifier
===========================================
Pure frame-averaged EfficientNet-B4 classifier to verify pipeline integrity
and establish the true performance floor (expected >= 0.90 AUC on FF++ c23).
No frequency branch, no ViT, no contrastive/decorrelation losses.
"""
import torch
import torch.nn as nn
import timm
from einops import rearrange


class BaselineB4(nn.Module):
    """Stripped EfficientNet-B4 baseline with frame logit averaging."""
    def __init__(self, pretrained=True, drop_rate=0.2):
        super().__init__()
        self.backbone = timm.create_model(
            "efficientnet_b4",
            pretrained=pretrained,
            num_classes=1,
            drop_rate=drop_rate,
        )

    def set_gradient_checkpointing(self, enable=True):
        if hasattr(self.backbone, "set_grad_checkpointing"):
            self.backbone.set_grad_checkpointing(enable)

    def forward(self, frames, conf=None, return_features=False):
        """
        Args:
            frames: (B, T, 3, 224, 224) video chunks
            conf: (B, T) face detection confidences (ignored by baseline)
            return_features: returns dummy tuple to match SMSDT signature if True
        Returns:
            logit: (B,) chunk-level binary logit
        """
        B, T = frames.shape[:2]
        x = rearrange(frames, "b t c h w -> (b t) c h w")
        frame_logits = self.backbone(x)  # (B*T, 1)
        chunk_logits = rearrange(frame_logits.squeeze(-1), "(b t) -> b t", b=B, t=T)
        logit = chunk_logits.mean(dim=1)  # (B,)

        if return_features:
            return logit, None, None, None
        return logit
