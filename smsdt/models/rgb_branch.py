import torch
import torch.nn as nn
import timm

class RGBBranch(nn.Module):
    def __init__(self, out_channels=256):
        super().__init__()
        self.backbone = timm.create_model(
            "efficientnet_b4", pretrained=True, features_only=True,
            out_indices=(2, 3, 4),  # roughly P2, P3, P4 strides
        )
        chs = self.backbone.feature_info.channels()
        self.lateral = nn.ModuleList([nn.Conv2d(c, out_channels, 1) for c in chs])
        self.smooth = nn.ModuleList([nn.Conv2d(out_channels, out_channels, 3, padding=1) for _ in chs])

    def forward(self, x):
        feats = self.backbone(x)                    # [P2, P3, P4], each (B, C, H, W)
        laterals = [l(f) for l, f in zip(self.lateral, feats)]
        # top-down fusion
        for i in range(len(laterals) - 1, 0, -1):
            up = nn.functional.interpolate(laterals[i], size=laterals[i - 1].shape[-2:], mode="nearest")
            laterals[i - 1] = laterals[i - 1] + up
        fpn_out = [s(l) for s, l in zip(self.smooth, laterals)]
        return fpn_out   # list of (B, 256, H_i, W_i)
