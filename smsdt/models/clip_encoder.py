"""CLIP ViT-L/14 Frame Encoder with LayerNorm-only Parameter-Efficient Fine-Tuning."""
import os
import torch
import torch.nn as nn

try:
    from transformers import CLIPVisionConfig, CLIPVisionModel
except ImportError:
    CLIPVisionConfig = None
    CLIPVisionModel = None

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


class CLIPFrameEncoder(nn.Module):
    """(BT, 3, 224, 224) in CLIP normalization -> cls (BT, 1024), patches (BT, 256, 1024).

    Only LayerNorm parameters require grad when tune == 'layernorm' (< 0.03% of total parameters).
    Non-reentrant gradient checkpointing ensures backpropagation flows through frozen layers
    to LayerNorms without memory explosion.
    """

    def __init__(
        self,
        name: str = "openai/clip-vit-large-patch14",
        tune: str = "layernorm",
        grad_ckpt: bool = True,
        config = None,
    ):
        super().__init__()
        if CLIPVisionModel is None:
            raise ImportError(
                "Hugging Face transformers is required for CLIPFrameEncoder. "
                "Install it with: pip install 'transformers>=4.40' safetensors"
            )

        if config is not None:
            self.vit = CLIPVisionModel(config)
        else:
            self.vit = CLIPVisionModel.from_pretrained(name)

        # Freeze everything except LayerNorm parameters when tune == "layernorm"
        for n, p in self.vit.named_parameters():
            p.requires_grad = (tune == "layernorm" and "norm" in n.lower()) or (tune == "full")

        if grad_ckpt:
            # non-reentrant checkpointing: gradients reach LayerNorms even though input
            # image and intermediate weights do not require grad
            self.vit.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

    def forward(self, x: torch.Tensor):
        out = self.vit(pixel_values=x)
        cls = out.pooler_output  # post-LayerNorm CLS, (BT, 1024)
        patches = out.last_hidden_state[:, 1:]  # (BT, 256, 1024) for 224 / 14 = 16x16
        return cls, patches
