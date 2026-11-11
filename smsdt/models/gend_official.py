"""Official GenD model definition (vendored) and an adapter for the S-MSDT evaluator.

Vendored from https://github.com/yermandy/GenD, file src/hf/modeling_gend.py, at commit
387a42266dd385fbe8f7626c5d3aa03eea3bfaab (the commit pinned in docs/decisions.md).
Copyright (c) 2025 Andy (GenD authors). MIT License; see third_party/GenD/LICENSE.
The vendored part is unchanged apart from this docstring and the imports being lazy.

Why this exists (gate G1, docs/decisions.md section 3): our re-implementation `gend_ours`
scores about 0.83 Celeb-DF AUC, against the published 0.946 (5-seed mean) / 0.960. The
pre-registered Branch B says claims must then be made relative to `gend_official`, the
official weights, evaluated through OUR pipeline. This adapter makes that possible:

    python -m smsdt.eval_cross --ckpt hf:yermandy/GenD_CLIP_L_14 --T 1 ...

Official output convention (third_party/GenD/app/run.py): softmax over 2 classes,
p_fake = probs[1]. The adapter returns the binary logit log(p1/p0) = z1 - z0, averaged
over the T frames of a clip, so it plugs into the same scoring code as our models.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import PretrainedConfig, PreTrainedModel


class LinearProbe(nn.Module):
    def __init__(self, input_dim, num_classes, normalize_inputs=False):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)
        self.normalize_inputs = normalize_inputs

    def forward(self, x: torch.Tensor, **kwargs):
        if self.normalize_inputs:
            x = F.normalize(x, p=2, dim=1)

        return self.linear(x)


class CLIPEncoder(nn.Module):
    def __init__(self, model_name="openai/clip-vit-large-patch14"):
        super().__init__()

        from transformers import CLIPModel, CLIPProcessor

        try:
            self._preprocess = CLIPProcessor.from_pretrained(model_name)
        except Exception:
            self._preprocess = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

        clip: CLIPModel = CLIPModel.from_pretrained(model_name)

        # take vision model from CLIP, maps image to vision_embed_dim
        self.vision_model = clip.vision_model

        self.model_name = model_name

        self.features_dim = self.vision_model.config.hidden_size

        # take visual_projection, maps vision_embed_dim to projection_dim
        self.visual_projection = clip.visual_projection

    def preprocess(self, image: Image) -> torch.Tensor:
        return self._preprocess(images=image, return_tensors="pt")["pixel_values"][0]

    def forward(self, preprocessed_images: torch.Tensor) -> torch.Tensor:
        return self.vision_model(preprocessed_images).pooler_output

    def get_features_dim(self):
        return self.features_dim


class DINOEncoder(nn.Module):
    def __init__(self, model_name="facebook/dinov2-with-registers-base"):
        super().__init__()

        from transformers import AutoImageProcessor, AutoModel, Dinov2Model, Dinov2WithRegistersModel

        self._preprocess = AutoImageProcessor.from_pretrained(model_name)
        self.backbone: Dinov2Model | Dinov2WithRegistersModel = AutoModel.from_pretrained(model_name)

        self.features_dim = self.backbone.config.hidden_size

    def preprocess(self, image: Image) -> torch.Tensor:
        return self._preprocess(images=image, return_tensors="pt")["pixel_values"][0]

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.backbone(inputs).last_hidden_state[:, 0]

    def get_features_dim(self) -> int:
        return self.features_dim


class PerceptionEncoder(nn.Module):
    def __init__(self, model_name="vit_pe_core_large_patch14_336"):
        super().__init__()

        import timm
        from timm.models.eva import Eva

        self.backbone: Eva = timm.create_model(
            model_name,
            pretrained=True,
            dynamic_img_size=True,
        )

        # Get model specific transforms (normalization, resize)
        data_config = timm.data.resolve_model_data_config(self.backbone)
        data_config["input_size"] = (3, 224, 224)

        self._preprocess = timm.data.create_transform(**data_config, is_training=False)

        # Remove head
        self.backbone.head = nn.Identity()

        self.features_dim = self.backbone.num_features

    def preprocess(self, image: Image.Image) -> torch.Tensor:
        return self._preprocess(image)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.backbone(inputs)

    def get_features_dim(self) -> int:
        return self.features_dim


class GenDConfig(PretrainedConfig):
    model_type = "GenD"

    def __init__(self, backbone: str = "openai/clip-vit-large-patch14", head: str = "linear", **kwargs):
        super().__init__(**kwargs)
        self.backbone = backbone
        self.head = head


class GenD(PreTrainedModel):
    config_class = GenDConfig

    def __init__(self, config):
        super().__init__(config)

        self.head = config.head
        self.backbone = config.backbone
        self.config = config

        self._init_feature_extractor()
        self._init_head()

    def _init_feature_extractor(self):
        backbone = self.backbone
        backbone_lowercase = backbone.lower()

        if "clip" in backbone_lowercase:
            self.feature_extractor = CLIPEncoder(backbone)

        elif "vit_pe" in backbone_lowercase:
            self.feature_extractor = PerceptionEncoder(backbone)

        elif "dino" in backbone_lowercase:
            self.feature_extractor = DINOEncoder(backbone)

        else:
            raise ValueError(f"Unknown backbone: {backbone}")

    def _init_head(self):
        features_dim = self.feature_extractor.get_features_dim()

        match self.head:
            case "linear":
                self.model = LinearProbe(features_dim, 2)

            case "LinearNorm":
                self.model = LinearProbe(features_dim, 2, True)

            case _:
                raise ValueError(f"Unknown head: {self.head}")

    def forward(self, inputs: torch.Tensor):
        features = self.feature_extractor(inputs)
        outputs = self.model.forward(features)
        return outputs


# ---------------------------------------------------------------------------
# Adapter (not part of the vendored file)
# ---------------------------------------------------------------------------

class OfficialGenDAdapter(nn.Module):
    """Wrap the official GenD model so it behaves like our detectors.

    Input:  (N, 3, 224, 224) or (N, T, 3, 224, 224), CLIP-normalised (OpenAI mean/std,
            identical to CLIPProcessor's normalisation).
    Output: (logit, None) with logit of shape (N,) = z_fake - z_real, mean over T.
    """

    def __init__(self, gend_model: nn.Module, fake_index: int = 1):
        super().__init__()
        self.gend = gend_model
        self.fake_index = int(fake_index)
        self.real_index = 1 - self.fake_index

    @staticmethod
    def binary_logit(two_class_logits: torch.Tensor, fake_index: int = 1) -> torch.Tensor:
        """log(p_fake / p_real) for a 2-class softmax = z_fake - z_real."""
        return two_class_logits[:, fake_index] - two_class_logits[:, 1 - fake_index]

    def forward(self, x: torch.Tensor, *args, **kwargs):
        five_d = x.ndim == 5
        if five_d:
            n, t = x.shape[:2]
            x = x.reshape(n * t, *x.shape[2:])
        z = self.gend(x)
        if hasattr(z, "logits"):
            z = z.logits
        logit = self.binary_logit(z.float(), self.fake_index)
        if five_d:
            logit = logit.view(n, t).mean(dim=1)
        return logit, None


def load_official_gend(repo_id: str = "yermandy/GenD_CLIP_L_14") -> OfficialGenDAdapter:
    """Download (or read from the HF cache) the official weights and wrap them.

    Requires `transformers` and network access to huggingface.co (or a warm cache;
    set HF_HUB_OFFLINE=1 to force the cache).
    """
    model = GenD.from_pretrained(repo_id)
    model.eval()
    return OfficialGenDAdapter(model)
