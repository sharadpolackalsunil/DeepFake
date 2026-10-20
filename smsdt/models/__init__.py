"""S-MSDT model architectures."""
from .smsdt import SMSDT, EvidenceReadout
from .nmsss_vit import NMSSSBlock, TemporalDiff
from .clip_encoder import CLIP_MEAN, CLIP_STD

try:
    from .clip_encoder import CLIPFrameEncoder
    from .smsdt_v2 import SMSDTv2
    _has_clip = True
except ImportError:
    CLIPFrameEncoder = None
    SMSDTv2 = None
    _has_clip = False

__all__ = [
    "SMSDT",
    "SMSDTv2",
    "EvidenceReadout",
    "NMSSSBlock",
    "TemporalDiff",
    "CLIPFrameEncoder",
    "CLIP_MEAN",
    "CLIP_STD",
]
