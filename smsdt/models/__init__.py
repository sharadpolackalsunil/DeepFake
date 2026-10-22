"""S-MSDT model architectures."""
from .smsdt import SMSDT, EvidenceReadout
from .nmsss_vit import NMSSSBlock, TemporalDiff
from .clip_encoder import CLIP_MEAN, CLIP_STD

try:
    from .clip_encoder import CLIPFrameEncoder
    from .smsdt_v2 import SMSDTv2
    from .smsdt_v3 import SMSDTv3
    from .v3_modules import ResidualStream, EvidenceReadoutV3, haar_dwt
    _has_clip = True
except ImportError:
    CLIPFrameEncoder = None
    SMSDTv2 = None
    SMSDTv3 = None
    ResidualStream = None
    EvidenceReadoutV3 = None
    haar_dwt = None
    _has_clip = False

__all__ = [
    "SMSDT",
    "SMSDTv2",
    "SMSDTv3",
    "EvidenceReadout",
    "EvidenceReadoutV3",
    "ResidualStream",
    "haar_dwt",
    "NMSSSBlock",
    "TemporalDiff",
    "CLIPFrameEncoder",
    "CLIP_MEAN",
    "CLIP_STD",
]
