import cv2
import numpy as np

# Standard ArcFace 5-point template (at 112x112 canonical resolution)
REF_5PT = np.array([
    [38.2946, 51.6963], [73.5318, 51.5014],
    [56.0252, 71.7366], [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)


def loose_template(out=256, margin=1.3):
    """ArcFace 5-point template rescaled so the face occupies 1/margin of the crop."""
    t = (REF_5PT - 56.0) / margin + 56.0      # shrink about the template centre
    return t * (out / 112.0)


def crop_frame(frame_bgr, kps5, out=256, margin=1.3):
    """Align and crop a frame using 5-point landmarks to a loose canonical template.
    
    Args:
        frame_bgr: (H, W, 3) BGR uint8 image
        kps5: (5, 2) facial landmarks
        out: output square resolution
        margin: crop margin (1.3 leaves surrounding jaw/hair context)
        
    Returns:
        crop_rgb: (out, out, 3) RGB uint8 image, or None if alignment fails
        M: (2, 3) affine partial transformation matrix, or None
    """
    dst = loose_template(out=out, margin=margin)
    M, _ = cv2.estimateAffinePartial2D(kps5.astype(np.float32), dst, method=cv2.LMEDS)
    if M is None:
        return None, None
    crop = cv2.warpAffine(frame_bgr, M, (out, out), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT101)
    return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), M


def warp_points(pts, M):
    """Apply a 2x3 affine matrix M to (K, 2) points.
    
    Args:
        pts: (K, 2) float points in original frame coordinates
        M: (2, 3) affine transformation matrix
        
    Returns:
        warped: (K, 2) float points in crop coordinates
    """
    return pts @ M[:, :2].T + M[:, 2]


def align_face(frame, kps, out_size=224):
    """Legacy backward-compatible function."""
    dst = REF_5PT * (out_size / 112.0)
    M, _ = cv2.estimateAffinePartial2D(kps.astype(np.float32), dst, method=cv2.LMEDS)
    if M is None:
        return None
    return cv2.warpAffine(frame, M, (out_size, out_size), borderValue=0.0)
