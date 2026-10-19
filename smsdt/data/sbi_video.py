"""Clip-level self-blended images (SBI) for video deepfake detection.

Samples blend and perturbation parameters ONCE per clip to prevent frame-to-frame
temporal flicker shortcuts. Uses 106-point facial landmarks to generate convex hull masks.
"""
import random
import cv2
import numpy as np


def _hull_mask(lmk, h, w):
    """Generate binary face mask from 106-point facial landmark convex hull."""
    m = np.zeros((h, w), np.float32)
    pts = cv2.convexHull(lmk.astype(np.int32))
    cv2.fillConvexPoly(m, pts, 1.0)
    return m


def _params(h):
    """Sample clip-level blend parameters once for all T frames."""
    return dict(
        ksize=random.choice([15, 25, 35]) | 1,          # mask blur (must be odd)
        dilate=random.randint(-int(0.04 * h), int(0.04 * h)),
        ratio=random.choice([0.25, 0.5, 0.75, 1.0, 1.0, 1.0]),
        hsv=(random.uniform(-0.03, 0.03), random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1)),
        bc=(random.uniform(0.9, 1.1), random.uniform(-15, 15)),
        scale=random.uniform(0.95, 1.05),
        shift=(random.uniform(-0.02, 0.02) * h, random.uniform(-0.02, 0.02) * h),
        sharpen=random.random() < 0.5,
    )


def _source(frame, p):
    """Slightly altered copy of the face: colour, contrast, sharpness, small scale/shift."""
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + p["hsv"][0] * 180) % 180
    hsv[..., 1] = np.clip(hsv[..., 1] * (1 + p["hsv"][1]), 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] * (1 + p["hsv"][2]), 0, 255)
    src = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)
    src = np.clip(src * p["bc"][0] + p["bc"][1], 0, 255)
    if p["sharpen"]:
        blur = cv2.GaussianBlur(src, (0, 0), 1.0)
        src = np.clip(src * 1.5 - blur * 0.5, 0, 255)
    M = np.float32([[p["scale"], 0, p["shift"][0] + (1 - p["scale"]) * w / 2],
                    [0, p["scale"], p["shift"][1] + (1 - p["scale"]) * h / 2]])
    return cv2.warpAffine(src, M, (w, h), borderMode=cv2.BORDER_REFLECT101)


def sbi_clip(clip_u8, lmk106):
    """Synthesize self-blended video clip from pristine real frames.
    
    Args:
        clip_u8: (T, H, W, 3) uint8 RGB frames
        lmk106: (T, 106, 2) float32 landmark crop coordinates
        
    Returns:
        fake_clip: (T, H, W, 3) uint8 RGB synthetic fake frames
        masks: (T, H, W) float32 blend boundary masks in [0.0, 1.0]
    """
    T, h, w, _ = clip_u8.shape
    p = _params(h)
    kernel = np.ones((3, 3), np.uint8)
    fakes, masks = [], []

    for t in range(T):
        tgt = clip_u8[t].astype(np.float32)
        m = _hull_mask(lmk106[t], h, w)
        if p["dilate"] > 0:
            m = cv2.dilate(m, kernel, iterations=p["dilate"])
        elif p["dilate"] < 0:
            m = cv2.erode(m, kernel, iterations=-p["dilate"])
        m = cv2.GaussianBlur(m, (p["ksize"], p["ksize"]), 0) * p["ratio"]
        src = _source(clip_u8[t], p)
        out = src * m[..., None] + tgt * (1.0 - m[..., None])
        fakes.append(np.clip(out, 0, 255).astype(np.uint8))
        masks.append(m)

    return np.stack(fakes), np.stack(masks)
