import cv2
import numpy as np

REF_5PT = np.array([
    [38.2946, 51.6963], [73.5318, 51.5014],
    [56.0252, 71.7366], [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)  # standard ArcFace-style template, scaled to 112 -> resized to 224

def align_face(frame, kps, out_size=224):
    dst = REF_5PT * (out_size / 112.0)
    M, _ = cv2.estimateAffinePartial2D(kps.astype(np.float32), dst, method=cv2.LMEDS)
    aligned = cv2.warpAffine(frame, M, (out_size, out_size), borderValue=0.0)
    return aligned
