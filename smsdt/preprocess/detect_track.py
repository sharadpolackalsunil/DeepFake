"""
Face Detection + Tracking using SCRFD + ByteTrack.
Processes videos to extract per-frame face detections with tracking IDs.
"""
import cv2
import numpy as np
from tqdm import tqdm


def create_detector(ctx_id=0, det_size=(320, 320)):
    """Lazily create the SCRFD face detector (avoids import-time CUDA init)."""
    from insightface.app import FaceAnalysis
    detector = FaceAnalysis(name="scrfd_10g_bnkps", providers=["CUDAExecutionProvider"])
    detector.prepare(ctx_id=ctx_id, det_size=det_size)
    return detector


def create_tracker(track_thresh=0.5, match_thresh=0.8, frame_rate=30):
    """Create a ByteTrack tracker instance."""
    try:
        from bytetrack.byte_tracker import BYTETracker

        class TrackerArgs:
            """Minimal args object that BYTETracker expects."""
            def __init__(self, track_thresh, match_thresh, frame_rate):
                self.track_thresh = track_thresh
                self.track_buffer = 30
                self.match_thresh = match_thresh
                self.mot20 = False

        args = TrackerArgs(track_thresh, match_thresh, frame_rate)
        return BYTETracker(args, frame_rate=frame_rate)
    except ImportError:
        print("Warning: ByteTrack not available. Using simple detection-only mode.")
        return None


def detect_and_track(video_path, detector=None, tracker=None, det_size=(320, 320)):
    """Run face detection + tracking on a video.

    Args:
        video_path: path to input video file
        detector: SCRFD detector (created lazily if None)
        tracker: ByteTrack tracker (created lazily if None)
        det_size: detection input resolution

    Returns:
        list of dicts with keys: frame, track_id, bbox, kps, conf
    """
    if detector is None:
        detector = create_detector(det_size=det_size)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    records = []
    frame_idx = 0

    for _ in tqdm(range(total_frames), desc=f"Detecting faces", leave=False):
        ok, frame = cap.read()
        if not ok:
            break

        faces = detector.get(frame)

        if tracker is not None and len(faces) > 0:
            # ByteTrack expects numpy array of [x1, y1, x2, y2, score]
            det_array = np.array([
                [*f.bbox, f.det_score] for f in faces
            ], dtype=np.float32)  # (N, 5)

            img_info = frame.shape[:2]  # (H, W)
            img_size = frame.shape[:2]

            tracks = tracker.update(det_array, img_info, img_size)

            for t in tracks:
                # Match track back to closest detection for keypoints
                track_bbox = t.tlbr
                best_face_idx = 0
                if len(faces) > 1:
                    centers = np.array([(f.bbox[0]+f.bbox[2])/2 for f in faces])
                    track_center = (track_bbox[0] + track_bbox[2]) / 2
                    best_face_idx = np.argmin(np.abs(centers - track_center))

                kps = faces[best_face_idx].kps if hasattr(faces[best_face_idx], 'kps') else None

                records.append({
                    "frame": frame_idx,
                    "track_id": int(t.track_id),
                    "bbox": track_bbox.tolist(),
                    "kps": kps.tolist() if kps is not None else None,
                    "conf": float(t.score),
                })
        else:
            # No tracker: use detection index as pseudo-track
            for fi, f in enumerate(faces):
                kps = f.kps if hasattr(f, 'kps') else None
                records.append({
                    "frame": frame_idx,
                    "track_id": fi,
                    "bbox": f.bbox.tolist(),
                    "kps": kps.tolist() if kps is not None else None,
                    "conf": float(f.det_score),
                })

        frame_idx += 1

    cap.release()
    return records
