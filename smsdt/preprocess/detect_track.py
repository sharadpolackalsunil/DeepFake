import cv2
from insightface.app import FaceAnalysis
# ByteTrack reference implementation vendored under third_party/bytetrack
from third_party.bytetrack.byte_tracker import BYTETracker

detector = FaceAnalysis(name="scrfd_10g_bnkps", providers=["CUDAExecutionProvider"])
detector.prepare(ctx_id=0, det_size=(320, 320))
tracker = BYTETracker(track_thresh=0.5, match_thresh=0.8, frame_rate=30)

def detect_and_track(video_path, out_path):
    cap = cv2.VideoCapture(video_path)
    records = []
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        faces = detector.get(frame)
        dets = [(f.bbox, f.det_score, f.kps) for f in faces]
        tracks = tracker.update(dets, frame.shape[:2])
        for t in tracks:
            records.append({
                "frame": frame_idx, "track_id": t.track_id,
                "bbox": t.tlbr.tolist(), "kps": t.kps.tolist(),
                "conf": float(t.score),
            })
        frame_idx += 1
    cap.release()
    return records
