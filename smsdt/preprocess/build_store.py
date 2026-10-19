"""Per-video loose 1.3x frame store builder for S-MSDT with multi-worker parallelism.

Replaces chunk-based WebDataset serialization with a clean, memory-mapped
per-video frame store preserving 106 facial landmarks and paired alignment geometry.

Usage:
    python -m smsdt.preprocess.build_store --index data/splits/index.csv --raw-root . --out data/store --size 256 --margin 1.3 --workers 8 --device 0
"""
import os
import csv
import argparse
import multiprocessing as mp
import numpy as np
import cv2
from scipy.signal import savgol_filter
from tqdm import tqdm

from smsdt.preprocess.align_crop import loose_template, warp_points


def get_detector(ctx_id=0, det_size=(320, 320)):
    """Initialize SCRFD detector with 106-landmark module."""
    import sys
    import torch

    # Ensure CUDA DLLs are found on Windows
    if sys.platform == "win32":
        torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
        if os.path.isdir(torch_lib):
            try:
                os.add_dll_directory(torch_lib)
            except Exception:
                pass
            os.environ["PATH"] = torch_lib + os.pathsep + os.environ.get("PATH", "")

    from insightface.app import FaceAnalysis
    has_cuda = torch.cuda.is_available()
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if has_cuda else ["CPUExecutionProvider"]
    ctx = ctx_id if has_cuda else -1

    app = FaceAnalysis(name="buffalo_l", allowed_modules=['detection', 'landmark_2d_106'], providers=providers)
    app.prepare(ctx_id=ctx, det_size=det_size)
    return app


def compute_iou(box1, box2):
    """Compute IoU between two bboxes [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def extract_video_track(video_path, detector, max_track_frames=250):
    """Extract frames and track face across video frames.
    
    Returns:
        dict: {frame_idx: {'bgr': frame, 'kps5': kps, 'lmk106': lmk, 'box': bbox}}
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {}

    track = {}
    prev_box = None
    frame_idx = 0

    while frame_idx < max_track_frames:
        ok, frame = cap.read()
        if not ok:
            break

        faces = detector.get(frame)
        if faces:
            if prev_box is None:
                # Frame 0 or first detection: pick largest face
                best_face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            else:
                # Subsequent frames: pick face with highest IoU to previous box
                best_face = max(faces, key=lambda f: compute_iou(f.bbox, prev_box))

            prev_box = best_face.bbox
            kps5 = best_face.kps
            lmk106 = getattr(best_face, 'landmark_2d_106', None)
            if lmk106 is None:
                lmk106 = kps5

            track[frame_idx] = {
                'bgr': frame,
                'kps5': kps5,
                'lmk106': lmk106,
                'box': best_face.bbox,
            }

        frame_idx += 1

    cap.release()
    return track


def select_segments(tracked_indices, max_frames=128, segment_len=32):
    """Select up to max_frames organized in contiguous segments."""
    tracked_indices = sorted(tracked_indices)
    if len(tracked_indices) <= max_frames:
        return tracked_indices

    # Find all contiguous runs
    runs = []
    current_run = [tracked_indices[0]]
    for idx in tracked_indices[1:]:
        if idx == current_run[-1] + 1:
            current_run.append(idx)
        else:
            if len(current_run) >= 8:
                runs.append(current_run)
            current_run = [idx]
    if len(current_run) >= 8:
        runs.append(current_run)

    if not runs:
        step = len(tracked_indices) // max_frames
        return tracked_indices[::max(1, step)][:max_frames]

    # Select up to 4 segments spread evenly
    selected = []
    num_segs = min(4, len(runs))
    seg_step = max(1, len(runs) // num_segs)
    for s_idx in range(0, len(runs), seg_step):
        run = runs[s_idx]
        take = min(segment_len, len(run))
        selected.extend(run[:take])
        if len(selected) >= max_frames:
            break

    return sorted(selected[:max_frames])


def process_video_entry(row, raw_root, out_root, detector, dst_template,
                         target_transforms_cache=None, size=256, margin=1.3):
    """Process a single video entry from index.csv into data/store/<video_id>/."""
    video_id = row["video_id"]
    save_dir = os.path.join(out_root, video_id)
    if os.path.exists(os.path.join(save_dir, "frames.npy")):
        return True  # Already processed

    video_path = row["path"]
    if not os.path.isabs(video_path):
        video_path = os.path.join(raw_root, video_path)

    if not os.path.exists(video_path):
        return False

    track = extract_video_track(video_path, detector)
    if len(track) < 8:
        return False

    indices = sorted(track.keys())
    kps5_seq = np.stack([track[i]['kps5'] for i in indices], axis=0)  # (N, 5, 2)

    # Non-causal landmark smoothing via Savitzky-Golay filter
    window = min(7, len(indices))
    if window % 2 == 0:
        window -= 1
    if window >= 3:
        smoothed_kps5 = savgol_filter(kps5_seq, window_length=window, polyorder=2, axis=0)
    else:
        smoothed_kps5 = kps5_seq

    chosen_indices = select_segments(indices, max_frames=128, segment_len=32)

    aligned_frames = []
    warped_lmks = []
    final_fidxs = []
    transforms = {}

    is_real = (row["label"] == "0" or row["label"] == 0 or row["method"] == "real")
    target_id = row["target_id"]

    # Check disk cache for real video affine transforms if processing a fake
    cached_disk_transforms = None
    if not is_real:
        real_transforms_path = os.path.join(out_root, f"real_{target_id}", "transforms.npy")
        if os.path.exists(real_transforms_path):
            try:
                cached_disk_transforms = np.load(real_transforms_path, allow_pickle=True).item()
            except Exception:
                cached_disk_transforms = None

    for idx in chosen_indices:
        bgr = track[idx]['bgr']
        lmk106 = track[idx]['lmk106']
        kps_idx = indices.index(idx)

        # Affine matrix
        M = None
        if not is_real:
            if target_transforms_cache is not None and target_id in target_transforms_cache:
                M = target_transforms_cache[target_id].get(idx, None)
            elif cached_disk_transforms is not None:
                M = cached_disk_transforms.get(idx, None)

        if M is None:
            kps_smooth = smoothed_kps5[kps_idx]
            M, _ = cv2.estimateAffinePartial2D(kps_smooth.astype(np.float32), dst_template, method=cv2.LMEDS)

        if M is None:
            continue

        if is_real:
            transforms[idx] = M

        crop = cv2.warpAffine(bgr, M, (size, size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        aligned_frames.append(rgb)

        lmk_crop = warp_points(lmk106, M)
        warped_lmks.append(lmk_crop.astype(np.float32))
        final_fidxs.append(idx)

    if len(aligned_frames) < 8:
        return False

    os.makedirs(save_dir, exist_ok=True)
    if is_real:
        if target_transforms_cache is not None:
            target_transforms_cache[target_id] = transforms
        np.save(os.path.join(save_dir, "transforms.npy"), transforms)

    np.save(os.path.join(save_dir, "frames.npy"), np.stack(aligned_frames, axis=0).astype(np.uint8))
    np.save(os.path.join(save_dir, "lmk106.npy"), np.stack(warped_lmks, axis=0).astype(np.float32))
    np.save(os.path.join(save_dir, "fidx.npy"), np.array(final_fidxs, dtype=np.int32))
    return True


# Multi-processing worker globals
_w_detector = None
_w_template = None
_w_raw_root = None
_w_out_root = None
_w_size = None
_w_margin = None


def _init_worker(device_id, raw_root, out_root, size, margin):
    global _w_detector, _w_template, _w_raw_root, _w_out_root, _w_size, _w_margin
    _w_template = loose_template(out=size, margin=margin)
    _w_detector = get_detector(ctx_id=device_id)
    _w_raw_root = raw_root
    _w_out_root = out_root
    _w_size = size
    _w_margin = margin


def _worker_entry(row):
    global _w_detector, _w_template, _w_raw_root, _w_out_root, _w_size, _w_margin
    try:
        return process_video_entry(
            row, _w_raw_root, _w_out_root, _w_detector, _w_template,
            size=_w_size, margin=_w_margin
        )
    except Exception as e:
        return False


def main():
    parser = argparse.ArgumentParser(description="Build S-MSDT per-video frame store")
    parser.add_argument("--index", default="data/splits/index.csv", help="Path to index.csv")
    parser.add_argument("--raw-root", default=".", help="Root directory for relative video paths in index")
    parser.add_argument("--out", default="data/store", help="Output directory for frame store")
    parser.add_argument("--size", type=int, default=256, help="Output crop size")
    parser.add_argument("--margin", type=float, default=1.3, help="Loose crop margin")
    parser.add_argument("--device", type=int, default=0, help="CUDA device ID (-1 for CPU)")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel worker processes")
    args = parser.parse_args()

    with open(args.index, newline="") as f:
        rows = list(csv.DictReader(f))

    print(f"Loaded {len(rows)} entries from {args.index}")

    real_rows = [r for r in rows if r["method"] == "real"]
    fake_rows = [r for r in rows if r["method"] != "real"]
    print(f"Processing {len(real_rows)} pristine real videos followed by {len(fake_rows)} fake videos (workers={args.workers})...")

    success_count = 0

    if args.workers > 1:
        ctx = mp.get_context("spawn")
        with ctx.Pool(
            processes=args.workers,
            initializer=_init_worker,
            initargs=(args.device, args.raw_root, args.out, args.size, args.margin),
        ) as pool:
            for ok in tqdm(pool.imap_unordered(_worker_entry, real_rows), total=len(real_rows), desc=f"Storing Reals ({args.workers} workers)"):
                if ok:
                    success_count += 1
            for ok in tqdm(pool.imap_unordered(_worker_entry, fake_rows), total=len(fake_rows), desc=f"Storing Fakes ({args.workers} workers)"):
                if ok:
                    success_count += 1
    else:
        dst_template = loose_template(out=args.size, margin=args.margin)
        detector = get_detector(ctx_id=args.device)
        target_transforms = {}

        for r in tqdm(real_rows, desc="Storing Reals"):
            ok = process_video_entry(r, args.raw_root, args.out, detector, dst_template,
                                     target_transforms_cache=target_transforms,
                                     size=args.size, margin=args.margin)
            if ok:
                success_count += 1

        for r in tqdm(fake_rows, desc="Storing Fakes"):
            ok = process_video_entry(r, args.raw_root, args.out, detector, dst_template,
                                     target_transforms_cache=target_transforms,
                                     size=args.size, margin=args.margin)
            if ok:
                success_count += 1

    print(f"Successfully processed {success_count} / {len(rows)} videos into {args.out}")


if __name__ == "__main__":
    main()
