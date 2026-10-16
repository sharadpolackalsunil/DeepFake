"""
Preprocessing Pipeline: Build WebDataset Shards
===============================================
Converts raw FF++ videos into aligned, chunked face crops stored as
webdataset .tar shards for efficient training I/O.

Pipeline: video → detect faces → align + crop → chunk T=8 frames → write .tar shards
"""
import os
import sys
import glob
import json
import argparse
import numpy as np
import cv2
try:
    import webdataset as wds
except ImportError:
    wds = None
from tqdm import tqdm

from concurrent.futures import ProcessPoolExecutor, as_completed

from smsdt.preprocess.align_crop import align_face


# FF++ dataset structure mapping
FFPP_DATASETS = {
    "original_sequences/youtube": "real",
    "original_sequences/actors": "real",
    "manipulated_sequences/Deepfakes": "Deepfakes",
    "manipulated_sequences/Face2Face": "Face2Face",
    "manipulated_sequences/FaceSwap": "FaceSwap",
    "manipulated_sequences/NeuralTextures": "NeuralTextures",
    "manipulated_sequences/FaceShifter": "FaceShifter",
    "manipulated_sequences/DeepFakeDetection": "DeepFakeDetection",
}


def extract_aligned_frames(video_path, detector, out_size=224, smooth_landmarks=True, smooth_alpha=0.6):
    """Extract aligned face crops from all frames of a video with temporal landmark smoothing.

    Args:
        video_path: path to .mp4 file
        detector: SCRFD detector instance
        out_size: output crop size
        smooth_landmarks: whether to temporally smooth facial landmarks to eliminate jitter
        smooth_alpha: weight for previous smoothed landmarks (EMA)

    Returns:
        frames: list of (out_size, out_size, 3) uint8 arrays
        confs: list of float confidence values
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return [], []

    frames = []
    confs = []
    prev_kps = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        faces = detector.get(frame)

        if len(faces) > 0:
            # Take the highest-confidence face
            best = max(faces, key=lambda f: f.det_score)
            kps = best.kps if hasattr(best, 'kps') and best.kps is not None else None

            if kps is not None and kps.shape == (5, 2):
                if smooth_landmarks:
                    if prev_kps is not None:
                        # Smooth landmarks across consecutive frames to prevent jitter artifacts
                        kps_to_use = smooth_alpha * prev_kps + (1.0 - smooth_alpha) * kps
                    else:
                        kps_to_use = kps.copy()
                    prev_kps = kps_to_use
                else:
                    kps_to_use = kps

                aligned = align_face(frame, kps_to_use, out_size=out_size)
                if aligned is not None:
                    frames.append(aligned)
                    confs.append(float(best.det_score))
                else:
                    x1, y1, x2, y2 = [int(v) for v in best.bbox]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                    crop = frame[y1:y2, x1:x2]
                    if crop.size > 0:
                        crop = cv2.resize(crop, (out_size, out_size))
                        frames.append(crop)
                        confs.append(float(best.det_score))
                    prev_kps = None

            else:
                # Fallback: crop from bbox
                x1, y1, x2, y2 = [int(v) for v in best.bbox]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                crop = frame[y1:y2, x1:x2]
                if crop.size > 0:
                    crop = cv2.resize(crop, (out_size, out_size))
                    frames.append(crop)
                    confs.append(float(best.det_score))
                prev_kps = None
        else:
            prev_kps = None
        # If no face detected, skip frame (sparse chunks are handled below)

    cap.release()
    return frames, confs


def make_chunks(frames, confs, chunk_length=8, stride=4):
    """Split aligned frames into overlapping chunks of fixed length.

    Args:
        frames: list of (H, W, 3) uint8 arrays
        confs: list of float confidence values
        chunk_length: number of frames per chunk (T)
        stride: step between chunk starts

    Returns:
        list of (chunk_frames, chunk_confs) tuples
        chunk_frames: (T, H, W, 3) uint8 array
        chunk_confs: (T,) float32 array
    """
    chunks = []
    n = len(frames)
    if n < chunk_length:
        return chunks

    for start in range(0, n - chunk_length + 1, stride):
        chunk_f = np.stack(frames[start:start + chunk_length], axis=0)  # (T, H, W, 3)
        chunk_c = np.array(confs[start:start + chunk_length], dtype=np.float32)  # (T,)
        chunks.append((chunk_f, chunk_c))

    return chunks


def process_single_video(args_tuple):
    """Process one video: detect → align → chunk. (Worker function for multiprocessing.)

    Returns list of chunk dicts ready for shard writing.
    """
    video_path, video_id, label, manip_type, out_size, chunk_length, stride = args_tuple

    # Each worker creates its own detector (can't pickle CUDA objects)
    from smsdt.preprocess.detect_track import create_detector
    detector = create_detector(ctx_id=0, det_size=(320, 320))

    frames, confs = extract_aligned_frames(video_path, detector, out_size)
    chunks = make_chunks(frames, confs, chunk_length, stride)

    results = []
    for i, (chunk_f, chunk_c) in enumerate(chunks):
        results.append({
            "video_id": video_id,
            "start_frame": i * stride,
            "frames": chunk_f,        # (T, H, W, 3) uint8
            "conf": chunk_c,           # (T,) float32
            "label": label,            # 0 or 1
            "manipulation_type": manip_type,
        })

    return results


def write_shard(chunks, shard_path):
    """Write a list of chunk dicts to a webdataset .tar shard."""
    if wds is None:
        raise ImportError("webdataset is required to write shards. Run 'pip install webdataset'")
    os.makedirs(os.path.dirname(shard_path), exist_ok=True)
    with wds.TarWriter(shard_path) as sink:
        for c in chunks:
            sink.write({
                "__key__": f"{c['video_id']}_{c['start_frame']:06d}",
                "frames.npy": c["frames"].astype(np.uint8),
                "conf.npy": c["conf"].astype(np.float32),
                "label.cls": str(c["label"]).encode("utf-8"),
                "manip.txt": c["manipulation_type"].encode("utf-8"),
            })


def discover_ffpp_videos(data_root, compression="c23"):
    """Discover FF++ videos from the standard directory structure.

    Expected structure (from download1.py):
        data_root/
            original_sequences/youtube/<compression>/videos/*.mp4
            manipulated_sequences/Deepfakes/<compression>/videos/*.mp4
            ...

    Returns:
        list of (video_path, video_id, label, manip_type)
    """
    videos = []

    for dataset_path, manip_type in FFPP_DATASETS.items():
        video_dir = os.path.join(data_root, dataset_path, compression, "videos")
        if not os.path.isdir(video_dir):
            continue

        label = 0 if manip_type == "real" else 1
        mp4_files = sorted(glob.glob(os.path.join(video_dir, "*.mp4")))

        for vpath in mp4_files:
            video_id = os.path.splitext(os.path.basename(vpath))[0]
            # Prefix with manip_type to avoid ID collisions across datasets
            full_id = f"{manip_type}_{video_id}"
            videos.append((vpath, full_id, label, manip_type))

    return videos


def build_split_manifests(videos, split_file=None):
    """Split videos into train/val/test without source identity leakage.

    If split_file is provided, use it (CSV or JSON with official FF++ splits:
    e.g. 720 train / 140 val / 140 test videos, source/target pairs kept together).
    Otherwise, groups by source identity and splits 80/10/10 so a real video
    and all fakes created from that identity stay strictly in the same split.

    Returns dict: {"train": [...], "val": [...], "test": [...]}
    """
    if split_file and os.path.exists(split_file):
        split_map = {}
        if split_file.endswith(".json"):
            with open(split_file, "r") as f:
                data = json.load(f)
            # Support {"train": [...], "val": [...], "test": [...]}
            if isinstance(data, dict):
                for split_name, id_list in data.items():
                    for item in id_list:
                        if isinstance(item, list):
                            for sub_id in item:
                                split_map[str(sub_id)] = split_name
                            split_map["_".join(map(str, item))] = split_name
                        else:
                            split_map[str(item)] = split_name
        else:
            import csv
            with open(split_file, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    split_map[str(row["video_id"])] = row["split"]

        splits = {"train": [], "val": [], "test": []}
        for v in videos:
            vpath, full_id, _, _ = v
            raw_id = os.path.splitext(os.path.basename(vpath))[0]
            # Match raw_id, source id, or full_id
            if raw_id in split_map:
                s = split_map[raw_id]
            elif "_" in raw_id and raw_id.split("_")[0] in split_map:
                s = split_map[raw_id.split("_")[0]]
            elif full_id in split_map:
                s = split_map[full_id]
            else:
                s = "train"
            splits.setdefault(s, []).append(v)
        return splits

    # Default leak-free split: Group videos by source actor/video identity (80/10/10)
    identity_groups = {}
    for v in videos:
        vpath, full_id, label, manip_type = v
        raw_id = os.path.splitext(os.path.basename(vpath))[0]
        # For pairs like 042_056, base identity is the source actor (042)
        base_id = raw_id.split("_")[0]
        identity_groups.setdefault(base_id, []).append(v)

    unique_identities = sorted(list(identity_groups.keys()))
    np.random.seed(42)
    perm_ids = np.random.permutation(unique_identities)
    n = len(perm_ids)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)

    train_ids = set(perm_ids[:n_train])
    val_ids = set(perm_ids[n_train:n_train + n_val])

    splits = {"train": [], "val": [], "test": []}
    for base_id, v_list in identity_groups.items():
        if base_id in train_ids:
            splits["train"].extend(v_list)
        elif base_id in val_ids:
            splits["val"].extend(v_list)
        else:
            splits["test"].extend(v_list)

    return splits


def main():
    parser = argparse.ArgumentParser(description="Build webdataset shards from FF++ videos")
    parser.add_argument("--data-root", type=str, required=True,
                        help="Root dir of raw FF++ data (e.g. data/raw/ffpp_c23)")
    parser.add_argument("--compression", type=str, default="c23",
                        choices=["raw", "c23", "c40"],
                        help="FF++ compression level")
    parser.add_argument("--out", type=str, default="data/cache/ffpp",
                        help="Output directory for shards")
    parser.add_argument("--chunk-length", type=int, default=8,
                        help="Frames per chunk (T)")
    parser.add_argument("--chunk-stride", type=int, default=4,
                        help="Stride between chunks (use T for non-overlapping)")
    parser.add_argument("--img-size", type=int, default=224,
                        help="Aligned crop size")
    parser.add_argument("--shard-size", type=int, default=500,
                        help="Max chunks per shard .tar file")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel workers (set >1 for multi-GPU preprocessing)")
    parser.add_argument("--split-file", type=str, default=None,
                        help="CSV file with video_id,split columns")
    args = parser.parse_args()

    print(f"Discovering FF++ videos in {args.data_root} (compression={args.compression})...")
    videos = discover_ffpp_videos(args.data_root, args.compression)
    print(f"Found {len(videos)} videos")

    if not videos:
        print("ERROR: No videos found. Check --data-root and --compression.")
        print(f"Expected structure: {args.data_root}/original_sequences/youtube/{args.compression}/videos/*.mp4")
        sys.exit(1)

    # Print summary
    from collections import Counter
    type_counts = Counter(v[3] for v in videos)
    print("Video counts by type:")
    for manip_type, count in sorted(type_counts.items()):
        print(f"  {manip_type}: {count}")

    # Split into train/val/test
    splits = build_split_manifests(videos, args.split_file)
    for split_name, split_videos in splits.items():
        print(f"  {split_name}: {len(split_videos)} videos")

    # Ensure SCRFD detection model is downloaded once on main thread before worker spawning
    from smsdt.preprocess.detect_track import create_detector
    _ = create_detector(ctx_id=0, det_size=(320, 320))

    # Process each split
    for split_name, split_videos in splits.items():
        print(f"\n{'='*60}")
        print(f"Processing {split_name} split ({len(split_videos)} videos)...")
        print(f"{'='*60}")

        all_chunks = []

        if args.workers <= 1:
            # Sequential processing (simpler, works with single GPU)
            from smsdt.preprocess.detect_track import create_detector
            detector = create_detector(ctx_id=0, det_size=(320, 320))

            for vpath, video_id, label, manip_type in tqdm(split_videos, desc=split_name):
                frames, confs = extract_aligned_frames(vpath, detector, args.img_size)
                chunks = make_chunks(frames, confs, args.chunk_length, args.chunk_stride)

                for i, (chunk_f, chunk_c) in enumerate(chunks):
                    all_chunks.append({
                        "video_id": video_id,
                        "start_frame": i * args.chunk_stride,
                        "frames": chunk_f,
                        "conf": chunk_c,
                        "label": label,
                        "manipulation_type": manip_type,
                    })
        else:
            # Parallel processing (CPU-bound parts)
            tasks = [
                (vpath, video_id, label, manip_type,
                 args.img_size, args.chunk_length, args.chunk_stride)
                for vpath, video_id, label, manip_type in split_videos
            ]
            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                futures = {executor.submit(process_single_video, t): t for t in tasks}
                for future in tqdm(as_completed(futures), total=len(futures), desc=split_name):
                    try:
                        results = future.result()
                        all_chunks.extend(results)
                    except Exception as e:
                        task = futures[future]
                        print(f"Error processing {task[0]}: {e}")

        print(f"  Total chunks: {len(all_chunks)}")

        # Shuffle training data
        if split_name == "train":
            np.random.seed(42)
            np.random.shuffle(all_chunks)

        # Write to shards
        shard_idx = 0
        for start in range(0, len(all_chunks), args.shard_size):
            shard_chunks = all_chunks[start:start + args.shard_size]
            shard_path = os.path.join(args.out, f"{split_name}-{shard_idx:06d}.tar")
            write_shard(shard_chunks, shard_path)
            shard_idx += 1

        print(f"  Written {shard_idx} shards to {args.out}")

    print(f"\nPreprocessing complete!")


if __name__ == "__main__":
    main()
