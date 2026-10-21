"""Build data/splits/celebdf_test.csv from the official Celeb-DF v2 test manifest.

Parses List_of_testing_videos.txt dynamically from the dataset directory,
verifies physical presence of each test video, assigns appropriate labels
(0 = Real for Celeb-real/YouTube-real, 1 = Fake for Celeb-synthesis),
and generates a clean, reproducible index CSV for S-MSDT evaluation.

Usage:
    python tools/build_celebdf_test_index.py --list-file data/raw/Celeb-DF-v2/List_of_testing_videos.txt --dataset-root data/raw/Celeb-DF-v2 --out data/splits/celebdf_test.csv
"""
import os
import csv
import argparse


def parse_celebdf_manifest(list_file: str, dataset_root: str):
    """Parse official List_of_testing_videos.txt and return index rows."""
    if not os.path.exists(list_file):
        raise FileNotFoundError(f"Manifest file not found: {list_file}")

    rows = []
    missing_files = []

    with open(list_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 2:
                print(f"Warning: line {line_num} has unexpected format: {line}")
                continue

            # Original Celeb-DF annotation: 1 = real, 0 = fake
            # S-MSDT / Standard protocol: 0 = real, 1 = fake
            raw_label = parts[0]
            rel_video = parts[1].replace("/", os.sep)
            video_full_path = os.path.join(dataset_root, rel_video)

            if not os.path.exists(video_full_path):
                missing_files.append(video_full_path)
                continue

            # Robust label mapping based on folder taxonomy
            if "synthesis" in rel_video.lower():
                label = 1
                method = "Celeb-synthesis"
            else:
                label = 0
                method = "real"

            # Clean unique video ID preserving directory and filename
            vid_id = "celebdf_" + rel_video.replace(os.sep, "_").replace(".mp4", "")

            rows.append({
                "video_id": vid_id,
                "split": "test",
                "label": label,
                "method": method,
                "target_id": "",
                "source_id": "",
                "path": os.path.normpath(video_full_path),
            })

    return rows, missing_files


def main():
    parser = argparse.ArgumentParser(description="Generate Celeb-DF v2 test index manifest")
    parser.add_argument(
        "--list-file",
        default="data/raw/Celeb-DF-v2/List_of_testing_videos.txt",
        help="Path to official List_of_testing_videos.txt",
    )
    parser.add_argument(
        "--dataset-root",
        default="data/raw/Celeb-DF-v2",
        help="Root directory where Celeb-real, Celeb-synthesis, YouTube-real reside",
    )
    parser.add_argument(
        "--out",
        default="data/splits/celebdf_test.csv",
        help="Output CSV path",
    )
    args = parser.parse_args()

    print(f"Reading manifest from: {args.list_file}")
    print(f"Dataset root: {args.dataset_root}")

    rows, missing = parse_celebdf_manifest(args.list_file, args.dataset_root)

    if missing:
        print(f"Warning: {len(missing)} videos in manifest could not be located on disk:")
        for m in missing[:5]:
            print(f"  * Missing: {m}")
        if len(missing) > 5:
            print(f"  ... and {len(missing) - 5} more.")

    reals = sum(1 for r in rows if r["label"] == 0)
    fakes = sum(1 for r in rows if r["label"] == 1)

    print(f"Successfully validated {len(rows)} test videos:")
    print(f"  * Real videos: {reals} (Celeb-real + YouTube-real)")
    print(f"  * Fake videos: {fakes} (Celeb-synthesis)")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fields = ["video_id", "split", "label", "method", "target_id", "source_id", "path"]

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved manifest to: {args.out}")


if __name__ == "__main__":
    main()
