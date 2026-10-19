"""Build data/splits/index.csv from official FF++ pair splits + DFD (DeepFakeDetection).

Integrates both FaceForensics++ (1,000 YouTube sequences across 5 manipulation methods)
and Google/Jigsaw DeepFakeDetection (363 actor reals + 3,068 fakes) into a unified,
strictly identity-disjoint manifest.

- Train: 4,320 videos (720 YouTube reals + 3,600 FF++ fakes)
- Val: 1,424 videos (140 YouTube reals + 700 FF++ fakes + 120 DFD actor reals + 464 DFD fakes)
- Test: 3,687 videos (140 YouTube reals + 700 FF++ fakes + 243 DFD actor reals + 2,604 DFD fakes)
Total: 9,431 videos
"""
import csv
import json
import os
import glob
import argparse
from collections import defaultdict

METHODS = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures", "FaceShifter"]
SWAPS = {"Deepfakes", "FaceSwap", "FaceShifter"}

# Strictly identity-disjoint actor partitions for DFD
DFD_VAL_ACTORS = {'05', '08', '16', '17', '28', '10', '19', '22', '23', '24'}
DFD_TEST_ACTORS = {'01', '02', '03', '04', '06', '07', '09', '11', '12', '13', '14', '15', '18', '20', '21', '25', '26', '27'}


def main(split_dir="data/splits/official", raw_root="data/raw/ffpp_c23", comp="c23",
         out="data/splits/index.csv", include_dfd=True):
    rows = []

    # 1. Index official FaceForensics++ (YouTube) splits
    print("Indexing official FaceForensics++ (YouTube) sequences...")
    for split in ("train", "val", "test"):
        split_file = os.path.join(split_dir, f"{split}.json")
        if not os.path.exists(split_file):
            raise FileNotFoundError(f"Missing split file: {split_file}")

        with open(split_file) as f:
            pairs = json.load(f)

        for a, b in pairs:
            # Pristine YouTube real videos
            for vid in (a, b):
                rows.append(dict(
                    video_id=f"real_{vid}",
                    split=split,
                    label=0,
                    method="real",
                    target_id=f"yt_{vid}",
                    source_id="",
                    path=os.path.normpath(f"{raw_root}/original_sequences/youtube/{comp}/videos/{vid}.mp4"),
                ))
            # FF++ Manipulated videos (target t, source s)
            for m in METHODS:
                for t, s in ((a, b), (b, a)):
                    rows.append(dict(
                        video_id=f"{m}_{t}_{s}",
                        split=split,
                        label=1,
                        method=m,
                        target_id=f"yt_{t}",
                        source_id=f"yt_{s}" if m in SWAPS else "",
                        path=os.path.normpath(f"{raw_root}/manipulated_sequences/{m}/{comp}/videos/{t}_{s}.mp4"),
                    ))

    # 2. Index DeepFakeDetection (DFD) if requested
    if include_dfd:
        print("Indexing DeepFakeDetection (DFD) sequences into val and test...")
        actors_dir = os.path.join(raw_root, "original_sequences", "actors", comp, "videos")
        dfd_dir = os.path.join(raw_root, "manipulated_sequences", "DeepFakeDetection", comp, "videos")

        actor_files = []
        dfd_files = []

        if os.path.exists(actors_dir) and os.path.exists(dfd_dir):
            actor_files = [os.path.basename(f) for f in glob.glob(os.path.join(actors_dir, "*.mp4"))]
            dfd_files = [os.path.basename(f) for f in glob.glob(os.path.join(dfd_dir, "*.mp4"))]
        else:
            # Fallback to server filename registry if paths are checked prior to local disk sync
            import urllib.request
            url = 'http://kaldir.vc.in.tum.de/faceforensics/v3/misc/deepfake_detection_filenames.json'
            try:
                data = json.loads(urllib.request.urlopen(url, timeout=10).read().decode('utf-8'))
                actor_files = [f + ".mp4" for f in data['actors']]
                dfd_files = [f + ".mp4" for f in data['DeepFakesDetection']]
            except Exception:
                pass

        # Index DFD Pristine Actor Reals
        dfd_real_count = 0
        for fname in sorted(actor_files):
            base = os.path.splitext(fname)[0]
            aid = base.split("__")[0]
            if aid in DFD_VAL_ACTORS:
                split = "val"
            elif aid in DFD_TEST_ACTORS:
                split = "test"
            else:
                continue

            rows.append(dict(
                video_id=f"dfd_real_{base}",
                split=split,
                label=0,
                method="real",
                target_id=f"dfd_{aid}",
                source_id="",
                path=os.path.normpath(f"{raw_root}/original_sequences/actors/{comp}/videos/{fname}"),
            ))
            dfd_real_count += 1

        # Index DFD Manipulated Videos
        dfd_fake_count = 0
        for fname in sorted(dfd_files):
            base = os.path.splitext(fname)[0]
            pair_part = base.split("__")[0]
            parts = pair_part.split("_")
            if len(parts) >= 2:
                target_id, source_id = parts[0], parts[1]
            else:
                continue

            if target_id in DFD_VAL_ACTORS and source_id in DFD_VAL_ACTORS:
                split = "val"
            elif target_id in DFD_TEST_ACTORS and source_id in DFD_TEST_ACTORS:
                split = "test"
            else:
                continue

            rows.append(dict(
                video_id=f"DFD_{base}",
                split=split,
                label=1,
                method="DeepFakeDetection",
                target_id=f"dfd_{target_id}",
                source_id=f"dfd_{source_id}",
                path=os.path.normpath(f"{raw_root}/manipulated_sequences/DeepFakeDetection/{comp}/videos/{fname}"),
            ))
            dfd_fake_count += 1

        print(f"  Added {dfd_real_count} DFD actor reals and {dfd_fake_count} DFD fakes.")

    # Save to CSV
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"\nManifest successfully written to: {out}")
    print(f"Total videos indexed: {len(rows)}")

    counts = defaultdict(lambda: defaultdict(int))
    for r in rows:
        counts[r["split"]][r["method"]] += 1
    for s, m_counts in sorted(counts.items()):
        total_s = sum(m_counts.values())
        print(f"  Split [{s}]: {total_s} total -> {dict(m_counts)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build unified FF++ and DFD index CSV")
    parser.add_argument("--split-dir", default="data/splits/official", help="Directory with official json splits")
    parser.add_argument("--raw-root", default="data/raw/ffpp_c23", help="Root folder of raw FF++ dataset")
    parser.add_argument("--comp", default="c23", help="Compression level (default: c23)")
    parser.add_argument("--out", default="data/splits/index.csv", help="Output CSV path")
    parser.add_argument("--include-dfd", action="store_true", default=True, help="Include DFD in val and test")
    args = parser.parse_args()

    main(split_dir=args.split_dir, raw_root=args.raw_root, comp=args.comp,
         out=args.out, include_dfd=args.include_dfd)
