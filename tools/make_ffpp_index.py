"""Build data/splits/index.csv from the official FF++ pair splits.

Each official file is a list of [a, b] pairs. Real videos are 'a' and 'b'.
Fake videos are named 'a_b' (target a, source b) and 'b_a'. Both members of a
pair, and every video made from them, go to the same split.
"""
import csv
import json
import os
import argparse

METHODS = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures", "FaceShifter"]
SWAPS = {"Deepfakes", "FaceSwap", "FaceShifter"}       # source identity appears in the fake


def main(split_dir="data/splits/official", raw_root="data/raw/ffpp_c23", comp="c23",
         out="data/splits/index.csv"):
    rows = []
    for split in ("train", "val", "test"):
        split_file = os.path.join(split_dir, f"{split}.json")
        if not os.path.exists(split_file):
            raise FileNotFoundError(f"Missing split file: {split_file}")

        with open(split_file) as f:
            pairs = json.load(f)

        for a, b in pairs:
            # Real videos
            for vid in (a, b):
                rows.append(dict(
                    video_id=f"real_{vid}",
                    split=split,
                    label=0,
                    method="real",
                    target_id=vid,
                    source_id="",
                    path=os.path.normpath(f"{raw_root}/original_sequences/youtube/{comp}/videos/{vid}.mp4"),
                ))
            # Manipulated videos (target t, source s)
            for m in METHODS:
                for t, s in ((a, b), (b, a)):
                    rows.append(dict(
                        video_id=f"{m}_{t}_{s}",
                        split=split,
                        label=1,
                        method=m,
                        target_id=t,
                        source_id=s if m in SWAPS else "",
                        path=os.path.normpath(f"{raw_root}/manipulated_sequences/{m}/{comp}/videos/{t}_{s}.mp4"),
                    ))

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"Index successfully generated at: {out}")
    print(f"Total videos indexed: {len(rows)}")

    # Summary by split and method
    from collections import defaultdict
    counts = defaultdict(lambda: defaultdict(int))
    for r in rows:
        counts[r["split"]][r["method"]] += 1
    for s, m_counts in sorted(counts.items()):
        print(f"  Split [{s}]: {sum(m_counts.values())} total -> {dict(m_counts)}")

    missing = [r["path"] for r in rows if not os.path.exists(r["path"])]
    print(f"Disk check: {len(missing)} of {len(rows)} video paths currently missing on disk.")
    if missing:
        print(f"  (Example missing path: {missing[0]})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build FF++ index CSV from official splits")
    parser.add_argument("--split-dir", default="data/splits/official", help="Directory with official json splits")
    parser.add_argument("--raw-root", default="data/raw/ffpp_c23", help="Root folder of raw FF++ dataset")
    parser.add_argument("--comp", default="c23", help="Compression level (default: c23)")
    parser.add_argument("--out", default="data/splits/index.csv", help="Output CSV path")
    args = parser.parse_args()

    main(split_dir=args.split_dir, raw_root=args.raw_root, comp=args.comp, out=args.out)
