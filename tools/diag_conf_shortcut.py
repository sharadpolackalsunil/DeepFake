"""AUC of detector confidence alone as a fake score (D2a).

Reads conf.npy and label.cls from shards; no model needed.
"""
import glob
import sys
import numpy as np

try:
    import webdataset as wds
    from webdataset.gopen import gopen_schemes, gopen_file
    for _letter in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ":
        gopen_schemes[_letter] = gopen_file
except ImportError:
    wds = None

from sklearn.metrics import roc_auc_score


def main(pattern="data/cache/ffpp/test-*.tar"):
    if wds is None:
        print("[ERROR] webdataset is not installed. Run: pip install webdataset")
        sys.exit(1)

    matching_shards = sorted(glob.glob(pattern))
    if not matching_shards:
        print(f"[ERROR] No shards found matching pattern: {pattern}")
        sys.exit(1)

    print(f"Reading {len(matching_shards)} shards matching '{pattern}'...")
    scores, labels = [], []
    ds = wds.WebDataset(matching_shards, shardshuffle=False).decode()
    count = 0
    for s in ds:
        if "conf.npy" not in s or "label.cls" not in s:
            continue
        conf = np.asarray(s["conf.npy"], dtype=np.float32)
        lab = s["label.cls"]
        lab = float(lab.decode() if isinstance(lab, bytes) else lab)
        scores.append(-conf.mean())   # lower confidence -> "more fake"
        labels.append(lab)
        count += 1
        if count % 5000 == 0:
            print(f"  Processed {count} chunks...")

    if not labels or len(set(labels)) < 2:
        print("[ERROR] Shards did not yield at least two classes (real and fake).")
        return

    auc = roc_auc_score(labels, scores)
    print(f"\nTotal chunks evaluated: {len(labels)}")
    print(f"conf-only AUC = {auc:.4f}")
    if abs(auc - 0.5) > 0.10:
        print("[WARNING] |AUC - 0.5| > 0.10! Landmark confidence alone leaks the label!")
        print("Recommendation: Drop detector confidence gating from the temporal difference module.")
    else:
        print("[OK] Landmark confidence does NOT leak label shortcut (|AUC - 0.5| <= 0.10).")


if __name__ == "__main__":
    pattern_arg = sys.argv[1] if len(sys.argv) > 1 else "data/cache/ffpp/test-*.tar"
    main(pattern_arg)
