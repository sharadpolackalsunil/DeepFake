"""Prepare FakeAVCeleb_v1.2 manifest and identity CSV without touching any model scores.

Rules implemented (docs/decisions.md Section 6.4):
1. Reals: Category A (RealVideo-RealAudio, method=='real') ONLY -> 500 videos / 500 identities.
2. Excludes Category B (RealVideo-FakeAudio, method=='rtvc'): visual frames are real.
3. Fakes: Categories C & D (FakeVideo-RealAudio, FakeVideo-FakeAudio) filtered to .mp4 files
   that exist on disk (meta_data.csv lists 22 missing files in FakeVideo-FakeAudio).
4. Deterministic stratified fake subsample across the 5 visual methods (default 500 per method
   = 2,500 fakes with seed=0; pass --fakes-per-method 0 to keep all existing visual fakes).

Usage on the machine holding raw FakeAVCeleb_v1.2:
    python -m tools.prep_fakeavceleb --raw-dir E:/DeepFake/data/raw/FakeAVCeleb_v1.2
"""
import argparse
import hashlib
import os
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

VISUAL_FAKE_METHODS = ("wav2lip", "fsgan", "fsgan-wav2lip", "faceswap-wav2lip", "faceswap")


def resolve_video_relpath(raw_dir: Path, row: pd.Series) -> Tuple[Path, str]:
    """Resolve disk path and repo-relative index path from a FakeAVCeleb meta_data.csv row."""
    folder = str(row.get("Unnamed: 9", "")).strip().replace("\\", "/")
    fname = str(row.get("path", "")).strip().replace("\\", "/")
    for prefix in ("FakeAVCeleb_v1.2/", "FakeAVCeleb/"):
        if folder.startswith(prefix):
            folder = folder[len(prefix):]
            break
    rel_under_raw = f"{folder.strip('/')}/{fname}" if folder else fname
    full = raw_dir / rel_under_raw
    return full, rel_under_raw


def prepare_fakeavceleb(
    raw_dir: Path,
    out_index: Path,
    out_identities: Path,
    fakes_per_method: int = 500,
    seed: int = 0,
    index_raw_prefix: str = "data/raw/FakeAVCeleb_v1.2",
) -> dict:
    meta_path = raw_dir / "meta_data.csv"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing meta_data.csv in {raw_dir}")
    df = pd.read_csv(meta_path)

    resolved_paths = []
    exists_mask = []
    for _, row in df.iterrows():
        full, rel_under_raw = resolve_video_relpath(raw_dir, row)
        exists_mask.append(full.exists())
        resolved_paths.append(f"{index_raw_prefix.rstrip('/')}/{rel_under_raw}")
    df = df.copy()
    df["_rel_path"] = resolved_paths
    df["_exists"] = exists_mask

    # Category A: RealVideo-RealAudio
    reals_all = df[df["type"].astype(str) == "RealVideo-RealAudio"]
    reals = reals_all[reals_all["_exists"]].copy()
    if reals["source"].duplicated().any():
        reals = reals.drop_duplicates(subset=["source"], keep="first").copy()

    # Categories C & D: visual fakes only (exclude Category B RealVideo-FakeAudio)
    vfakes_all = df[df["type"].astype(str).isin(["FakeVideo-RealAudio", "FakeVideo-FakeAudio"])]
    n_missing_fakes = int((~vfakes_all["_exists"]).sum())
    vfakes = vfakes_all[vfakes_all["_exists"]].copy()

    rng = np.random.default_rng(seed)
    sampled_parts = []
    if fakes_per_method > 0:
        for method in VISUAL_FAKE_METHODS:
            sub = vfakes[vfakes["method"].astype(str) == method].sort_values(["category", "_rel_path"])
            if len(sub) <= fakes_per_method:
                sampled_parts.append(sub)
            else:
                # Stratify proportionally across categories C and D within each method
                parts = []
                cats = sorted(sub["category"].astype(str).unique())
                remaining = fakes_per_method
                for idx_c, cat in enumerate(cats):
                    sub_c = sub[sub["category"].astype(str) == cat]
                    if idx_c == len(cats) - 1:
                        take = remaining
                    else:
                        take = min(len(sub_c), int(round(fakes_per_method * len(sub_c) / len(sub))))
                    take = min(take, len(sub_c))
                    chosen = rng.choice(len(sub_c), size=take, replace=False)
                    parts.append(sub_c.iloc[np.sort(chosen)])
                    remaining -= take
                sampled_parts.append(pd.concat(parts, ignore_index=False))
        fakes_sel = pd.concat(sampled_parts, ignore_index=True)
    else:
        fakes_sel = vfakes.sort_values(["method", "category", "_rel_path"]).reset_index(drop=True)

    index_rows = []
    id_rows = []

    for _, r in reals.sort_values(["source", "_rel_path"]).iterrows():
        src = str(r["source"]).strip()
        stem = Path(str(r["path"]).strip()).stem
        vid = f"favc_real_{src}_{stem}"
        index_rows.append({
            "video_id": vid,
            "split": "test",
            "label": 0,
            "method": "real",
            "target_id": src,
            "source_id": "",
            "path": r["_rel_path"],
        })
        id_rows.append({"video_id": vid, "identity": src})

    for idx, (_, r) in enumerate(fakes_sel.iterrows()):
        cat = str(r["category"]).strip()
        meth = str(r["method"]).strip()
        src = str(r["source"]).strip()
        tgt = str(r["target1"]).strip()
        tgt_clean = src if tgt in ("", "-", "nan", "None") else tgt
        stem = Path(str(r["path"]).strip()).stem
        vid = f"favc_fake_{cat}_{meth.replace('-', '_')}_{idx:05d}_{stem}"
        index_rows.append({
            "video_id": vid,
            "split": "test",
            "label": 1,
            "method": meth,
            "target_id": tgt_clean,
            "source_id": src,
            "path": r["_rel_path"],
        })
        id_rows.append({"video_id": vid, "identity": f"fake_{src}"})

    idx_df = pd.DataFrame(index_rows)
    id_df = pd.DataFrame(id_rows)

    out_index.parent.mkdir(parents=True, exist_ok=True)
    out_identities.parent.mkdir(parents=True, exist_ok=True)
    idx_df.to_csv(out_index, index=False)
    id_df.to_csv(out_identities, index=False)

    sha_index = hashlib.sha256(out_index.read_bytes()).hexdigest()
    sha_ids = hashlib.sha256(out_identities.read_bytes()).hexdigest()
    return {
        "n_reals": int((idx_df["label"] == 0).sum()),
        "n_real_identities": int(id_df[id_df["video_id"].str.startswith("favc_real_")]["identity"].nunique()),
        "n_fakes": int((idx_df["label"] == 1).sum()),
        "n_missing_fakes_skipped": n_missing_fakes,
        "fake_method_counts": idx_df[idx_df["label"] == 1]["method"].value_counts().to_dict(),
        "sha256_index": sha_index,
        "sha256_identities": sha_ids,
    }


def prune_from_store(
    store_dir: Path,
    out_index: Path,
    out_identities: Path,
    min_contiguous_frames: int = 8,
) -> dict:
    """Filter out videos from index and identities where build_store failed or has < min_contiguous_frames."""
    if not out_index.exists():
        raise FileNotFoundError(f"Missing index file: {out_index}")
    if not store_dir.exists() or not any(store_dir.iterdir()):
        raise RuntimeError(
            f"Store directory '{store_dir}' does not exist or is empty! "
            f"build_store must successfully run and populate '{store_dir}' before pruning. "
            f"Refusing to wipe out {out_index}!"
        )
    idx_df = pd.read_csv(out_index)
    init_total = len(idx_df)
    if init_total == 0:
        raise ValueError(f"Index file {out_index} has 0 rows. Cannot prune an empty index.")

    valid_mask = []
    for vid in idx_df["video_id"]:
        fidx_path = store_dir / str(vid) / "fidx.npy"
        if not fidx_path.exists():
            valid_mask.append(False)
            continue
        try:
            fidx = np.load(fidx_path)
            span = min_contiguous_frames - 1
            has_contiguous = len(fidx) >= min_contiguous_frames and any(
                fidx[s + span] - fidx[s] == span for s in range(len(fidx) - span)
            )
            valid_mask.append(bool(has_contiguous))
        except Exception:
            valid_mask.append(False)

    if not any(valid_mask):
        raise RuntimeError(
            f"Pruning would drop ALL {init_total} videos! None had >= {min_contiguous_frames} contiguous frames "
            f"in '{store_dir}'. Check build_store output before pruning."
        )

    pruned_idx = idx_df[valid_mask].copy()
    pruned_idx.to_csv(out_index, index=False)

    sha_ids = ""
    n_real_ids = 0
    if out_identities.exists():
        id_df = pd.read_csv(out_identities)
        id_df = id_df[id_df["video_id"].isin(pruned_idx["video_id"])].copy()
        id_df.to_csv(out_identities, index=False)
        sha_ids = hashlib.sha256(out_identities.read_bytes()).hexdigest()
        n_real_ids = int(id_df[id_df["video_id"].str.startswith("favc_real_")]["identity"].nunique())

    sha_index = hashlib.sha256(out_index.read_bytes()).hexdigest()
    return {
        "initial_total": init_total,
        "pruned_total": len(pruned_idx),
        "dropped_count": init_total - len(pruned_idx),
        "n_reals": int((pruned_idx["label"] == 0).sum()),
        "n_real_identities": n_real_ids,
        "n_fakes": int((pruned_idx["label"] == 1).sum()),
        "fake_method_counts": pruned_idx[pruned_idx["label"] == 1]["method"].value_counts().to_dict(),
        "sha256_index": sha_index,
        "sha256_identities": sha_ids,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--raw-dir", default="data/raw/FakeAVCeleb_v1.2")
    ap.add_argument("--out-index", default="data/splits/fakeavceleb_index.csv")
    ap.add_argument("--out-identities", default="data/splits/fakeavceleb_identities.csv")
    ap.add_argument("--index-raw-prefix", default="data/raw/FakeAVCeleb_v1.2")
    ap.add_argument("--fakes-per-method", type=int, default=500,
                    help="Stratified fakes per visual method (5 methods -> 2,500 fakes). 0 = keep all.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prune-from-store", default=None,
                    help="Path to frame store (e.g. data/store_fakeavceleb) to prune unindexed/failed videos.")
    args = ap.parse_args()

    if args.prune_from_store:
        stats = prune_from_store(
            store_dir=Path(args.prune_from_store),
            out_index=Path(args.out_index),
            out_identities=Path(args.out_identities),
        )
        print(f"[Store Pruning Complete]")
        print(f"Total videos: {stats['initial_total']} -> {stats['pruned_total']} (dropped {stats['dropped_count']})")
        print(f"Reals: {stats['n_reals']} ({stats['n_real_identities']} unique identities)")
        print(f"Visual fakes: {stats['n_fakes']}")
        print(f"Fake methods: {stats['fake_method_counts']}")
        print(f"Updated index sha256: {stats['sha256_index']}")
        print(f"Updated identities sha256: {stats['sha256_identities']}")
        return

    stats = prepare_fakeavceleb(
        raw_dir=Path(args.raw_dir),
        out_index=Path(args.out_index),
        out_identities=Path(args.out_identities),
        fakes_per_method=args.fakes_per_method,
        seed=args.seed,
        index_raw_prefix=args.index_raw_prefix,
    )
    print(f"Wrote {args.out_index} (sha256={stats['sha256_index']})")
    print(f"Wrote {args.out_identities} (sha256={stats['sha256_identities']})")
    print(f"Reals: {stats['n_reals']} ({stats['n_real_identities']} unique identities)")
    print(f"Visual fakes: {stats['n_fakes']} (missing on disk skipped: {stats['n_missing_fakes_skipped']})")
    print(f"Fake methods: {stats['fake_method_counts']}")


if __name__ == "__main__":
    main()

