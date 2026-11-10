"""Statistical testing and inference engine for S-MSDT TMLR study.

Implements:
1. Fast ROC AUC computation
2. Paired cluster bootstrap for multi-seed ΔAUC with Bonferroni-adjusted CIs
3. Equivalence testing with pre-registered margin δ
4. Fast DeLong variance and covariance for paired ROC AUC
"""
import math
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional


def fast_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Compute Area Under the ROC Curve via rank sum (Mann-Whitney U).
    
    Args:
        labels: 1D array of binary labels {0, 1}
        scores: 1D array of continuous prediction scores
        
    Returns:
        float: ROC AUC in [0.0, 1.0]
    """
    labels = np.asarray(labels, dtype=np.int32)
    scores = np.asarray(scores, dtype=np.float64)
    
    n_pos = int(np.sum(labels == 1))
    n_neg = int(np.sum(labels == 0))
    if n_pos == 0 or n_neg == 0:
        return float('nan')
        
    order = np.lexsort((np.random.RandomState(42).uniform(size=len(scores)), scores))
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    
    # Handle ties with average ranks
    unique_scores, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    if len(unique_scores) < len(scores):
        tie_ranks = np.bincount(inverse, weights=ranks) / counts
        ranks = tie_ranks[inverse]
        
    u_stat = np.sum(ranks[labels == 1]) - (n_pos * (n_pos + 1)) / 2.0
    return float(u_stat / (n_pos * n_neg))


def delong_roc_variance(labels: np.ndarray, scores: np.ndarray) -> Tuple[float, float]:
    """DeLong variance estimation for single ROC curve.
    
    Returns:
        (auc, variance)
    """
    labels = np.asarray(labels, dtype=np.int32)
    scores = np.asarray(scores, dtype=np.float64)
    
    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]
    m, n = len(pos_idx), len(neg_idx)
    if m == 0 or n == 0:
        return float('nan'), float('nan')
        
    pos_scores = scores[pos_idx]
    neg_scores = scores[neg_idx]
    
    # Structural components
    v10 = np.zeros(m, dtype=np.float64)
    v01 = np.zeros(n, dtype=np.float64)
    
    for i in range(m):
        v10[i] = np.mean((pos_scores[i] > neg_scores) + 0.5 * (pos_scores[i] == neg_scores))
    for j in range(n):
        v01[j] = np.mean((pos_scores > neg_scores[j]) + 0.5 * (pos_scores == neg_scores[j]))
        
    auc = float(np.mean(v10))
    s10 = np.var(v10, ddof=1) if m > 1 else 0.0
    s01 = np.var(v01, ddof=1) if n > 1 else 0.0
    
    var_auc = (s10 / m) + (s01 / n)
    return auc, float(var_auc)


def paired_cluster_boot_dauc(
    dfs_a: List[pd.DataFrame],
    dfs_b: List[pd.DataFrame],
    cluster_col: str = "cluster_id",
    label_col: str = "label",
    score_col: str = "score_mean",
    n_boot: int = 10000,
    level: float = 0.95,
    delta: float = 0.01,
    random_seed: int = 42
) -> Dict:
    """Paired cluster bootstrap across multiple random seeds for models A and B.
    
    Protocol:
    1. Resamples clusters with replacement.
    2. Gathers all records belonging to selected clusters.
    3. Evaluates seed-averaged AUC for model A and model B on the identical resample.
    4. Computes Delta = AUC(A) - AUC(B).
    5. Derives empirical percentile CI at confidence level `level`.
    6. Classifies against equivalence margin delta.
    
    Args:
        dfs_a: List of DataFrames for model A (one per seed, e.g. 5 seeds)
        dfs_b: List of DataFrames for model B (one per seed, matching seeds)
        cluster_col: Column name identifying identity/video cluster
        label_col: Ground truth label column
        score_col: Predicted score column
        n_boot: Number of bootstrap resamples (default 10,000)
        level: Confidence level, e.g. 0.95 or Bonferroni-adjusted (1 - 0.05/m)
        delta: Equivalence boundary margin (pre-registered δ = 0.01)
        random_seed: RNG seed for deterministic reproduction
        
    Returns:
        dict: {
            'mean_a': float,
            'mean_b': float,
            'mean_delta': float,
            'ci_lower': float,
            'ci_upper': float,
            'level': float,
            'delta': float,
            'classification': 'better' | 'equivalent' | 'worse' | 'inconclusive',
            'p_value_equiv': float,
            'n_clusters': int,
            'n_samples': int
        }
    """
    assert len(dfs_a) == len(dfs_b), "Must provide identical number of seed runs for A and B"
    n_seeds = len(dfs_a)

    # Align every run on the same set of videos. Sorting alone is not enough: if one
    # run is missing a video, row i of A and row i of B would silently refer to
    # different videos and the paired bootstrap would be meaningless.
    n_dropped = 0
    if "video_id" in dfs_a[0].columns:
        dfs_a, dfs_b, n_dropped = align_runs_on_video_id(dfs_a, dfs_b, label_col=label_col)

    ref_df = dfs_a[0]
    clusters = ref_df[cluster_col].unique()
    n_clusters = len(clusters)
    n_samples = len(ref_df)
    
    # Pre-map cluster_id -> row indices for lightning-fast lookup
    cluster_indices = {}
    for idx, c in enumerate(ref_df[cluster_col].values):
        cluster_indices.setdefault(c, []).append(idx)
    cluster_indices = {k: np.array(v, dtype=np.int64) for k, v in cluster_indices.items()}
    
    # Extract labels and scores into pure numpy matrices: (n_seeds, n_samples)
    labels = ref_df[label_col].values.astype(np.int32)
    scores_a = np.stack([df[score_col].values for df in dfs_a], axis=0)  # (K, N)
    scores_b = np.stack([df[score_col].values for df in dfs_b], axis=0)  # (K, N)
    
    # Point estimates on full sample
    aucs_a = [fast_auc(labels, scores_a[k]) for k in range(n_seeds)]
    aucs_b = [fast_auc(labels, scores_b[k]) for k in range(n_seeds)]
    mean_a = float(np.mean(aucs_a))
    mean_b = float(np.mean(aucs_b))
    point_delta = mean_a - mean_b
    
    # Bootstrap loop
    rng = np.random.RandomState(random_seed)
    deltas = np.zeros(n_boot, dtype=np.float64)
    
    for b in range(n_boot):
        # Resample clusters with replacement. A resample containing a single class has
        # no AUC; redraw it. (The previous `continue` left deltas[b] = 0, which pulled
        # the interval toward zero.)
        for _attempt in range(1000):
            boot_clusters = rng.choice(clusters, size=n_clusters, replace=True)
            boot_idx = np.concatenate([cluster_indices[c] for c in boot_clusters])
            boot_y = labels[boot_idx]
            if not (np.all(boot_y == 0) or np.all(boot_y == 1)):
                break
        else:
            raise RuntimeError("Could not draw a two-class bootstrap resample in 1000 attempts.")

        boot_auc_a = np.mean([fast_auc(boot_y, scores_a[k][boot_idx]) for k in range(n_seeds)])
        boot_auc_b = np.mean([fast_auc(boot_y, scores_b[k][boot_idx]) for k in range(n_seeds)])
        deltas[b] = boot_auc_a - boot_auc_b
        
    alpha = (1.0 - level) / 2.0
    ci_lower = float(np.percentile(deltas, alpha * 100))
    ci_upper = float(np.percentile(deltas, (1.0 - alpha) * 100))
    
    # Pre-registered equivalence classification rule:
    # Better: L > delta
    # Equivalent: L >= -delta AND U <= +delta
    # Worse: U < -delta
    # Inconclusive: Otherwise
    if ci_lower > delta:
        classification = "better"
    elif ci_lower >= -delta and ci_upper <= delta:
        classification = "equivalent"
    elif ci_upper < -delta:
        classification = "worse"
    else:
        classification = "inconclusive"
        
    return {
        "mean_a": mean_a,
        "mean_b": mean_b,
        "mean_delta": point_delta,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "level": level,
        "delta": delta,
        "classification": classification,
        "n_clusters": int(n_clusters),
        "n_samples": int(n_samples),
        "n_dropped_unaligned": int(n_dropped),
        "boot_sd": float(np.std(deltas)),
        "ci_half_width": float((ci_upper - ci_lower) / 2.0),
    }


# ---------------------------------------------------------------------------
# Alignment, pure-source negatives, dataset roles and pre-registered verdicts
# ---------------------------------------------------------------------------

# Which real videos count as negatives for each dataset (docs/score_schema.md,
# "Pure Negatives"). None means "all reals in the file".
PURE_NEGATIVE_SOURCES = {
    "ffpp": ["ffpp_youtube"],
    "ffpp_c23": ["ffpp_youtube"],
    "dfd": ["dfd_actor"],
    "celebdf": None,
    "dfdc": None,
    "wdf": None,
    "ffiw": None,
}

# Dataset roles. Only "confirmatory" datasets decide H1 (pre-registration: "on
# unseen datasets"). FF++ is the training distribution. DFD reals and fakes are in
# the validation (dev) set used for calibration, so DFD is not unseen either.
DEFAULT_DATASET_ROLES = {
    "ffpp": "in_domain",
    "ffpp_c23": "in_domain",
    "dfd": "dev_overlap",
    "celebdf": "confirmatory",
    "dfdc": "confirmatory",
    "wdf": "confirmatory",
    "ffiw": "confirmatory",
}

# Comparisons. H1 is pre-registered against S ("Neither ST nor Full improves AUC
# over S"), so only those two pairs are confirmatory. Everything else is exploratory.
CONFIRMATORY_PAIRS = [
    ("smsdt_st", "smsdt_s", "H1: ST vs S"),
    ("smsdt_full", "smsdt_s", "H1: Full vs S"),
]
EXPLORATORY_PAIRS = [
    ("smsdt_full", "smsdt_st", "Exploratory: Full vs ST (forensic stream + LSE readout)"),
    ("smsdt_s", "gend_ours", "Exploratory: S vs gend_ours (training recipe, same pipeline)"),
    ("smsdt_full", "gend_ours", "Exploratory: Full vs gend_ours"),
    ("smsdt_s", "gend_official", "G1: S vs official GenD weights (same pipeline)"),
    ("gend_ours", "gend_official", "G1: gend_ours vs official GenD weights (same pipeline)"),
]


def align_runs_on_video_id(
    dfs_a: List[pd.DataFrame],
    dfs_b: List[pd.DataFrame],
    label_col: str = "label",
) -> Tuple[List[pd.DataFrame], List[pd.DataFrame], int]:
    """Restrict every run to the videos present in all runs, sorted by video_id.

    Raises ValueError if a video has different labels in different runs, or if a
    run contains duplicate video_ids.

    Returns:
        (aligned_a, aligned_b, n_dropped) where n_dropped is the number of videos of
        the largest run that were not present in every run.
    """
    all_dfs = list(dfs_a) + list(dfs_b)
    for df in all_dfs:
        if df["video_id"].duplicated().any():
            dup = df.loc[df["video_id"].duplicated(), "video_id"].iloc[0]
            raise ValueError(f"Duplicate video_id in a score file: {dup}")
    common = set(all_dfs[0]["video_id"])
    for df in all_dfs[1:]:
        common &= set(df["video_id"])
    if not common:
        raise ValueError("Runs share no video_id; cannot pair them.")
    n_max = max(len(df) for df in all_dfs)

    def _restrict(df):
        out = df[df["video_id"].isin(common)].sort_values("video_id").reset_index(drop=True)
        return out

    aligned_a = [_restrict(df) for df in dfs_a]
    aligned_b = [_restrict(df) for df in dfs_b]
    ref_labels = aligned_a[0][label_col].values
    for df in aligned_a[1:] + aligned_b:
        if not np.array_equal(df[label_col].values, ref_labels):
            raise ValueError("Label mismatch between runs for the same video_id.")
    return aligned_a, aligned_b, int(n_max - len(common))


def dataset_key(dataset_name: str) -> str:
    """Normalise a dataset token from a file name (e.g. 'celebdf', 'FFPP') to a key."""
    return str(dataset_name).strip().lower()


def apply_pure_negatives(df: pd.DataFrame, dataset_name: str) -> Tuple[pd.DataFrame, Dict]:
    """Keep all fakes and only the reals that belong to the dataset itself.

    Example: DFD score files contain 383 reals (140 FF++ YouTube + 243 DFD actors).
    The pure DFD AUC uses the 243 DFD-actor reals only.

    Returns:
        (filtered_df, info) where info records how many reals were kept and removed.
    """
    key = dataset_key(dataset_name)
    sources = PURE_NEGATIVE_SOURCES.get(key, None)
    n_real = int((df["label"] == 0).sum())
    if sources is None or "real_source" not in df.columns:
        return df, {"pure": sources is None, "n_real_kept": n_real, "n_real_removed": 0}
    keep = (df["label"] == 1) | (df["real_source"].astype(str).isin(sources))
    out = df[keep].reset_index(drop=True)
    kept = int((out["label"] == 0).sum())
    if kept == 0:
        raise ValueError(
            f"No reals with real_source in {sources} for dataset '{dataset_name}'. "
            "Check the real_source column before trusting any pure-source number."
        )
    return out, {"pure": True, "n_real_kept": kept, "n_real_removed": n_real - kept}


def bonferroni_level(n_comparisons: int, family_alpha: float = 0.05) -> float:
    """Two-sided confidence level for each of m comparisons (pre-registered: 1 - 0.05/m)."""
    m = max(1, int(n_comparisons))
    return 1.0 - family_alpha / m


def h1_verdict(classifications: List[str]) -> str:
    """Pre-registered H1 rule (docs/decisions.md, section 1.1).

    Supported if every confirmatory comparison is 'equivalent' or 'worse'.
    Falsified if any confirmatory comparison is 'better'.
    Otherwise (some 'inconclusive', none 'better') the study cannot decide H1.
    """
    if not classifications:
        return "not_evaluated"
    if any(c == "better" for c in classifications):
        return "falsified"
    if all(c in ("equivalent", "worse") for c in classifications):
        return "supported"
    return "undetermined (underpowered)"


def required_size_multiplier(ci_half_width: float, point_delta: float, delta: float) -> float:
    """Rough factor by which the evaluation set must grow so that the CI can resolve
    the comparison against the equivalence margin.

    For an equivalence verdict the interval must fit inside (-delta, +delta), i.e. the
    half-width must shrink below delta - |point_delta|. Half-width scales like
    1/sqrt(N), so N must grow by (current / target)^2. Returns inf when |point_delta|
    >= delta (equivalence cannot be shown; only 'better'/'worse' remain possible).
    """
    target = delta - abs(point_delta)
    if target <= 0:
        return float("inf")
    return float((ci_half_width / target) ** 2)


paired_cluster_bootstrap = paired_cluster_boot_dauc


def resolve_cluster_id(df: pd.DataFrame) -> pd.Series:
    """Ensure fine-grained identity/video cluster IDs for cluster bootstrap."""
    if "cluster_id" in df.columns and df["cluster_id"].nunique() > 10:
        return df["cluster_id"]
    
    # Derive from video_id
    clusters = []
    import re
    for vid in df["video_id"]:
        vid_str = str(vid)
        m = re.search(r"id\d+", vid_str)
        if m:
            clusters.append(m.group(0))
        else:
            clusters.append(vid_str)
    return pd.Series(clusters, index=df.index)


def _parse_score_filename(path: str):
    """'smsdt_full_seed2_celebdf.csv' -> ('smsdt_full', 'seed2', 'celebdf')."""
    import os
    base = os.path.splitext(os.path.basename(path))[0]
    parts = base.split("_")
    for idx, p in enumerate(parts):
        if p.startswith("seed") and idx + 1 < len(parts):
            return "_".join(parts[:idx]), p, "_".join(parts[idx + 1:])
    return None


def main():
    import glob
    import json
    import os
    import argparse
    from tools.make_tables import df_to_markdown

    parser = argparse.ArgumentParser(description="Statistical Testing & Hypothesis Evaluation (H1)")
    parser.add_argument("--scores-dir", type=str, default="results/scores", help="Scores directory")
    parser.add_argument("--n-boot", type=int, default=10000,
                        help="Bootstrap iterations (pre-registered: 10,000)")
    parser.add_argument("--delta", type=float, default=None,
                        help="Equivalence margin. Default: pre-registered rule (0.01, or min(2*sd_S, 0.02) "
                             "if the seed SD of S on Celeb-DF exceeds 0.005)")
    parser.add_argument("--confirmatory", nargs="*", default=None,
                        help="Override the confirmatory datasets (default: celebdf, dfdc, wdf, ffiw if present)")
    parser.add_argument("--mixed-negatives", action="store_true",
                        help="Use all reals in each file (old behaviour). Default is pure-source negatives.")
    parser.add_argument("--out-dir", type=str, default="results/tables")
    args = parser.parse_args()

    score_files = glob.glob(os.path.join(args.scores_dir, "*.csv"))
    if not score_files:
        print(f"No score CSVs found in {args.scores_dir}.")
        return

    runs = {}
    purity = {}
    for sf in score_files:
        parsed = _parse_score_filename(sf)
        if parsed is None:
            continue
        model, seed, dataset = parsed
        df = pd.read_csv(sf)
        if not args.mixed_negatives:
            df, info = apply_pure_negatives(df, dataset)
            purity[(model, seed, dataset)] = info
        if "video_id" in df.columns:
            df = df.sort_values("video_id").reset_index(drop=True)
            df["resolved_cluster_id"] = resolve_cluster_id(df).values
        else:
            df["resolved_cluster_id"] = df["cluster_id"] if "cluster_id" in df.columns else df.index.astype(str)
        runs.setdefault((model, dataset), {})[seed] = df

    datasets = sorted({ds for (_, ds) in runs.keys()})
    roles = {ds: DEFAULT_DATASET_ROLES.get(dataset_key(ds), "unknown") for ds in datasets}
    if args.confirmatory is not None:
        conf = {dataset_key(d) for d in args.confirmatory}
        roles = {ds: ("confirmatory" if dataset_key(ds) in conf else
                      ("in_domain" if roles[ds] == "in_domain" else "exploratory")) for ds in datasets}

    # Pre-registered margin rule
    delta = args.delta
    delta_note = "explicit --delta"
    if delta is None:
        delta, delta_note = 0.01, "pre-registered default"
        s_runs = runs.get(("smsdt_s", "celebdf"), {})
        if len(s_runs) >= 2:
            s_aucs = [fast_auc(d["label"].values, d["score_mean"].values) for d in s_runs.values()]
            sd = float(np.std(s_aucs, ddof=1))
            if sd > 0.005:
                delta = min(2 * sd, 0.02)
                delta_note = f"pre-registered rule: sd_S(Celeb-DF)={sd:.4f} > 0.005"

    # Bonferroni family = confirmatory pairs x confirmatory datasets that exist
    conf_ds = [ds for ds in datasets if roles[ds] == "confirmatory"]
    m_family = sum(1 for (a, b, _) in CONFIRMATORY_PAIRS for ds in conf_ds
                   if (a, ds) in runs and (b, ds) in runs)
    level_conf = bonferroni_level(m_family) if m_family > 0 else 0.95

    print("=" * 90)
    print("HYPOTHESIS H1: PAIRED CLUSTER BOOTSTRAP DELTA-AUC TESTS")
    print(f"  delta = {delta:.4f} ({delta_note}); n_boot = {args.n_boot}; "
          f"confirmatory family m = {m_family}; per-comparison level = {level_conf:.4f}")
    print(f"  negatives: {'MIXED (all reals)' if args.mixed_negatives else 'pure-source'}")
    print(f"  dataset roles: {roles}")
    print("=" * 90)

    results = []
    for pair_list, kind in ((CONFIRMATORY_PAIRS, "confirmatory"), (EXPLORATORY_PAIRS, "exploratory")):
        for mod_a, mod_b, desc in pair_list:
            for ds in datasets:
                if (mod_a, ds) not in runs or (mod_b, ds) not in runs:
                    continue
                common_seeds = sorted(set(runs[(mod_a, ds)]) & set(runs[(mod_b, ds)]))
                if not common_seeds and (mod_b == "gend_official" or mod_a == "gend_official"):
                    # Official weights have no training seed: pair every seed of the
                    # other model with the single official run.
                    off = runs[(mod_b, ds)] if mod_b == "gend_official" else runs[(mod_a, ds)]
                    off_df = list(off.values())[0]
                    other = runs[(mod_a, ds)] if mod_b == "gend_official" else runs[(mod_b, ds)]
                    common_seeds = sorted(other)
                    dfs_a = [other[s] for s in common_seeds] if mod_b == "gend_official" else [off_df] * len(common_seeds)
                    dfs_b = [off_df] * len(common_seeds) if mod_b == "gend_official" else [other[s] for s in common_seeds]
                elif not common_seeds:
                    continue
                else:
                    dfs_a = [runs[(mod_a, ds)][s] for s in common_seeds]
                    dfs_b = [runs[(mod_b, ds)][s] for s in common_seeds]

                is_conf = kind == "confirmatory" and roles[ds] == "confirmatory"
                level = level_conf if is_conf else 0.95
                st = paired_cluster_boot_dauc(
                    dfs_a, dfs_b,
                    cluster_col="resolved_cluster_id",
                    n_boot=args.n_boot,
                    level=level,
                    delta=delta,
                )
                results.append({
                    "Dataset": ds.upper(),
                    "Role": roles[ds],
                    "Kind": "confirmatory" if is_conf else "exploratory",
                    "Model A": mod_a,
                    "Model B": mod_b,
                    "Comparison": desc,
                    "AUC (A)": st["mean_a"],
                    "AUC (B)": st["mean_b"],
                    "dAUC (A-B)": st["mean_delta"],
                    "CI level": level,
                    "CI Lower": st["ci_lower"],
                    "CI Upper": st["ci_upper"],
                    "Classification": st["classification"],
                    "N multiplier to resolve": required_size_multiplier(
                        st["ci_half_width"], st["mean_delta"], delta),
                    "Seeds": len(common_seeds),
                    "Clusters": st["n_clusters"],
                    "Samples": st["n_samples"],
                    "Dropped (unaligned)": st["n_dropped_unaligned"],
                })

    if not results:
        print("No matching model pairs found in scores.")
        return

    res_df = pd.DataFrame(results)
    conf_rows = res_df[res_df["Kind"] == "confirmatory"]
    verdict = h1_verdict(conf_rows["Classification"].tolist())
    seeds_available = sorted({int(s.replace("seed", "")) for (m, ds), d in runs.items() for s in d
                              if s.replace("seed", "").isdigit()})

    print("\n" + df_to_markdown(res_df) + "\n")
    print(f"H1 VERDICT (pre-registered rule): {verdict}")
    if len(seeds_available) < 5:
        print(f"  NOTE: pre-registration specifies seeds 1-5; available seeds: {seeds_available}. "
              "Log this as a deviation until all seeds exist.")
    print("=" * 90)

    os.makedirs(args.out_dir, exist_ok=True)
    res_df.to_csv(os.path.join(args.out_dir, "hypothesis_h1_stats.csv"), index=False, encoding="utf-8")
    meta = {
        "verdict": verdict,
        "delta": delta,
        "delta_rule": delta_note,
        "n_boot": args.n_boot,
        "bonferroni_m": m_family,
        "confirmatory_level": level_conf,
        "negatives": "mixed" if args.mixed_negatives else "pure",
        "dataset_roles": roles,
        "seeds_available": seeds_available,
        "purity": {f"{k[0]}_{k[1]}_{k[2]}": v for k, v in purity.items()},
    }
    with open(os.path.join(args.out_dir, "hypothesis_h1_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    with open(os.path.join(args.out_dir, "hypothesis_h1_stats.md"), "w", encoding="utf-8") as f:
        f.write("# Hypothesis H1: Paired Cluster Bootstrap Delta-AUC\n\n")
        f.write(f"- Verdict (pre-registered rule): **{verdict}**\n")
        f.write(f"- delta = {delta:.4f} ({delta_note}); n_boot = {args.n_boot}; "
                f"Bonferroni m = {m_family}, per-comparison level = {level_conf:.4f}\n")
        f.write(f"- Negatives: {'mixed' if args.mixed_negatives else 'pure-source'}; seeds: {seeds_available}\n")
        f.write("- Only rows with Kind = confirmatory decide H1. 'N multiplier to resolve' is the rough factor "
                "by which the evaluation set must grow before an equivalence verdict is possible.\n\n")
        f.write(df_to_markdown(res_df) + "\n")
    print(f"Saved {args.out_dir}/hypothesis_h1_stats.csv, .md and hypothesis_h1_meta.json")


if __name__ == "__main__":
    main()
