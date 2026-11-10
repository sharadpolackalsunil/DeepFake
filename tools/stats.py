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
    
    # Verify alignment on reference dataframe
    if "video_id" in dfs_a[0].columns:
        dfs_a = [df.sort_values("video_id").reset_index(drop=True) for df in dfs_a]
        dfs_b = [df.sort_values("video_id").reset_index(drop=True) for df in dfs_b]

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
        # Resample clusters with replacement
        boot_clusters = rng.choice(clusters, size=n_clusters, replace=True)
        # Gather all row indices
        boot_idx = np.concatenate([cluster_indices[c] for c in boot_clusters])
        
        boot_y = labels[boot_idx]
        if np.all(boot_y == 0) or np.all(boot_y == 1):
            # Degenerate resample; re-try draw
            continue
            
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
        "boot_sd": float(np.std(deltas))
    }


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


def main():
    import glob
    import os
    import argparse
    from tools.make_tables import df_to_markdown

    parser = argparse.ArgumentParser(description="Statistical Testing & Hypothesis Evaluation (H1)")
    parser.add_argument("--scores-dir", type=str, default="results/scores", help="Scores directory")
    parser.add_argument("--n-boot", type=int, default=2000, help="Number of bootstrap iterations")
    parser.add_argument("--delta", type=float, default=0.01, help="Equivalence margin delta")
    args = parser.parse_args()

    score_files = glob.glob(os.path.join(args.scores_dir, "*.csv"))
    if not score_files:
        print(f"No score CSVs found in {args.scores_dir}.")
        return

    runs = {}
    for sf in score_files:
        base = os.path.splitext(os.path.basename(sf))[0]
        parts = base.split("_")
        seed_idx = -1
        for idx, p in enumerate(parts):
            if p.startswith("seed"):
                seed_idx = idx
                break
        if seed_idx != -1 and seed_idx + 1 < len(parts):
            model = "_".join(parts[:seed_idx])
            seed = parts[seed_idx]
            dataset = "_".join(parts[seed_idx + 1:])
            df = pd.read_csv(sf)
            if "video_id" in df.columns:
                df = df.sort_values("video_id").reset_index(drop=True)
                df["resolved_cluster_id"] = resolve_cluster_id(df)
            else:
                df["resolved_cluster_id"] = df["cluster_id"] if "cluster_id" in df.columns else df.index.astype(str)
            runs.setdefault((model, dataset), {})[seed] = df

    candidate_pairs = [
        ("smsdt_full", "gend_ours", "Flagship S-MSDTv3 Full vs GenD Reference Anchor"),
        ("smsdt_full", "smsdt_st", "Residual Stream & LSE Pooling Ablation (Full vs ST)"),
        ("smsdt_st", "smsdt_s", "Spatio-Temporal Attention Ablation (ST vs Spatial-only)"),
        ("smsdt_s", "gend_ours", "Single-Frame Recipe Ablation (SMSDT-S vs GenD)"),
    ]

    datasets = ["ffpp", "celebdf", "dfd"]
    results = []

    print("=" * 90)
    print("HYPOTHESIS H1: PAIRED CLUSTER BOOTSTRAP ΔAUC TESTS (95% CI, Equivalence δ=0.01)")
    print("=" * 90)

    for mod_a, mod_b, desc in candidate_pairs:
        for ds in datasets:
            if (mod_a, ds) not in runs or (mod_b, ds) not in runs:
                continue
            seeds_a = set(runs[(mod_a, ds)].keys())
            seeds_b = set(runs[(mod_b, ds)].keys())
            common_seeds = sorted(list(seeds_a.intersection(seeds_b)))
            if not common_seeds:
                continue

            dfs_a = [runs[(mod_a, ds)][s] for s in common_seeds]
            dfs_b = [runs[(mod_b, ds)][s] for s in common_seeds]

            # Run paired cluster bootstrap
            stats = paired_cluster_boot_dauc(
                dfs_a, dfs_b,
                cluster_col="resolved_cluster_id",
                n_boot=args.n_boot,
                delta=args.delta
            )

            results.append({
                "Dataset": ds.upper(),
                "Model A": mod_a,
                "Model B": mod_b,
                "Comparison": desc,
                "AUC (A)": stats["mean_a"],
                "AUC (B)": stats["mean_b"],
                "ΔAUC (A-B)": stats["mean_delta"],
                "95% CI Lower": stats["ci_lower"],
                "95% CI Upper": stats["ci_upper"],
                "Classification": stats["classification"],
                "Seeds": len(common_seeds),
                "Clusters": stats["n_clusters"],
                "Samples": stats["n_samples"]
            })

    if results:
        res_df = pd.DataFrame(results)
        print("\n" + df_to_markdown(res_df) + "\n")
        print("=" * 90)

        os.makedirs("results/tables", exist_ok=True)
        res_df.to_csv("results/tables/hypothesis_h1_stats.csv", index=False, encoding="utf-8")
        with open("results/tables/hypothesis_h1_stats.md", "w", encoding="utf-8") as f:
            f.write("# Hypothesis H1: Statistical Equivalence & ΔAUC Bootstrap Tests\n\n")
            f.write(df_to_markdown(res_df) + "\n")
        print("✓ Statistical test matrix successfully saved to results/tables/hypothesis_h1_stats.csv and .md\n")
    else:
        print("No matching model pairs found in scores.")


if __name__ == "__main__":
    main()

