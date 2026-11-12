"""Threshold transfer audit and error decomposition engine for S-MSDT TMLR study.

Computes:
1. HTER at three thresholds: raw 0.50, validation-frozen tau_val, and test-optimal oracle tau*
2. Operating point gap: g = HTER(tau_val) - min_tau HTER(tau) >= 0, with a cluster-bootstrap CI
3. Decomposition of error into FPR (per real_source, Wilson CI) and FNR (per manipulation)
4. False-positive inflation: FPR at the frozen threshold on each dataset relative to the
   in-domain FF++ FPR of the same run (exploratory hypothesis H4)

Correctness notes (2026-11-10 audit):
- tau_val from `eval_cross --fit-calib` is on the Platt-calibrated scale. It is applied
  to sigmoid(a * logit_mean + b), equivalently to logit_mean at the frozen logit
  threshold (tools/calib_utils.py). It is never compared with raw score_mean.
- A run without a calibration file is not silently audited at 0.5; see --allow-raw-threshold.
- The oracle threshold is searched exactly over all score values, not a 1000-point grid.
- Negatives are pure-source by default (DFD fakes vs DFD-actor reals only).
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

from tools.calib_utils import load_calib, resolve_decision_scores, wilson_interval


def compute_rates(labels: np.ndarray, scores: np.ndarray, tau: float) -> Tuple[float, float, float]:
    """Compute (FPR, FNR, HTER) at threshold tau (predict fake if score >= tau)."""
    preds = (scores >= tau).astype(np.int32)
    pos = (labels == 1)
    neg = (labels == 0)

    n_pos = int(np.sum(pos))
    n_neg = int(np.sum(neg))

    fn = int(np.sum((preds == 0) & pos))
    fp = int(np.sum((preds == 1) & neg))

    fnr = float(fn / n_pos) if n_pos > 0 else 0.0
    fpr = float(fp / n_neg) if n_neg > 0 else 0.0
    hter = 0.5 * (fpr + fnr)
    return fpr, fnr, hter


def _hter_curve(labels: np.ndarray, scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Exact FPR, FNR and HTER at every candidate threshold.

    Candidates are every distinct score plus one value above the maximum (predict all
    real). Rule: fake if score >= t.
    """
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=np.float64)
    neg = np.sort(scores[labels == 0])
    pos = np.sort(scores[labels == 1])
    cand = np.unique(scores)
    cand = np.append(cand, cand[-1] + 1.0 if len(cand) else 1.0)
    n_neg, n_pos = max(len(neg), 1), max(len(pos), 1)
    fpr = (len(neg) - np.searchsorted(neg, cand, side="left")) / n_neg
    fnr = np.searchsorted(pos, cand, side="left") / n_pos
    return cand, fpr, fnr, 0.5 * (fpr + fnr)


def find_oracle_tau(labels: np.ndarray, scores: np.ndarray, num_thresh: Optional[int] = None) -> Tuple[float, float, float, float]:
    """Find threshold tau* that minimizes test-set HTER (oracle reference).

    The search is exact over all score values. `num_thresh` is accepted for backward
    compatibility and ignored.

    Returns:
        (best_tau, best_fpr, best_fnr, min_hter)
    """
    cand, fpr, fnr, hter = _hter_curve(labels, scores)
    i = int(np.argmin(hter))
    return float(cand[i]), float(fpr[i]), float(fnr[i]), float(hter[i])


def bootstrap_gap_ci(
    labels: np.ndarray,
    scores: np.ndarray,
    tau_val: float,
    clusters: np.ndarray,
    n_boot: int = 10000,
    level: float = 0.95,
    random_seed: int = 0,
) -> Tuple[float, float, np.ndarray]:
    """Cluster-bootstrap CI for g = HTER(tau_val) - min_tau HTER(tau).

    The oracle is re-optimised on every resample, as the per-dataset optimum is part
    of the quantity being estimated.
    """
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=np.float64)
    clusters = np.asarray(clusters)
    uniq = np.unique(clusters)
    idx_by = {c: np.where(clusters == c)[0] for c in uniq}
    rng = np.random.default_rng(random_seed)
    gaps = np.empty(n_boot)
    for b in range(n_boot):
        for _ in range(1000):
            pick = rng.choice(uniq, size=len(uniq), replace=True)
            idx = np.concatenate([idx_by[c] for c in pick])
            y = labels[idx]
            if 0 < y.sum() < len(y):
                break
        s = scores[idx]
        _, _, h_val = compute_rates(y, s, tau_val)
        _, _, _, h_ora = find_oracle_tau(y, s)
        gaps[b] = max(0.0, h_val - h_ora)
    alpha = (1 - level) / 2
    return float(np.quantile(gaps, alpha)), float(np.quantile(gaps, 1 - alpha)), gaps


def audit_threshold_transfer(
    df: pd.DataFrame,
    tau_val: float,
    score_col: str = "score_mean",
    n_boot: int = 0,
    cluster_col: Optional[str] = None,
    random_seed: int = 0,
) -> Dict:
    """Perform full threshold-transfer audit on a dataset.

    Args:
        df: DataFrame matching score schema
        tau_val: Validation-frozen threshold ON THE SCALE OF score_col
        score_col: Score column name (use a logit column with a logit threshold when the
            frozen threshold comes from Platt calibration)
        n_boot: if > 0, compute a cluster-bootstrap CI for the gap g
        cluster_col: cluster column for the bootstrap (defaults to cluster_id if present)

    Returns:
        dict: Full decomposition dictionary
    """
    labels = df["label"].values.astype(np.int32)
    scores = df[score_col].values.astype(np.float64)

    # 1. Raw threshold 0.50 (only meaningful on a probability column)
    if "score_mean" in df.columns:
        fpr_raw, fnr_raw, hter_raw = compute_rates(labels, df["score_mean"].values.astype(np.float64), 0.50)
    else:
        fpr_raw, fnr_raw, hter_raw = compute_rates(labels, scores, 0.50)

    # 2. Frozen validation threshold
    fpr_val, fnr_val, hter_val = compute_rates(labels, scores, tau_val)

    # 3. Oracle test-set threshold (exact)
    tau_ora, fpr_ora, fnr_ora, hter_ora = find_oracle_tau(labels, scores)

    # 4. Operating point gap g
    g = max(0.0, hter_val - hter_ora)
    g_lo, g_hi = float("nan"), float("nan")
    if n_boot and n_boot > 0:
        # Resolve identity clusters (a coarse cluster_id such as Celeb-DF's 3 source
        # labels collapses the bootstrap; see docs/decisions.md D13).
        from tools.fpr_control import resolve_clusters
        clusters = resolve_clusters(df, cluster_col or "cluster_id")
        g_lo, g_hi, _ = bootstrap_gap_ci(labels, scores, tau_val, clusters, n_boot=n_boot, random_seed=random_seed)

    # 5. Decomposition per real_source (FPR breakdown, Wilson 95% CI)
    real_df = df[df["label"] == 0]
    fpr_by_source = {}
    src_col = "real_source" if "real_source" in df.columns else None
    for src in (real_df[src_col].fillna("").unique() if src_col else ["all"]):
        sub = real_df[real_df[src_col].fillna("") == src] if src_col else real_df
        n_src = len(sub)
        fp_src = int(np.sum(sub[score_col].values >= tau_val))
        lo, hi = wilson_interval(fp_src, n_src)
        fpr_by_source[str(src)] = {
            "n_reals": n_src,
            "false_positives": fp_src,
            "fpr": float(fp_src / n_src) if n_src > 0 else 0.0,
            "fpr_ci95": [lo, hi],
        }

    # 6. Decomposition per manipulation (FNR breakdown)
    fake_df = df[df["label"] == 1]
    fnr_by_manip = {}
    man_col = "manipulation" if "manipulation" in df.columns else None
    for manip in (fake_df[man_col].fillna("").unique() if man_col else ["all"]):
        sub = fake_df[fake_df[man_col].fillna("") == manip] if man_col else fake_df
        n_manip = len(sub)
        fn_manip = int(np.sum(sub[score_col].values < tau_val))
        fnr_by_manip[str(manip)] = {
            "n_fakes": n_manip,
            "false_negatives": fn_manip,
            "fnr": float(fn_manip / n_manip) if n_manip > 0 else 0.0
        }

    n_neg = int((labels == 0).sum())
    fp_total = int(np.sum(scores[labels == 0] >= tau_val))
    fpr_lo, fpr_hi = wilson_interval(fp_total, n_neg)

    return {
        "tau_val": float(tau_val),
        "tau_oracle": float(tau_ora),
        "hter_raw_05": float(hter_raw),
        "fpr_raw_05": float(fpr_raw),
        "fnr_raw_05": float(fnr_raw),
        "hter_val_frozen": float(hter_val),
        "fpr_val_frozen": float(fpr_val),
        "fpr_val_frozen_ci95": [fpr_lo, fpr_hi],
        "fnr_val_frozen": float(fnr_val),
        "hter_oracle": float(hter_ora),
        "fpr_oracle": float(fpr_ora),
        "fnr_oracle": float(fnr_ora),
        "gap_g": float(g),
        "gap_g_ci95": [g_lo, g_hi],
        "n_reals": n_neg,
        "fpr_by_source": fpr_by_source,
        "fnr_by_manipulation": fnr_by_manip
    }


def h2_verdict(gap_cis_by_dataset: Dict[str, Tuple[float, float]], threshold: float = 0.05) -> str:
    """Pre-registered H2 rule for one model (docs/decisions.md 1.1).

    Supported if the lower CI bound of g is >= 0.05 on at least one confirmatory dataset.
    Falsified if the upper CI bound of g is < 0.05 on every confirmatory dataset.
    """
    if not gap_cis_by_dataset:
        return "not_evaluated"
    if any(lo >= threshold for lo, _ in gap_cis_by_dataset.values()):
        return "supported"
    if all(hi < threshold for _, hi in gap_cis_by_dataset.values()):
        return "falsified"
    return "undetermined"


def print_audit_report(res: Dict, dataset_name: str = "Evaluation Dataset"):
    """Format and display audit report."""
    print("=" * 80)
    print(f"THRESHOLD-TRANSFER AUDIT REPORT: {dataset_name}")
    print("=" * 80)
    print(f"  Validation-Frozen Threshold (tau_val): {res['tau_val']:.4f}")
    print(f"  Test-Optimal Oracle Threshold (tau*):  {res['tau_oracle']:.4f}")
    print(f"  Operating Point Gap g (tau_val vs tau*): {res['gap_g']:.4f}  CI95 {res['gap_g_ci95']}")
    print("-" * 80)
    print(f"{'Operating Point':<30} | {'HTER':<10} | {'FPR':<10} | {'FNR':<10}")
    print("-" * 80)
    print(f"{'Raw Threshold (0.50)':<30} | {res['hter_raw_05']:<10.4f} | {res['fpr_raw_05']:<10.4f} | {res['fnr_raw_05']:<10.4f}")
    print(f"{'Frozen Calibrated (tau_val)':<30} | {res['hter_val_frozen']:<10.4f} | {res['fpr_val_frozen']:<10.4f} | {res['fnr_val_frozen']:<10.4f}")
    print(f"{'[Oracle] Optimum (tau*)':<30} | {res['hter_oracle']:<10.4f} | {res['fpr_oracle']:<10.4f} | {res['fnr_oracle']:<10.4f}")
    print("-" * 80)

    print("\n[False-Alarm Decomposition per Real Source (FPR, Wilson 95% CI)]")
    for src, stats in res["fpr_by_source"].items():
        src_label = src if src else "unspecified_real"
        lo, hi = stats["fpr_ci95"]
        print(f"  • {src_label:<25} : FPR = {stats['fpr']:6.2%} ({stats['false_positives']}/{stats['n_reals']}) [{lo:.2%}, {hi:.2%}]")

    print("\n[Miss-Rate Decomposition per Manipulation (FNR)]")
    for manip, stats in res["fnr_by_manipulation"].items():
        manip_label = manip if manip else "unspecified_fake"
        print(f"  • {manip_label:<25} : FNR = {stats['fnr']:6.2%} ({stats['false_negatives']}/{stats['n_fakes']})")
    print("=" * 80)


def main():
    import glob
    import argparse
    from tools.make_tables import df_to_markdown
    from tools.stats import (apply_pure_negatives, dataset_key, DEFAULT_DATASET_ROLES,
                             _parse_score_filename)

    parser = argparse.ArgumentParser(description="Operating Point Transfer & Error Decomposition (H2)")
    parser.add_argument("--scores", type=str, default=None, help="Specific score CSV path to audit")
    parser.add_argument("--scores-dir", type=str, default="results/scores", help="Scores directory to audit")
    parser.add_argument("--calib-dir", type=str, default="results/calib", help="Calibration parameters directory")
    parser.add_argument("--tau-val", type=float, default=None,
                        help="Explicit threshold on the score_mean scale (single-file mode only)")
    parser.add_argument("--allow-raw-threshold", action="store_true",
                        help="Audit runs without a calibration file at raw 0.5 (labelled raw_0.5, excluded from H2)")
    parser.add_argument("--mixed-negatives", action="store_true", help="Use all reals (old behaviour)")
    parser.add_argument("--n-boot", type=int, default=10000, help="Cluster-bootstrap resamples for the CI of g")
    parser.add_argument("--out-dir", type=str, default="results/tables")
    args = parser.parse_args()

    if args.scores:
        if not os.path.exists(args.scores):
            print(f"File not found: {args.scores}")
            return
        df = pd.read_csv(args.scores)
        if args.tau_val is None:
            raise SystemExit("Single-file mode needs --tau-val (on the score_mean scale).")
        res = audit_threshold_transfer(df, tau_val=args.tau_val, n_boot=args.n_boot)
        print_audit_report(res, dataset_name=os.path.basename(args.scores))
        return

    score_files = sorted(glob.glob(os.path.join(args.scores_dir, "*.csv")))
    if not score_files:
        print(f"No score CSVs found in {args.scores_dir}.")
        return

    # Pass 1: load everything so the in-domain FPR of each run is available
    loaded = {}
    for sf in score_files:
        parsed = _parse_score_filename(sf)
        if parsed is None:
            continue
        model, seed, dname = parsed
        df = pd.read_csv(sf)
        if not args.mixed_negatives:
            df, _ = apply_pure_negatives(df, dname)
        loaded[(model, seed, dname)] = df

    summary_rows, skipped = [], []
    details = {}
    ffpp_fpr = {}
    print("=" * 90)
    print("HYPOTHESIS H2: OPERATING POINT TRANSFER & THRESHOLD DEGRADATION AUDIT")
    print("=" * 90)

    # FF++ first so that inflation ratios can be computed
    order = sorted(loaded, key=lambda k: (0 if dataset_key(k[2]).startswith("ffpp") else 1, k))
    for (model, seed, dname) in order:
        run_name = f"{model}_{seed}"
        df = loaded[(model, seed, dname)]
        calib = load_calib(run_name, args.calib_dir)
        try:
            scores, thr, tau_source = resolve_decision_scores(df, calib, allow_raw=args.allow_raw_threshold)
        except FileNotFoundError:
            skipped.append((run_name, dname, "no calibration file"))
            continue
        work = df.copy()
        work["_decision"] = scores
        res = audit_threshold_transfer(
            work, tau_val=thr, score_col="_decision", n_boot=args.n_boot,
            cluster_col="cluster_id" if "cluster_id" in work.columns else None,
        )
        key = dataset_key(dname)
        if key.startswith("ffpp"):
            ffpp_fpr[run_name] = res["fpr_val_frozen"]
        ref = ffpp_fpr.get(run_name)
        inflation = (res["fpr_val_frozen"] / ref) if ref and ref > 0 else float("nan")
        details[f"{run_name}_{dname}"] = res
        summary_rows.append({
            "Model": model,
            "Seed": seed,
            "Dataset": dname.upper(),
            "Role": DEFAULT_DATASET_ROLES.get(key, "unknown"),
            "Threshold source": tau_source,
            "tau_cal": calib["tau"] if calib else 0.5,
            "HTER@0.50 (raw)": res["hter_raw_05"],
            "HTER(tau_val)": res["hter_val_frozen"],
            "HTER(tau*)": res["hter_oracle"],
            "Gap g": res["gap_g"],
            "g CI low": res["gap_g_ci95"][0],
            "g CI high": res["gap_g_ci95"][1],
            "FPR(tau_val)": res["fpr_val_frozen"],
            "FPR CI low": res["fpr_val_frozen_ci95"][0],
            "FPR CI high": res["fpr_val_frozen_ci95"][1],
            "FNR(tau_val)": res["fnr_val_frozen"],
            "N reals": res["n_reals"],
            "FPR inflation vs FF++": inflation,
        })

    if skipped:
        print(f"\nSkipped {len(skipped)} run/dataset pairs with no calibration file "
              f"(fit one on the dev split, or use --allow-raw-threshold):")
        for r in skipped:
            print(f"  - {r[0]} / {r[1]}")

    if not summary_rows:
        print("Nothing audited.")
        return

    sum_df = pd.DataFrame(summary_rows)

    # Pre-registered H2 verdicts: for S and for gend_ours, on confirmatory datasets,
    # calibrated thresholds only. Seeds are pooled conservatively: a dataset counts as
    # "supported" only if every seed's lower bound clears 0.05, "falsified" only if every
    # seed's upper bound is below it.
    verdicts = {}
    for model in ("smsdt_s", "gend_ours"):
        rows = sum_df[(sum_df["Model"] == model) & (sum_df["Role"] == "confirmatory")
                      & (sum_df["Threshold source"] == "calib")]
        cis = {}
        for ds, grp in rows.groupby("Dataset"):
            cis[ds] = (float(grp["g CI low"].min()), float(grp["g CI high"].max()))
        verdicts[model] = h2_verdict(cis)

    print("\n" + df_to_markdown(sum_df) + "\n")
    print(f"H2 verdicts (pre-registered rule, calibrated thresholds, confirmatory datasets): {verdicts}")
    print("=" * 90)

    os.makedirs(args.out_dir, exist_ok=True)
    sum_df.to_csv(os.path.join(args.out_dir, "threshold_transfer_audit.csv"), index=False, encoding="utf-8")
    with open(os.path.join(args.out_dir, "threshold_transfer_details.json"), "w", encoding="utf-8") as f:
        json.dump({"verdicts": verdicts, "skipped": skipped, "details": details}, f, indent=2)
    with open(os.path.join(args.out_dir, "threshold_transfer_audit.md"), "w", encoding="utf-8") as f:
        f.write("# Hypothesis H2: Operating Point Transfer & Threshold Degradation Audit\n\n")
        f.write(f"- H2 verdicts (pre-registered rule): {verdicts}\n")
        f.write("- tau_val is applied on the calibrated scale (logit threshold from a, b, tau).\n")
        f.write("- Rows with Threshold source = raw_0.5 are NOT tau_val and do not enter H2.\n")
        f.write("- 'FPR inflation vs FF++' = FPR at the frozen threshold on this dataset / FPR on FF++ "
                "(pure YouTube reals) for the same run. Exploratory (H4).\n\n")
        f.write(df_to_markdown(sum_df) + "\n")
    print(f"Saved {args.out_dir}/threshold_transfer_audit.csv, .md and threshold_transfer_details.json")


if __name__ == "__main__":
    main()
