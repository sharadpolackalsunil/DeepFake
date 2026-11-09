"""Threshold transfer audit and error decomposition engine for S-MSDT TMLR study.

Computes:
1. HTER at three thresholds: raw 0.50, validation-frozen tau_val, and test-optimal oracle tau*
2. Operating point gap: g = HTER(tau_val) - min_tau HTER(tau) >= 0
3. Decomposition of error into FPR (per real_source) and FNR (per manipulation)
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional


def compute_rates(labels: np.ndarray, scores: np.ndarray, tau: float) -> Tuple[float, float, float]:
    """Compute (FPR, FNR, HTER) at threshold tau."""
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


def find_oracle_tau(labels: np.ndarray, scores: np.ndarray, num_thresh: int = 1000) -> Tuple[float, float, float, float]:
    """Find threshold tau* that minimizes test-set HTER (oracle reference).
    
    Returns:
        (best_tau, best_fpr, best_fnr, min_hter)
    """
    thresholds = np.linspace(scores.min(), scores.max(), num_thresh)
    best_hter = float("inf")
    best_tau = 0.5
    best_fpr, best_fnr = 0.0, 0.0
    
    for t in thresholds:
        fpr, fnr, hter = compute_rates(labels, scores, t)
        if hter < best_hter:
            best_hter = hter
            best_tau = float(t)
            best_fpr = fpr
            best_fnr = fnr
            
    return best_tau, best_fpr, best_fnr, best_hter


def audit_threshold_transfer(
    df: pd.DataFrame,
    tau_val: float,
    score_col: str = "score_mean"
) -> Dict:
    """Perform full threshold-transfer audit on a dataset.
    
    Args:
        df: DataFrame matching score schema
        tau_val: Validation-frozen Platt+Youden threshold
        score_col: Score column name
        
    Returns:
        dict: Full decomposition dictionary
    """
    labels = df["label"].values.astype(np.int32)
    scores = df[score_col].values.astype(np.float64)
    
    # 1. Raw threshold 0.50
    fpr_raw, fnr_raw, hter_raw = compute_rates(labels, scores, 0.50)
    
    # 2. Frozen validation threshold
    fpr_val, fnr_val, hter_val = compute_rates(labels, scores, tau_val)
    
    # 3. Oracle test-set threshold
    tau_ora, fpr_ora, fnr_ora, hter_ora = find_oracle_tau(labels, scores)
    
    # 4. Operating point gap g
    g = max(0.0, hter_val - hter_ora)
    
    # 5. Decomposition per real_source (FPR breakdown)
    real_df = df[df["label"] == 0]
    fpr_by_source = {}
    for src in real_df["real_source"].unique():
        sub = real_df[real_df["real_source"] == src]
        n_src = len(sub)
        fp_src = int(np.sum(sub[score_col].values >= tau_val))
        fpr_by_source[str(src)] = {
            "n_reals": n_src,
            "false_positives": fp_src,
            "fpr": float(fp_src / n_src) if n_src > 0 else 0.0
        }
        
    # 6. Decomposition per manipulation (FNR breakdown)
    fake_df = df[df["label"] == 1]
    fnr_by_manip = {}
    for manip in fake_df["manipulation"].unique():
        sub = fake_df[fake_df["manipulation"] == manip]
        n_manip = len(sub)
        fn_manip = int(np.sum(sub[score_col].values < tau_val))
        fnr_by_manip[str(manip)] = {
            "n_fakes": n_manip,
            "false_negatives": fn_manip,
            "fnr": float(fn_manip / n_manip) if n_manip > 0 else 0.0
        }
        
    return {
        "tau_val": float(tau_val),
        "tau_oracle": float(tau_ora),
        "hter_raw_05": float(hter_raw),
        "fpr_raw_05": float(fpr_raw),
        "fnr_raw_05": float(fnr_raw),
        "hter_val_frozen": float(hter_val),
        "fpr_val_frozen": float(fpr_val),
        "fnr_val_frozen": float(fnr_val),
        "hter_oracle": float(hter_ora),
        "fpr_oracle": float(fpr_ora),
        "fnr_oracle": float(fnr_ora),
        "gap_g": float(g),
        "fpr_by_source": fpr_by_source,
        "fnr_by_manipulation": fnr_by_manip
    }


def print_audit_report(res: Dict, dataset_name: str = "Evaluation Dataset"):
    """Format and display audit report."""
    print("=" * 80)
    print(f"THRESHOLD-TRANSFER AUDIT REPORT: {dataset_name}")
    print("=" * 80)
    print(f"  Validation-Frozen Threshold (tau_val): {res['tau_val']:.4f}")
    print(f"  Test-Optimal Oracle Threshold (tau*):  {res['tau_oracle']:.4f}")
    print(f"  Operating Point Gap g (tau_val vs tau*): {res['gap_g']:.4f}")
    print("-" * 80)
    print(f"{'Operating Point':<30} | {'HTER':<10} | {'FPR':<10} | {'FNR':<10}")
    print("-" * 80)
    print(f"{'Raw Threshold (0.50)':<30} | {res['hter_raw_05']:<10.4f} | {res['fpr_raw_05']:<10.4f} | {res['fnr_raw_05']:<10.4f}")
    print(f"{'Frozen Calibrated (tau_val)':<30} | {res['hter_val_frozen']:<10.4f} | {res['fpr_val_frozen']:<10.4f} | {res['fnr_val_frozen']:<10.4f}")
    print(f"{'[Oracle] Optimum (tau*)':<30} | {res['hter_oracle']:<10.4f} | {res['fpr_oracle']:<10.4f} | {res['fnr_oracle']:<10.4f}")
    print("-" * 80)
    
    print("\n[False-Alarm Decomposition per Real Source (FPR)]")
    for src, stats in res["fpr_by_source"].items():
        src_label = src if src else "unspecified_real"
        print(f"  • {src_label:<25} : FPR = {stats['fpr']:6.2%} ({stats['false_positives']}/{stats['n_reals']})")
        
    print("\n[Miss-Rate Decomposition per Manipulation (FNR)]")
    for manip, stats in res["fnr_by_manipulation"].items():
        manip_label = manip if manip else "unspecified_fake"
        print(f"  • {manip_label:<25} : FNR = {stats['fnr']:6.2%} ({stats['false_negatives']}/{stats['n_fakes']})")
    print("=" * 80)


if __name__ == "__main__":
    # Test on synthetic dataset matching schema
    rng = np.random.RandomState(42)
    synthetic_records = []
    
    # Authentic reals: 178 Celeb-DF reals
    for i in range(178):
        synthetic_records.append({
            "video_id": f"real_{i:04d}",
            "dataset": "celebdf",
            "split": "test",
            "label": 0,
            "real_source": "celebdf_real",
            "manipulation": "",
            "cluster_id": f"actor_{i % 50}",
            "n_clips": 8,
            "logit_mean": float(rng.normal(-2.5, 0.8)),
            "score_mean": float(1.0 / (1.0 + np.exp(-rng.normal(-2.5, 0.8)))),
            "score_topk50": 0.1
        })
        
    # Manipulated fakes: 340 Celeb-DF swaps
    for i in range(340):
        synthetic_records.append({
            "video_id": f"fake_{i:04d}",
            "dataset": "celebdf",
            "split": "test",
            "label": 1,
            "real_source": "",
            "manipulation": "celeb_swap",
            "cluster_id": f"actor_{i % 50}",
            "n_clips": 8,
            "logit_mean": float(rng.normal(0.2, 1.2)),
            "score_mean": float(1.0 / (1.0 + np.exp(-rng.normal(0.2, 1.2)))),
            "score_topk50": 0.8
        })
        
    test_df = pd.DataFrame(synthetic_records)
    audit = audit_threshold_transfer(test_df, tau_val=0.536)
    print_audit_report(audit, dataset_name="Celeb-DF v2 (Audit Test)")
