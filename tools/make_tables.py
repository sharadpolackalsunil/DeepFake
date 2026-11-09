"""Automated table generation engine for S-MSDT TMLR study.

Generates the full experiment matrix and paper tables directly from raw score CSVs in results/scores/
and calibration parameters in results/calib/. Hand-typed numbers are strictly prohibited.
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import glob
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional

from tools.stats import fast_auc
from tools.threshold_transfer import compute_rates, find_oracle_tau


def compute_dataset_metrics(
    df: pd.DataFrame,
    tau_val: float = 0.50,
    score_col: str = "score_mean"
) -> Dict[str, float]:
    """Compute primary evaluation metrics on a single score DataFrame."""
    labels = df["label"].values.astype(np.int32)
    scores = df[score_col].values.astype(np.float64)
    
    auc = fast_auc(labels, scores)
    
    # HTER and degradation gap g
    fpr_val, fnr_val, hter_val = compute_rates(labels, scores, tau_val)
    _, _, _, hter_ora = find_oracle_tau(labels, scores)
    gap_g = max(0.0, hter_val - hter_ora)
    
    # Biometric utility: TPR @ 5% FPR
    # Sort reals to find score at 95th percentile (5% FPR threshold)
    reals = scores[labels == 0]
    fakes = scores[labels == 1]
    
    if len(reals) > 0 and len(fakes) > 0:
        thresh_5fpr = float(np.percentile(reals, 95.0))
        tpr_5fpr = float(np.mean(fakes >= thresh_5fpr))
    else:
        tpr_5fpr = float("nan")
        
    return {
        "auc": float(auc),
        "hter_val": float(hter_val),
        "gap_g": float(gap_g),
        "tpr_5fpr": float(tpr_5fpr),
        "n_samples": len(df)
    }


def compute_pure_ffpp_macro_auc(df: pd.DataFrame, score_col: str = "score_mean") -> float:
    """Compute pure-negative FF++ Macro AUC (5 methods scored against ffpp_youtube reals only)."""
    pure_reals = df[(df["label"] == 0) & (df["real_source"] == "ffpp_youtube")]
    if len(pure_reals) == 0:
        # Fallback to all reals in split if pure tag not populated
        pure_reals = df[df["label"] == 0]
        
    methods = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures", "FaceShifter"]
    method_aucs = []
    
    for m in methods:
        sub_fakes = df[(df["label"] == 1) & (df["manipulation"] == m)]
        if len(sub_fakes) > 0:
            eval_df = pd.concat([pure_reals, sub_fakes], ignore_index=True)
            m_auc = fast_auc(eval_df["label"].values, eval_df[score_col].values)
            method_aucs.append(m_auc)
            
    return float(np.mean(method_aucs)) if len(method_aucs) > 0 else float("nan")


def generate_experiment_matrix(
    scores_dir: str = "results/scores",
    calib_dir: str = "results/calib"
) -> pd.DataFrame:
    """Scan score directory and generate master experiment matrix across all models."""
    run_dirs = [d for d in glob.glob(os.path.join(scores_dir, "*")) if os.path.isdir(d)]
    
    records = []
    
    for rdir in run_dirs:
        run_name = os.path.basename(rdir)
        # Parse model, seed, date from directory name: e.g., smsdt_s_seed1_20261109
        parts = run_name.split("_")
        seed = 1
        for p in parts:
            if p.startswith("seed"):
                try:
                    seed = int(p.replace("seed", ""))
                except ValueError:
                    pass
                    
        # Match calibration file if available
        calib_file = os.path.join(calib_dir, f"{run_name}.json")
        tau_val = 0.50
        if os.path.exists(calib_file):
            try:
                with open(calib_file) as f:
                    cdata = json.load(f)
                    tau_val = float(cdata.get("youden_tau", 0.50))
            except Exception:
                pass
                
        # Evaluate each score CSV inside run dir
        csvs = glob.glob(os.path.join(rdir, "*.csv"))
        row = {
            "run_name": run_name,
            "seed": seed,
            "tau_val": tau_val
        }
        
        for cpath in csvs:
            dname = os.path.splitext(os.path.basename(cpath))[0]
            try:
                df = pd.read_csv(cpath)
                metrics = compute_dataset_metrics(df, tau_val=tau_val)
                row[f"{dname}_auc"] = metrics["auc"]
                row[f"{dname}_hter"] = metrics["hter_val"]
                row[f"{dname}_gap_g"] = metrics["gap_g"]
                row[f"{dname}_tpr5"] = metrics["tpr_5fpr"]
                
                if dname in ["ffpp", "ffpp_c23"]:
                    row["ffpp_pure_macro_auc"] = compute_pure_ffpp_macro_auc(df)
            except Exception as e:
                row[f"{dname}_err"] = str(e)
                
        records.append(row)
        
    return pd.DataFrame(records)


def main():
    os.makedirs("results/tables", exist_ok=True)
    matrix = generate_experiment_matrix()
    
    out_csv = "results/tables/experiment_matrix.csv"
    out_md = "results/tables/experiment_matrix.md"
    
    matrix.to_csv(out_csv, index=False)
    
    with open(out_md, "w") as f:
        f.write("# Master Experiment Matrix (Auto-Generated)\n\n")
        f.write(f"*Generated from raw score CSVs in `results/scores/` ({len(matrix)} total runs)*\n\n")
        if len(matrix) > 0:
            f.write(matrix.to_markdown(index=False))
        else:
            f.write("*No run scores recorded in `results/scores/` yet.*\n")
            
    print(f"✓ Experiment matrix successfully written to: {out_csv} and {out_md}")


if __name__ == "__main__":
    main()
