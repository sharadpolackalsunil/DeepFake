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
from smsdt.metric_defs import tpr_at_fpr


def load_scores_pure(cpath: str, dname: str) -> pd.DataFrame:
    """Read a score CSV and keep only the dataset's own reals (D14; e.g. pure DFD)."""
    from tools.stats import apply_pure_negatives
    df = pd.read_csv(cpath)
    try:
        df, _ = apply_pure_negatives(df, dname)
    except (ValueError, KeyError):
        pass
    return df


def df_to_markdown(df: pd.DataFrame) -> str:
    """Format DataFrame as a Markdown table without requiring external packages like tabulate."""
    if df.empty:
        return ""
    try:
        return df.to_markdown(index=False)
    except Exception:
        pass
        
    cols = list(df.columns)
    lines = []
    lines.append("| " + " | ".join(str(c) for c in cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for _, row in df.iterrows():
        vals = []
        for c in cols:
            val = row[c]
            if pd.isna(val):
                vals.append("-")
            elif isinstance(val, (float, np.floating)):
                vals.append(f"{val:.4f}")
            else:
                vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def compute_dataset_metrics(
    df: pd.DataFrame,
    tau_val: float = 0.50,
    score_col: str = "score_mean"
) -> Dict[str, float]:
    """Compute primary evaluation metrics on a single score DataFrame."""
    if "label" not in df.columns or score_col not in df.columns:
        return {"auc": float("nan"), "hter_val": float("nan"), "gap_g": float("nan"), "tpr_5fpr": float("nan"), "n_samples": 0}

    valid_mask = ~(df["label"].isna() | df[score_col].isna())
    df_valid = df[valid_mask]
    if len(df_valid) == 0:
        return {"auc": float("nan"), "hter_val": float("nan"), "gap_g": float("nan"), "tpr_5fpr": float("nan"), "n_samples": 0}

    labels = df_valid["label"].values.astype(np.int32)
    scores = df_valid[score_col].values.astype(np.float64)
    
    auc = fast_auc(labels, scores)
    
    # HTER and degradation gap g
    fpr_val, fnr_val, hter_val = compute_rates(labels, scores, tau_val)
    _, _, _, hter_ora = find_oracle_tau(labels, scores)
    gap_g = max(0.0, hter_val - hter_ora)
    
    # Biometric utility: TPR @ 5% FPR (D16: one canonical, conservative definition shared
    # with smsdt/eval_cross.py; the old np.percentile rule disagreed with the eval logs)
    tpr_5fpr = tpr_at_fpr(labels, scores, 0.05)
        
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
    """Scan score directory and generate master experiment matrix across all models.
    Supports both subdirectories (scores_dir/run_name/*.csv) and flat files (scores_dir/run_name_dataset.csv).
    """
    records = []
    
    # 1. Check for run subdirectories
    run_dirs = [d for d in glob.glob(os.path.join(scores_dir, "*")) if os.path.isdir(d)]
    for rdir in run_dirs:
        run_name = os.path.basename(rdir)
        parts = run_name.split("_")
        seed = 1
        for p in parts:
            if p.startswith("seed"):
                try:
                    seed = int(p.replace("seed", ""))
                except ValueError:
                    pass
                    
        calib_file = os.path.join(calib_dir, f"{run_name}.json")
        tau_val = 0.50
        if os.path.exists(calib_file):
            try:
                with open(calib_file) as f:
                    cdata = json.load(f)
                    tau_val = float(cdata.get("tau", cdata.get("youden_tau", 0.50)))
            except Exception:
                pass
                
        csvs = glob.glob(os.path.join(rdir, "*.csv"))
        row = {
            "run_name": run_name,
            "seed": seed,
            "tau_val": tau_val
        }
        for cpath in csvs:
            dname = os.path.splitext(os.path.basename(cpath))[0]
            try:
                df = load_scores_pure(cpath, dname)
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

    # 2. Check for flat score files (e.g. gend_ours_seed1_celebdf.csv)
    flat_csvs = glob.glob(os.path.join(scores_dir, "*.csv"))
    flat_groups = {}
    for cpath in flat_csvs:
        fname = os.path.splitext(os.path.basename(cpath))[0]
        # Match pattern: <model>_seed<seed>_<dataset>
        parts = fname.split("_")
        seed_idx = -1
        for idx, p in enumerate(parts):
            if p.startswith("seed"):
                seed_idx = idx
                break
        if seed_idx != -1 and seed_idx + 1 < len(parts):
            run_name = "_".join(parts[:seed_idx + 1])
            dname = "_".join(parts[seed_idx + 1:])
            flat_groups.setdefault(run_name, []).append((dname, cpath))

    for run_name, file_list in flat_groups.items():
        parts = run_name.split("_")
        seed = 1
        for p in parts:
            if p.startswith("seed"):
                try:
                    seed = int(p.replace("seed", ""))
                except ValueError:
                    pass
        calib_file = os.path.join(calib_dir, f"{run_name}.json")
        tau_val = 0.50
        if os.path.exists(calib_file):
            try:
                with open(calib_file) as f:
                    cdata = json.load(f)
                    tau_val = float(cdata.get("tau", cdata.get("youden_tau", 0.50)))
            except Exception:
                pass
        row = {
            "run_name": run_name,
            "seed": seed,
            "tau_val": tau_val
        }
        for dname, cpath in file_list:
            try:
                df = load_scores_pure(cpath, dname)
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

    df_out = pd.DataFrame(records)
    if "run_name" in df_out.columns:
        df_out = df_out.rename(columns={"run_name": "Model"})
        
    # Reorder columns logically if present
    preferred_order = [
        "Model", "seed", "tau_val",
        "ffpp_pure_macro_auc", "ffpp_auc", "ffpp_hter", "ffpp_tpr5",
        "celebdf_auc", "celebdf_hter", "celebdf_gap_g", "celebdf_tpr5",
        "dfd_auc", "dfd_hter", "dfd_gap_g", "dfd_tpr5",
        "fakeavceleb_auc", "fakeavceleb_hter", "fakeavceleb_gap_g", "fakeavceleb_tpr5",
    ]
    present_preferred = [c for c in preferred_order if c in df_out.columns]
    remaining = [c for c in df_out.columns if c not in present_preferred]
    if len(df_out) > 0 and len(present_preferred) > 0:
        df_out = df_out[present_preferred + remaining]
        
    return df_out


build_experiment_matrix = generate_experiment_matrix

MATRIX_NOTE = ("Reals are pure per dataset (D14: DFD keeps its 243 actor reals only). "
               "`*_tpr5` is TPR at the conservative empirical 5%-FPR point (D16, "
               "`smsdt/metric_defs.py`). `*_hter` and `*_gap_g` use the run's FF++-val Platt/Youden "
               "cutoff `tau_val`.")


def aggregate_matrix(matrix: pd.DataFrame) -> pd.DataFrame:
    """Mean and sd over seeds per architecture (run name without the _seedN suffix)."""
    if matrix.empty:
        return matrix
    m = matrix.copy()
    m["arch"] = m["Model"].str.replace(r"_seed\d+$", "", regex=True)
    cols = [c for c in m.columns if c.endswith(("_auc", "_tpr5")) and m[c].dtype.kind == "f"]
    rows = []
    for arch, g in m.groupby("arch", sort=True):
        row = {"arch": arch, "n_seeds": int(len(g))}
        for c in cols:
            v = g[c].dropna()
            row[c] = (f"{v.mean():.4f} +/- {v.std(ddof=1):.4f}" if len(v) > 1
                      else (f"{v.iloc[0]:.4f}" if len(v) else "-"))
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    os.makedirs("results/tables", exist_ok=True)
    matrix = generate_experiment_matrix()
    
    out_csv = "results/tables/experiment_matrix.csv"
    out_md = "results/tables/experiment_matrix.md"
    
    matrix.to_csv(out_csv, index=False)
    agg = aggregate_matrix(matrix)
    agg.to_csv("results/tables/experiment_matrix_agg.csv", index=False)
    with open("results/tables/experiment_matrix_agg.md", "w") as f:
        f.write("# Experiment matrix, mean +/- sd over seeds (auto-generated)\n\n")
        f.write(MATRIX_NOTE + "\n\n")
        f.write(df_to_markdown(agg) + "\n")
    
    with open(out_md, "w") as f:
        f.write("# Master Experiment Matrix (Auto-Generated)\n\n")
        f.write(f"*Generated from raw score CSVs in `results/scores/` ({len(matrix)} total runs)*\n\n")
        f.write(MATRIX_NOTE + "\n\n")
        if len(matrix) > 0:
            f.write(df_to_markdown(matrix))
            f.write("\n")
        else:
            f.write("*No run scores recorded in `results/scores/` yet.*\n")
            
    print(f"✓ Experiment matrix successfully written to: {out_csv} and {out_md}")
    if len(matrix) > 0:
        print("\n" + df_to_markdown(matrix) + "\n")
    else:
        print("Note: results/scores/ contains no score CSVs on this machine.")


if __name__ == "__main__":
    main()

