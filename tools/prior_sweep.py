"""Prior-shift sweep and Saerens-style Expectation-Maximization prior correction.

Evaluates detector behavior across simulated operational class priors pi in {0.1, 0.3, 0.5, 0.7, 0.9}
by subsampling test datasets (200 repeats per condition).
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional


def saerens_em_prior(
    posteriors: np.ndarray,
    pi_source: float = 0.50,
    max_iter: int = 100,
    tol: float = 1e-6
) -> Tuple[float, np.ndarray]:
    """Saerens-Latinne-Decaestecker (2002) EM algorithm for target class prior estimation.
    
    Args:
        posteriors: Array of source-calibrated probabilities p_i = P(Y=1|x) under pi_source
        pi_source: Source training class prior (0.50 for class-balanced Platt)
        max_iter: Maximum EM iterations
        tol: Convergence tolerance
        
    Returns:
        pi_hat: Estimated target class prior in [0, 1]
        adjusted_posteriors: Calibrated probabilities adjusted for prior shift
    """
    p = np.clip(posteriors.astype(np.float64), 1e-7, 1.0 - 1e-7)
    n = len(p)
    pi_target = 0.50  # Start neutral
    
    for _ in range(max_iter):
        ratio_pos = pi_target / pi_source
        ratio_neg = (1.0 - pi_target) / (1.0 - pi_source)
        
        # E-step: calculate posteriors under current target prior estimate
        num = ratio_pos * p
        denom = num + ratio_neg * (1.0 - p)
        q = num / np.clip(denom, 1e-12, None)
        
        # M-step: update target prior
        new_pi = float(np.mean(q))
        if abs(new_pi - pi_target) < tol:
            pi_target = new_pi
            break
        pi_target = new_pi
        
    # Compute final adjusted posteriors
    ratio_pos = pi_target / pi_source
    ratio_neg = (1.0 - pi_target) / (1.0 - pi_source)
    final_posteriors = (ratio_pos * p) / (ratio_pos * p + ratio_neg * (1.0 - p))
    return float(pi_target), final_posteriors


def resample_to_prior(
    df: pd.DataFrame,
    target_prior: float,
    rng: np.random.RandomState,
    fixed_total: Optional[int] = None
) -> pd.DataFrame:
    """Subsample majority class to achieve target fake class prior pi in [0, 1]."""
    reals = df[df["label"] == 0]
    fakes = df[df["label"] == 1]
    n_reals = len(reals)
    n_fakes = len(fakes)
    
    # Calculate required counts
    if target_prior >= 0.5:
        # Keep all reals, subsample fakes or vice versa
        # pi = n_f / (n_r + n_f) -> n_r = n_f * (1 - pi) / pi
        n_f_target = n_fakes
        n_r_target = int(np.round(n_f_target * (1.0 - target_prior) / target_prior))
        if n_r_target > n_reals:
            n_r_target = n_reals
            n_f_target = int(np.round(n_r_target * target_prior / (1.0 - target_prior)))
    else:
        n_r_target = n_reals
        n_f_target = int(np.round(n_r_target * target_prior / (1.0 - target_prior)))
        if n_f_target > n_fakes:
            n_f_target = n_fakes
            n_r_target = int(np.round(n_f_target * (1.0 - target_prior) / target_prior))
            
    n_r_target = max(10, min(n_r_target, n_reals))
    n_f_target = max(10, min(n_f_target, n_fakes))
    
    chosen_reals = reals.iloc[rng.choice(n_reals, size=n_r_target, replace=False)]
    chosen_fakes = fakes.iloc[rng.choice(n_fakes, size=n_f_target, replace=False)]
    
    resampled = pd.concat([chosen_reals, chosen_fakes], ignore_index=True)
    return resampled


def run_prior_sweep(
    df: pd.DataFrame,
    tau_val: float = 0.50,
    priors: List[float] = [0.1, 0.3, 0.5, 0.7, 0.9],
    n_repeats: int = 200,
    random_seed: int = 42,
    prob_col: str = "score_mean",
) -> pd.DataFrame:
    """Perform prior sweep experiment across multiple simulated operational priors.

    `prob_col` must hold posteriors calibrated under a balanced source prior (class-
    balanced Platt), because that is what Saerens EM assumes. Pass a calibrated column.

    Note on metrics: HTER = (FPR + FNR) / 2 does not depend on the class prior, and EM
    only re-weights posteriors monotonically, so EM cannot be expected to lower HTER at a
    fixed threshold. What EM targets is the Bayes decision under the new prior, i.e.
    accuracy (or expected cost). Both are reported: `acc_frozen` thresholds the source
    posterior at tau_val; `acc_em_bayes` thresholds the EM-adjusted posterior at 0.5.
    """
    rng = np.random.RandomState(random_seed)
    results = []
    
    for pi in priors:
        prior_errors = []
        pre_hters = []
        post_hters = []
        acc_pre = []
        acc_post = []
        
        for rep in range(n_repeats):
            sub_df = resample_to_prior(df, target_prior=pi, rng=rng)
            y = sub_df["label"].values.astype(np.int32)
            p_source = sub_df[prob_col].values.astype(np.float64)
            
            # Pre-correction HTER at frozen threshold
            fpr_pre = float(np.mean(p_source[y == 0] >= tau_val))
            fnr_pre = float(np.mean(p_source[y == 1] < tau_val))
            hter_pre = 0.5 * (fpr_pre + fnr_pre)
            
            # Saerens EM prior correction
            pi_hat, p_adjusted = saerens_em_prior(p_source, pi_source=0.50)
            
            # Post-correction HTER at frozen threshold
            fpr_post = float(np.mean(p_adjusted[y == 0] >= tau_val))
            fnr_post = float(np.mean(p_adjusted[y == 1] < tau_val))
            hter_post = 0.5 * (fpr_post + fnr_post)
            
            prior_errors.append(abs(pi_hat - pi))
            pre_hters.append(hter_pre)
            post_hters.append(hter_post)
            acc_pre.append(float(np.mean((p_source >= tau_val).astype(int) == y)))
            acc_post.append(float(np.mean((p_adjusted >= 0.5).astype(int) == y)))
            
        results.append({
            "target_prior_pi": pi,
            "mean_prior_error": float(np.mean(prior_errors)),
            "std_prior_error": float(np.std(prior_errors)),
            "mean_hter_frozen": float(np.mean(pre_hters)),
            "std_hter_frozen": float(np.std(pre_hters)),
            "mean_hter_em_corrected": float(np.mean(post_hters)),
            "std_hter_em_corrected": float(np.std(post_hters)),
            "hter_reduction": float(np.mean(pre_hters) - np.mean(post_hters)),
            "acc_frozen": float(np.mean(acc_pre)),
            "acc_em_bayes": float(np.mean(acc_post)),
            "acc_gain": float(np.mean(acc_post) - np.mean(acc_pre)),
        })
        
    return pd.DataFrame(results)


def main():
    import glob
    import argparse
    from tools.make_tables import df_to_markdown

    parser = argparse.ArgumentParser(description="Prior Shift & Saerens EM Prior Correction Audit (H3)")
    parser.add_argument("--scores", type=str, default=None, help="Specific score CSV path to sweep")
    parser.add_argument("--scores-dir", type=str, default="results/scores", help="Scores directory")
    parser.add_argument("--n-repeats", type=int, default=50, help="Number of subsampling repeats per prior")
    parser.add_argument("--tau-val", type=float, default=0.50, help="Frozen decision threshold (raw mode only)")
    parser.add_argument("--calib-dir", type=str, default="results/calib",
                        help="Use Platt-calibrated posteriors and the frozen tau from this directory")
    parser.add_argument("--allow-raw", action="store_true",
                        help="Allow runs without calibration (raw score_mean treated as a posterior; labelled raw)")
    args = parser.parse_args()
    from tools.calib_utils import load_calib, calibrated_probs

    target_files = []
    if args.scores:
        if os.path.exists(args.scores):
            target_files.append(args.scores)
        else:
            print(f"File not found: {args.scores}")
            return
    else:
        # Prioritize Celeb-DF cross-dataset evaluations
        celeb_files = glob.glob(os.path.join(args.scores_dir, "*celebdf.csv"))
        target_files = celeb_files if celeb_files else glob.glob(os.path.join(args.scores_dir, "*.csv"))

    if not target_files:
        print(f"No score CSVs found to audit in {args.scores_dir}.")
        return

    all_dfs = []
    print("=" * 90)
    print("HYPOTHESIS H3: PRIOR-SHIFT SWEEP & SAERENS EM CORRECTION AUDIT")
    print("=" * 90)

    for sf in sorted(target_files):
        model_name = os.path.splitext(os.path.basename(sf))[0]
        try:
            df = pd.read_csv(sf)
            if "label" not in df.columns or "score_mean" not in df.columns:
                continue
            run_name = model_name.rsplit("_", 1)[0] if "_seed" in model_name else model_name
            calib = load_calib(run_name, args.calib_dir)
            if calib is not None:
                df = df.copy()
                df["p_cal"] = calibrated_probs(df, calib)
                prob_col, tau, mode = "p_cal", float(calib["tau"]), "calibrated"
            elif args.allow_raw:
                prob_col, tau, mode = "score_mean", args.tau_val, "raw"
            else:
                print(f"Skipping {model_name}: no calibration file for {run_name} (use --allow-raw to force).")
                continue
            sweep_df = run_prior_sweep(df, tau_val=tau, priors=[0.1, 0.3, 0.5, 0.7, 0.9],
                                       n_repeats=args.n_repeats, prob_col=prob_col)
            sweep_df.insert(0, "posteriors", mode)
            sweep_df.insert(0, "model", model_name)
            all_dfs.append(sweep_df)
            print(f"\n--- Model: {model_name} (Target Prior Sweep, N_repeats={args.n_repeats}) ---")
            print(df_to_markdown(sweep_df))
        except Exception as e:
            print(f"Error sweeping {sf}: {e}")

    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        os.makedirs("results/tables", exist_ok=True)
        combined_df.to_csv("results/tables/prior_shift_sweep.csv", index=False)
        with open("results/tables/prior_shift_sweep.md", "w") as f:
            f.write("# Hypothesis H3: Operational Prior Shift Sweep & Saerens EM Correction\n\n")
            f.write(df_to_markdown(combined_df) + "\n")
        print("\n" + "=" * 90)
        print("✓ Prior shift audit successfully saved to results/tables/prior_shift_sweep.csv and .md\n")


if __name__ == "__main__":
    main()
