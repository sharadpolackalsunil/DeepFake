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
    random_seed: int = 42
) -> pd.DataFrame:
    """Perform prior sweep experiment across multiple simulated operational priors."""
    rng = np.random.RandomState(random_seed)
    results = []
    
    for pi in priors:
        prior_errors = []
        pre_hters = []
        post_hters = []
        
        for rep in range(n_repeats):
            sub_df = resample_to_prior(df, target_prior=pi, rng=rng)
            y = sub_df["label"].values.astype(np.int32)
            p_source = sub_df["score_mean"].values.astype(np.float64)
            
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
            
        results.append({
            "target_prior_pi": pi,
            "mean_prior_error": float(np.mean(prior_errors)),
            "std_prior_error": float(np.std(prior_errors)),
            "mean_hter_frozen": float(np.mean(pre_hters)),
            "std_hter_frozen": float(np.std(pre_hters)),
            "mean_hter_em_corrected": float(np.mean(post_hters)),
            "std_hter_em_corrected": float(np.std(post_hters)),
            "hter_reduction": float(np.mean(pre_hters) - np.mean(post_hters))
        })
        
    return pd.DataFrame(results)


if __name__ == "__main__":
    # Smoke test on synthetic data
    rng = np.random.RandomState(42)
    syn_records = []
    for i in range(200):
        syn_records.append({"label": 0, "score_mean": float(rng.beta(2, 8))})
    for i in range(300):
        syn_records.append({"label": 1, "score_mean": float(rng.beta(8, 2))})
    test_df = pd.DataFrame(syn_records)
    
    sweep_df = run_prior_sweep(test_df, tau_val=0.50, priors=[0.1, 0.3, 0.5, 0.7, 0.9], n_repeats=20)
    print("=" * 80)
    print("PRIOR SWEEP SMOKE TEST RESULTS")
    print("=" * 80)
    print(sweep_df.to_string(index=False))
    print("=" * 80)
