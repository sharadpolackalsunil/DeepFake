"""Canonical metric definitions shared by smsdt/eval_cross.py and tools/ (numpy only).

D16 (docs/decisions.md): before this module the repository had two definitions of
"TPR at 5% FPR" that gave different numbers for the same score file:

* ``smsdt/eval_cross.py`` linearly interpolated the ROC curve at FPR = 0.05;
* ``tools/make_tables.py`` thresholded at ``np.percentile(reals, 95)`` (itself an
  interpolated quantile) and counted fakes ``>=`` that value.

Both can report a TPR whose achieved empirical FPR is above the target. The canonical
definition below is the conservative empirical one used in biometrics (ISO/IEC 19795):
choose the lowest threshold whose empirical FPR is <= target, flag scores strictly above it.
"""
from typing import Tuple

import numpy as np


def threshold_at_fpr(real_scores: np.ndarray, target_fpr: float) -> float:
    """Lowest threshold t with mean(real_scores > t) <= target_fpr.

    With n reals, at most k = floor(target_fpr * n) reals may lie strictly above t, so t is
    the (k+1)-th largest real score. Returns +inf when there are no reals.
    """
    r = np.sort(np.asarray(real_scores, dtype=np.float64))[::-1]
    n = len(r)
    if n == 0:
        return float("inf")
    k = int(np.floor(target_fpr * n + 1e-12))
    if k >= n:
        return -float("inf")
    return float(r[k])


def tpr_at_fpr(y: np.ndarray, s: np.ndarray, target_fpr: float) -> float:
    """TPR at the conservative empirical operating point with FPR <= target_fpr."""
    tpr, _ = tpr_fpr_at_fpr(y, s, target_fpr)
    return tpr


def tpr_fpr_at_fpr(y: np.ndarray, s: np.ndarray, target_fpr: float) -> Tuple[float, float]:
    """(TPR, achieved FPR) at the conservative operating point. NaN if a class is missing."""
    y = np.asarray(y).astype(int)
    s = np.asarray(s, dtype=np.float64)
    reals, fakes = s[y == 0], s[y == 1]
    if len(reals) == 0 or len(fakes) == 0:
        return float("nan"), float("nan")
    t = threshold_at_fpr(reals, target_fpr)
    return float(np.mean(fakes > t)), float(np.mean(reals > t))
