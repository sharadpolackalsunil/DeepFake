"""Shared helpers for frozen calibration (Platt + Youden) used by the audit tools.

`smsdt/eval_cross.py --fit-calib` writes a JSON with keys a, b, tau where
    p_cal = sigmoid(a * logit_mean + b)
and tau is the Youden threshold on p_cal (fitted on the dev split only).

tau is therefore on the CALIBRATED probability scale. Comparing it with the raw
`score_mean` column (an uncalibrated probability) is a scale error. These helpers
apply the frozen calibration correctly.
"""
import json
import math
import os
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


def load_calib(run_name: str, calib_dir: str = "results/calib") -> Optional[Dict]:
    """Load results/calib/{run_name}.json (e.g. run_name = 'smsdt_s_seed1').

    Accepts 'tau' or 'youden_tau'. Returns None if the file does not exist.
    """
    path = os.path.join(calib_dir, f"{run_name}.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if "tau" not in data and "youden_tau" in data:
        data["tau"] = data["youden_tau"]
    for key in ("a", "b", "tau"):
        if key not in data:
            raise ValueError(f"Calibration file {path} is missing '{key}'.")
    data["_path"] = path
    return data


def calibrated_probs(df: pd.DataFrame, calib: Dict, logit_col: str = "logit_mean") -> np.ndarray:
    """Platt-calibrated probability sigmoid(a * logit + b) for every row."""
    u = df[logit_col].values.astype(np.float64)
    z = float(calib["a"]) * u + float(calib["b"])
    return 1.0 / (1.0 + np.exp(-z))


def frozen_logit_threshold(calib: Dict) -> float:
    """Threshold on the raw logit equivalent to p_cal >= tau.

    p_cal >= tau  <=>  a*u + b >= logit(tau)  <=>  u >= (logit(tau) - b) / a   (a > 0)
    """
    a, b, tau = float(calib["a"]), float(calib["b"]), float(calib["tau"])
    if a <= 0:
        raise ValueError(f"Platt slope a={a} <= 0: scores are anti-correlated with labels on dev.")
    tau = min(max(tau, 1e-12), 1 - 1e-12)
    return (math.log(tau / (1 - tau)) - b) / a


def resolve_decision_scores(
    df: pd.DataFrame,
    calib: Optional[Dict],
    allow_raw: bool = False,
) -> Tuple[np.ndarray, float, str]:
    """Return (scores, threshold, source) such that `scores >= threshold` is the frozen
    decision rule.

    With a calibration file: scores = logit_mean, threshold = frozen logit threshold.
    Without one: raises unless allow_raw, in which case scores = score_mean and
    threshold = 0.5 (labelled 'raw_0.5' so tables cannot pass it off as tau_val).
    """
    if calib is not None:
        if "logit_mean" not in df.columns:
            raise ValueError("Calibration needs the logit_mean column in the score file.")
        return df["logit_mean"].values.astype(np.float64), frozen_logit_threshold(calib), "calib"
    if not allow_raw:
        raise FileNotFoundError(
            "No calibration file for this run. A validation-frozen threshold cannot be "
            "evaluated without it. Fit it with `smsdt.eval_cross --split val --fit-calib "
            "results/calib/<run>.json`, or pass --allow-raw-threshold to audit the raw 0.5 "
            "threshold explicitly (it will be labelled raw_0.5, not tau_val)."
        )
    return df["score_mean"].values.astype(np.float64), 0.5, "raw_0.5"


def wilson_interval(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score interval for a binomial proportion k/n."""
    if n <= 0:
        return float("nan"), float("nan")
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)
