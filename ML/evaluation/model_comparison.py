"""
Statistical Model Comparison and Block Bootstrap Inference Engine.
Provides paired forecast error evaluation, block bootstrap confidence intervals,
and effective sample size estimation accounting for temporal/spatial autocorrelation.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd


def compute_effective_sample_size(series: np.ndarray) -> float:
    """
    Estimate effective sample size (N_eff) accounting for lag-1 serial autocorrelation.
    N_eff = N * (1 - rho_1) / (1 + rho_1)
    """
    arr = np.asarray(series, dtype=float)
    arr = arr[~np.isnan(arr)]
    N = len(arr)
    if N <= 2:
        return float(N)

    # Compute lag-1 autocorrelation
    arr_centered = arr - np.mean(arr)
    var = np.var(arr)
    if var < 1e-12:
        return float(N)

    rho1 = float(np.corrcoef(arr_centered[:-1], arr_centered[1:])[0, 1])
    if np.isnan(rho1) or rho1 < 0.0:
        return float(N)

    # Clamp rho1 to max 0.95 to avoid division by zero
    rho1 = min(rho1, 0.95)
    n_eff = N * ((1.0 - rho1) / (1.0 + rho1))
    return float(max(2.0, min(N, n_eff)))


def block_bootstrap_ci(
    diffs: np.ndarray,
    block_size: int = 4,
    n_bootstraps: int = 1000,
    confidence_level: float = 0.95,
    random_seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Calculate moving block bootstrap confidence interval for the mean paired error difference.

    Parameters
    ----------
    diffs : np.ndarray
        Array of paired differences: (e.g. error_model_a - error_model_b).
    block_size : int
        Temporal block length to preserve autocorrelation structure.
    n_bootstraps : int
        Number of bootstrap replications.
    confidence_level : float
        Two-sided confidence level (e.g. 0.95).

    Returns
    -------
    mean_diff : float
    ci_lower : float
    ci_upper : float
    """
    rng = np.random.default_rng(random_seed)
    arr = np.asarray(diffs, dtype=float)
    arr = arr[~np.isnan(arr)]
    N = len(arr)
    if N == 0:
        return 0.0, 0.0, 0.0
    if N < block_size:
        block_size = max(1, N // 2)

    mean_diff = float(np.mean(arr))
    if N < 5 or block_size < 1:
        return mean_diff, mean_diff, mean_diff

    # Construct all contiguous blocks
    num_blocks = N - block_size + 1
    blocks = [arr[i : i + block_size] for i in range(num_blocks)]
    needed_blocks = int(np.ceil(N / block_size))

    boot_means = np.empty(n_bootstraps, dtype=float)
    for b in range(n_bootstraps):
        selected_idx = rng.integers(0, num_blocks, size=needed_blocks)
        sample = np.concatenate([blocks[idx] for idx in selected_idx])[:N]
        boot_means[b] = np.mean(sample)

    alpha = 1.0 - confidence_level
    ci_lower = float(np.percentile(boot_means, 100.0 * (alpha / 2.0)))
    ci_upper = float(np.percentile(boot_means, 100.0 * (1.0 - alpha / 2.0)))

    return mean_diff, ci_lower, ci_upper


def evaluate_paired_model_comparison(
    df: pd.DataFrame,
    baseline_model: str = "equal_weight",
    candidate_model: str = "spatiotemporal_cnn",
    obs_col: str = "observed_value",
    pred_col: str = "forecast_value",
    model_col: str = "model_name",
    block_size: int = 4,
    n_bootstraps: int = 1000,
) -> Dict[str, Any]:
    """
    Execute rigorous paired statistical comparison between two models across matched verification samples.
    """
    base_df = df[df[model_col] == baseline_model]
    cand_df = df[df[model_col] == candidate_model]

    # Join on common keys
    join_keys = [c for c in ["timestamp", "latitude", "longitude", "variable", "lead_time"] if c in df.columns]
    merged = pd.merge(
        base_df,
        cand_df,
        on=join_keys,
        suffixes=("_base", "_cand"),
    )

    if merged.empty:
        return {
            "baseline_model": baseline_model,
            "candidate_model": candidate_model,
            "sample_size": 0,
            "effective_sample_size": 0.0,
            "mae_difference_mean": 0.0,
            "mae_diff_ci_lower": 0.0,
            "mae_diff_ci_upper": 0.0,
            "significant_at_95": False,
            "interpretation": "No overlapping verification records found.",
        }

    # Absolute errors
    obs = merged[f"{obs_col}_base"].values
    pred_base = merged[f"{pred_col}_base"].values
    pred_cand = merged[f"{pred_col}_cand"].values

    err_base = np.abs(pred_base - obs)
    err_cand = np.abs(pred_cand - obs)
    diffs = err_cand - err_base  # Negative means candidate is better (lower error)

    mean_diff, ci_low, ci_high = block_bootstrap_ci(
        diffs, block_size=block_size, n_bootstraps=n_bootstraps
    )
    n_eff = compute_effective_sample_size(diffs)

    # Check if CI excludes zero
    significant = (ci_high < 0.0) or (ci_low > 0.0)

    if ci_high < 0.0:
        interpretation = f"{candidate_model} achieves statistically lower MAE than {baseline_model} (95% CI: [{ci_low:.3f}, {ci_high:.3f}])."
    elif ci_low > 0.0:
        interpretation = f"{candidate_model} has statistically higher MAE than {baseline_model} (95% CI: [{ci_low:.3f}, {ci_high:.3f}])."
    else:
        interpretation = f"Difference between {candidate_model} and {baseline_model} is not statistically significant at 95% confidence level."

    return {
        "baseline_model": baseline_model,
        "candidate_model": candidate_model,
        "sample_size": len(diffs),
        "effective_sample_size": round(n_eff, 1),
        "mae_baseline": round(float(np.mean(err_base)), 4),
        "mae_candidate": round(float(np.mean(err_cand)), 4),
        "mae_difference_mean": round(mean_diff, 4),
        "median_difference": round(float(np.median(diffs)), 4),
        "mae_diff_ci_lower": round(ci_low, 4),
        "mae_diff_ci_upper": round(ci_high, 4),
        "significant_at_95": significant,
        "interpretation": interpretation,
    }
