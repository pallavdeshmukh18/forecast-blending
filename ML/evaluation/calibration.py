"""
Probability Calibration and Reliability Analysis Module for Phase 5.

Implements Brier Score, Brier Skill Score, Reliability Curve generation,
Sharpness analysis, and event probability calibration evaluation.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from configs import AppConfig, get_config


def brier_score(
    obs_events: np.ndarray,
    forecast_probs: np.ndarray,
) -> float:
    """
    Compute Brier Score (BS):
        BS = 1/N * sum_i (p_i - o_i)^2
    where o_i in {0, 1} is binary event indicator and p_i in [0, 1] is forecast probability.
    """
    o = np.asarray(obs_events, dtype=np.float64)
    p = np.clip(np.asarray(forecast_probs, dtype=np.float64), 0.0, 1.0)
    valid = ~np.isnan(o) & ~np.isnan(p)
    if not np.any(valid):
        return 0.0
    return float(np.mean((p[valid] - o[valid]) ** 2))


def brier_skill_score(
    bs_model: float,
    bs_reference: float,
) -> float:
    """
    Compute Brier Skill Score (BSS) relative to reference (e.g. climatology):
        BSS = 1 - (BS_model / BS_ref)
    """
    if bs_reference <= 1e-6:
        return 0.0
    return float(1.0 - (bs_model / bs_reference))


def compute_reliability_curve(
    obs_events: np.ndarray,
    forecast_probs: np.ndarray,
    n_bins: int = 10,
    min_samples_per_bin: int = 5,
) -> Dict[str, Any]:
    """
    Compute reliability diagram data and calibration metrics.

    Parameters
    ----------
    obs_events : np.ndarray of binary ground-truth events (0 or 1)
    forecast_probs : np.ndarray of predicted probabilities in [0, 1]
    n_bins : number of probability bins (e.g. 10 for [0, 0.1), ..., [0.9, 1.0])
    min_samples_per_bin : minimum number of observations required for valid bin metrics

    Returns
    -------
    dict with:
        - bin_edges : list of bin boundaries
        - bin_centers : list of bin midpoints
        - mean_predicted_probs : list of average forecast probability per bin
        - observed_frequencies : list of observed event relative frequency per bin
        - bin_counts : list of sample counts per bin
        - bin_weights : list of bin sample proportions
        - reliability_error : weighted mean squared difference sum(w_k * (p_k - o_k)^2)
        - sharpness_histogram : distribution of predicted probabilities
    """
    o = np.asarray(obs_events, dtype=np.float64).ravel()
    p = np.clip(np.asarray(forecast_probs, dtype=np.float64).ravel(), 0.0, 1.0)

    valid = ~np.isnan(o) & ~np.isnan(p)
    o = o[valid]
    p = p[valid]
    N = len(o)

    if N == 0:
        return {
            "bin_edges": [],
            "bin_centers": [],
            "mean_predicted_probs": [],
            "observed_frequencies": [],
            "bin_counts": [],
            "bin_weights": [],
            "reliability_error": 0.0,
            "brier_score": 0.0,
            "sample_count": 0,
        }

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    mean_preds = []
    obs_freqs = []
    counts = []
    weights = []
    rel_error = 0.0

    for i in range(n_bins):
        low = bin_edges[i]
        high = bin_edges[i + 1]

        if i == n_bins - 1:
            in_bin = (p >= low) & (p <= high)
        else:
            in_bin = (p >= low) & (p < high)

        count = int(np.sum(in_bin))
        counts.append(count)
        weight = float(count / N)
        weights.append(weight)

        if count >= min_samples_per_bin:
            mean_p = float(np.mean(p[in_bin]))
            obs_f = float(np.mean(o[in_bin]))
            mean_preds.append(mean_p)
            obs_freqs.append(obs_f)
            rel_error += weight * ((mean_p - obs_f) ** 2)
        elif count > 0:
            mean_p = float(np.mean(p[in_bin]))
            obs_f = float(np.mean(o[in_bin]))
            mean_preds.append(mean_p)
            obs_freqs.append(obs_f)
        else:
            mean_preds.append(float(bin_centers[i]))
            obs_freqs.append(np.nan)

    bs = brier_score(o, p)
    climatology = float(np.mean(o))
    bs_ref = brier_score(o, np.full_like(p, climatology))
    bss = brier_skill_score(bs, bs_ref)

    return {
        "bin_edges": [round(float(x), 3) for x in bin_edges],
        "bin_centers": [round(float(x), 3) for x in bin_centers],
        "mean_predicted_probs": [round(float(x), 4) if not np.isnan(x) else None for x in mean_preds],
        "observed_frequencies": [round(float(x), 4) if not np.isnan(x) else None for x in obs_freqs],
        "bin_counts": counts,
        "bin_weights": [round(float(w), 4) for w in weights],
        "reliability_error": round(float(rel_error), 6),
        "brier_score": round(float(bs), 6),
        "brier_skill_score": round(float(bss), 4),
        "sample_count": N,
        "event_frequency": round(climatology, 4),
    }


def evaluate_event_calibration(
    obs_values: np.ndarray,
    forecast_probs_or_dist: Any,
    threshold: float,
    operator: str = ">=",
) -> Dict[str, Any]:
    """
    Evaluate binary event probability calibration against ground truth observations.

    Parameters
    ----------
    obs_values : np.ndarray of ground truth values (e.g. rainfall mm or wind m/s)
    forecast_probs_or_dist : np.ndarray of probabilities or ProbabilisticForecast object
    threshold : event threshold (e.g. 25.0 mm)
    operator : '>=' or '<='
    """
    y = np.asarray(obs_values)
    if operator in [">=", ">"]:
        obs_events = (y >= threshold).astype(np.float32)
    else:
        obs_events = (y <= threshold).astype(np.float32)

    # Extract predicted probability
    if hasattr(forecast_probs_or_dist, "prob_exceeds"):
        if operator in [">=", ">"]:
            probs = forecast_probs_or_dist.prob_exceeds(threshold)
        else:
            probs = forecast_probs_or_dist.cdf(threshold)
    else:
        probs = np.asarray(forecast_probs_or_dist, dtype=np.float32)

    diag = compute_reliability_curve(obs_events, probs)
    diag["threshold"] = threshold
    diag["operator"] = operator
    return diag


# Alias
reliability_diagram = compute_reliability_curve


def probability_calibration_error(obs_events: np.ndarray, forecast_probs: np.ndarray) -> float:
    """Compute reliability / calibration error."""
    diag = compute_reliability_curve(obs_events, forecast_probs)
    return float(diag["reliability_error"])
