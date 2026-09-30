"""
Probabilistic & Spatial Verification Metrics for Phase 5.

Implements Continuous Ranked Probability Score (CRPS - Gaussian & Ensemble), CRPSS,
Prediction Interval Coverage Probability (PICP), Mean Prediction Interval Width (MPIW),
Sharpness, Spatial MAE/RMSE, and Spatial Weight Entropy.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import scipy.stats as stats


def crps_gaussian(
    obs: Union[float, np.ndarray],
    mu: Union[float, np.ndarray],
    sigma: Union[float, np.ndarray],
) -> np.ndarray:
    """
    Calculate Continuous Ranked Probability Score (CRPS) for a Gaussian predictive distribution N(mu, sigma^2).

    Formula:
        CRPS(N(mu, sigma^2), y) = sigma * [ z * (2*Phi(z) - 1) + 2*phi(z) - 1/sqrt(pi) ]
    where z = (y - mu) / sigma.

    Parameters
    ----------
    obs : float or np.ndarray of ground truth observations
    mu : float or np.ndarray of predictive mean
    sigma : float or np.ndarray of predictive standard deviation

    Returns
    -------
    crps : np.ndarray of CRPS values (same shape as inputs)
    """
    y = np.asarray(obs, dtype=np.float64)
    m = np.asarray(mu, dtype=np.float64)
    s = np.maximum(1e-6, np.asarray(sigma, dtype=np.float64))

    z = (y - m) / s
    phi = stats.norm.pdf(z)
    Phi = stats.norm.cdf(z)

    crps_vals = s * (z * (2.0 * Phi - 1.0) + 2.0 * phi - 1.0 / np.sqrt(np.pi))
    return np.asarray(crps_vals, dtype=np.float32)


def crps_ensemble(
    obs: Union[float, np.ndarray],
    ensemble_members: np.ndarray,
) -> np.ndarray:
    """
    Calculate empirical ensemble Continuous Ranked Probability Score (CRPS).

    Formula:
        CRPS(F_ens, y) = 1/M * sum_m |F_m - y| - 1/(2*M^2) * sum_m sum_k |F_m - F_k|

    Parameters
    ----------
    obs : np.ndarray of shape (N,) or scalar
    ensemble_members : np.ndarray of shape (N, M) or (M,) where M is number of members

    Returns
    -------
    crps : np.ndarray of CRPS values
    """
    y = np.asarray(obs, dtype=np.float64)
    E = np.asarray(ensemble_members, dtype=np.float64)

    if E.ndim == 1:
        E = E[np.newaxis, :]
        y = np.asarray([y])

    N, M = E.shape
    # Term 1: 1/M * sum_m |F_m - y|
    term1 = np.mean(np.abs(E - y[:, np.newaxis]), axis=1)

    # Term 2: 1/(2*M^2) * sum_m sum_k |F_m - F_k|
    diff_matrix = np.abs(E[:, :, np.newaxis] - E[:, np.newaxis, :])
    term2 = np.sum(diff_matrix, axis=(1, 2)) / (2.0 * M * M)

    return (term1 - term2).astype(np.float32)


def crps_skill_score(
    crps_model: float,
    crps_reference: float,
) -> float:
    """
    Calculate CRPS Skill Score (CRPSS) relative to a reference forecast (e.g. climatology or equal weight).
    CRPSS = 1 - (CRPS_model / CRPS_ref).
    Positive indicates improvement over reference.
    """
    if crps_reference <= 1e-6:
        return 0.0
    return float(1.0 - (crps_model / crps_reference))


def prediction_interval_coverage_probability(
    obs: np.ndarray,
    lower_bound: np.ndarray,
    upper_bound: np.ndarray,
) -> float:
    """
    Calculate Prediction Interval Coverage Probability (PICP).
    Fraction of ground truth observations falling within [lower_bound, upper_bound].
    """
    y = np.asarray(obs)
    low = np.asarray(lower_bound)
    high = np.asarray(upper_bound)
    valid = ~np.isnan(y) & ~np.isnan(low) & ~np.isnan(high)
    if not np.any(valid):
        return 0.0
    inside = (y[valid] >= low[valid]) & (y[valid] <= high[valid])
    return float(np.mean(inside))


def mean_prediction_interval_width(
    lower_bound: np.ndarray,
    upper_bound: np.ndarray,
) -> float:
    """
    Calculate Mean Prediction Interval Width (MPIW).
    MPIW = mean(upper_bound - lower_bound).
    """
    low = np.asarray(lower_bound)
    high = np.asarray(upper_bound)
    diff = high - low
    valid = ~np.isnan(diff)
    if not np.any(valid):
        return 0.0
    return float(np.mean(diff[valid]))


def sharpness(std_or_spread: np.ndarray) -> float:
    """
    Compute forecast sharpness (mean predictive standard deviation).
    Lower values indicate sharper, more confident predictive distributions.
    """
    s = np.asarray(std_or_spread)
    valid = ~np.isnan(s)
    if not np.any(valid):
        return 0.0
    return float(np.mean(s[valid]))


def spatial_mae(pred_2d: np.ndarray, obs_2d: np.ndarray) -> float:
    """Calculate Spatial Mean Absolute Error over 2D grid."""
    diff = np.abs(np.asarray(pred_2d) - np.asarray(obs_2d))
    return float(np.nanmean(diff))


def spatial_rmse(pred_2d: np.ndarray, obs_2d: np.ndarray) -> float:
    """Calculate Spatial Root Mean Squared Error over 2D grid."""
    diff_sq = (np.asarray(pred_2d) - np.asarray(obs_2d)) ** 2
    return float(np.sqrt(np.nanmean(diff_sq)))


def mean_absolute_weight_difference(w1: np.ndarray, w2: np.ndarray) -> float:
    """Calculate Mean Absolute Difference between two weight fields."""
    diff = np.abs(np.asarray(w1) - np.asarray(w2))
    return float(np.nanmean(diff))


def spatial_weight_entropy(weights: np.ndarray, epsilon: float = 1e-6) -> float:
    """
    Calculate spatial Shannon weight entropy: -sum_m W_m * log(W_m + eps).
    Higher entropy indicates more distributed weights across models; lower indicates model selection.
    """
    W = np.asarray(weights, dtype=np.float64)
    # W shape: (M, H, W) or (N, M, H, W)
    # Assume model axis is axis 0 or 1
    model_axis = 0 if W.ndim == 3 else 1
    entropy = -np.sum(W * np.log(np.maximum(epsilon, W)), axis=model_axis)
    return float(np.nanmean(entropy))


# Aliases for convenience
compute_crpss = crps_skill_score


def picp_mpiw(obs: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> Tuple[float, float]:
    """Calculate PICP and MPIW together."""
    return prediction_interval_coverage_probability(obs, lower, upper), mean_prediction_interval_width(lower, upper)


def evaluate_probabilistic_forecast(
    obs: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    quantiles_dict: Optional[Dict[float, np.ndarray]] = None,
) -> Dict[str, float]:
    """
    Comprehensive summary evaluation of probabilistic predictions against observations.
    """
    crps_vals = crps_gaussian(obs, mean, std)
    mean_crps = float(np.nanmean(crps_vals))
    sharp = sharpness(std)

    metrics = {
        "crps": round(mean_crps, 4),
        "sharpness": round(sharp, 4),
    }

    if quantiles_dict and 0.10 in quantiles_dict and 0.90 in quantiles_dict:
        picp, mpiw = picp_mpiw(obs, quantiles_dict[0.10], quantiles_dict[0.90])
        metrics["picp_80"] = round(picp, 4)
        metrics["mpiw_80"] = round(mpiw, 4)

    return metrics
