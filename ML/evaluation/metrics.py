"""
Standard meteorological verification metrics: MAE, RMSE, Bias, and Pearson Correlation.
"""

from typing import Dict, Union
import numpy as np
import pandas as pd


def _clean_arrays(
    obs: Union[np.ndarray, pd.Series, list],
    pred: Union[np.ndarray, pd.Series, list],
) -> tuple[np.ndarray, np.ndarray]:
    """Ensure inputs are 1D float arrays with matching length and filtered finite values."""
    y_true = np.asarray(obs, dtype=float).ravel()
    y_pred = np.asarray(pred, dtype=float).ravel()

    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Observed and predicted arrays must have equal length, got {len(y_true)} vs {len(y_pred)}"
        )

    valid_mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true_clean = y_true[valid_mask]
    y_pred_clean = y_pred[valid_mask]

    if len(y_true_clean) == 0:
        raise ValueError("No valid finite observation/prediction pairs remaining for metric calculation.")

    return y_true_clean, y_pred_clean


def mean_absolute_error(
    obs: Union[np.ndarray, pd.Series, list],
    pred: Union[np.ndarray, pd.Series, list],
) -> float:
    """
    Calculate Mean Absolute Error (MAE):

    .. math::
        \\text{MAE} = \\frac{1}{M} \\sum_{i=1}^M |\\hat{y}_i - y_i|

    Parameters
    ----------
    obs : array-like
        Ground-truth observations.
    pred : array-like
        Forecast predictions.

    Returns
    -------
    float
        MAE score.
    """
    y_true, y_pred = _clean_arrays(obs, pred)
    return float(np.mean(np.abs(y_pred - y_true)))


def root_mean_squared_error(
    obs: Union[np.ndarray, pd.Series, list],
    pred: Union[np.ndarray, pd.Series, list],
) -> float:
    """
    Calculate Root Mean Squared Error (RMSE):

    .. math::
        \\text{RMSE} = \\sqrt{\\frac{1}{M} \\sum_{i=1}^M (\\hat{y}_i - y_i)^2}

    Parameters
    ----------
    obs : array-like
        Ground-truth observations.
    pred : array-like
        Forecast predictions.

    Returns
    -------
    float
        RMSE score.
    """
    y_true, y_pred = _clean_arrays(obs, pred)
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def mean_bias_error(
    obs: Union[np.ndarray, pd.Series, list],
    pred: Union[np.ndarray, pd.Series, list],
) -> float:
    """
    Calculate Mean Bias Error (MBE / Bias):

    .. math::
        \\text{Bias} = \\frac{1}{M} \\sum_{i=1}^M (\\hat{y}_i - y_i)

    Positive value indicates overforecasting; negative indicates underforecasting.

    Parameters
    ----------
    obs : array-like
        Ground-truth observations.
    pred : array-like
        Forecast predictions.

    Returns
    -------
    float
        Mean bias error.
    """
    y_true, y_pred = _clean_arrays(obs, pred)
    return float(np.mean(y_pred - y_true))


def pearson_correlation(
    obs: Union[np.ndarray, pd.Series, list],
    pred: Union[np.ndarray, pd.Series, list],
) -> float:
    """
    Calculate Pearson Correlation Coefficient (r):

    .. math::
        r = \\frac{\\sum (y_i - \\bar{y})(\\hat{y}_i - \\bar{\\hat{y}})}{\\sqrt{\\sum (y_i - \\bar{y})^2 \\sum (\\hat{y}_i - \\bar{\\hat{y}})^2}}

    Handles zero-variance edge cases safely by returning 0.0 or 1.0 if identical.

    Parameters
    ----------
    obs : array-like
        Ground-truth observations.
    pred : array-like
        Forecast predictions.

    Returns
    -------
    float
        Pearson correlation coefficient between -1.0 and 1.0.
    """
    y_true, y_pred = _clean_arrays(obs, pred)
    if len(y_true) < 2:
        return 1.0 if np.allclose(y_true, y_pred) else 0.0

    std_true = np.std(y_true)
    std_pred = np.std(y_pred)

    if std_true < 1e-12 or std_pred < 1e-12:
        # If both are constant and identical, correlation is effectively 1.0, otherwise 0.0
        return 1.0 if np.allclose(y_true, y_pred) else 0.0

    corr = np.corrcoef(y_true, y_pred)[0, 1]
    return float(np.clip(corr, -1.0, 1.0))


def compute_all_metrics(
    obs: Union[np.ndarray, pd.Series, list],
    pred: Union[np.ndarray, pd.Series, list],
) -> Dict[str, float]:
    """
    Compute full suite of standard evaluation metrics.

    Returns
    -------
    Dict[str, float]
        Dictionary with keys: 'mae', 'rmse', 'bias', 'correlation', 'sample_count'.
    """
    y_true, y_pred = _clean_arrays(obs, pred)
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": root_mean_squared_error(y_true, y_pred),
        "bias": mean_bias_error(y_true, y_pred),
        "correlation": pearson_correlation(y_true, y_pred),
        "sample_count": len(y_true),
    }
