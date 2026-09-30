"""
Heatwave and extreme temperature detection module.
"""

from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd

from configs import get_config


def classify_heatwave(
    values: Union[np.ndarray, pd.Series, list],
    thresholds: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    """
    Classify 2m temperature values into heatwave categories:
    - normal: value < mild threshold
    - mild_heatwave: mild <= value < severe threshold
    - severe_heatwave: value >= severe threshold

    Parameters
    ----------
    values : array-like
        Temperature values in degrees Celsius.
    thresholds : Optional[Dict[str, float]]
        Custom thresholds with keys 'mild' and 'severe'.

    Returns
    -------
    np.ndarray
        Array of category strings.
    """
    val_arr = np.asarray(values, dtype=float)
    cfg = get_config().extreme_event_thresholds.heatwave

    th_mild = thresholds.get("mild", cfg.mild) if thresholds else cfg.mild
    th_severe = thresholds.get("severe", cfg.severe) if thresholds else cfg.severe

    conditions = [
        val_arr < th_mild,
        (val_arr >= th_mild) & (val_arr < th_severe),
        val_arr >= th_severe,
    ]
    labels = ["normal", "mild_heatwave", "severe_heatwave"]

    return np.select(conditions, labels, default="normal")


def detect_heatwave_events(
    df: pd.DataFrame,
    value_col: str = "forecast_value",
    threshold: Optional[float] = None,
) -> pd.DataFrame:
    """
    Filter DataFrame for heatwave temperature events.

    Parameters
    ----------
    df : pd.DataFrame
        Meteorological DataFrame.
    value_col : str
        Column name for temperature values.
    threshold : Optional[float]
        Temperature threshold in Celsius. Defaults to 'mild' threshold (40.0 C).

    Returns
    -------
    pd.DataFrame
        Filtered records exceeding the heatwave threshold.
    """
    th = threshold if threshold is not None else get_config().extreme_event_thresholds.heatwave.mild
    temp_mask = (df["variable"] == "temperature") & (df[value_col] >= th)
    return df[temp_mask].copy().reset_index(drop=True)
