"""
Extreme rainfall analysis and threshold categorization module.
"""

from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd

from configs import AppConfig, get_config


def classify_rainfall_intensity(
    values: Union[np.ndarray, pd.Series, list],
    thresholds: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    """
    Classify rainfall values into meteorological intensity categories:
    - none: 0.0 mm
    - light: 0.0 < value <= moderate threshold
    - moderate: moderate < value <= heavy threshold
    - heavy: heavy < value <= very_heavy threshold
    - very_heavy: very_heavy < value <= extreme threshold
    - extreme: value > extreme threshold

    Parameters
    ----------
    values : array-like
        Rainfall precipitation values in mm.
    thresholds : Optional[Dict[str, float]]
        Custom threshold dictionary with keys 'moderate', 'heavy', 'very_heavy', 'extreme'.

    Returns
    -------
    np.ndarray
        Array of category strings.
    """
    val_arr = np.asarray(values, dtype=float)
    cfg = get_config().extreme_event_thresholds.rainfall

    th_mod = thresholds.get("moderate", cfg.moderate) if thresholds else cfg.moderate
    th_heavy = thresholds.get("heavy", cfg.heavy) if thresholds else cfg.heavy
    th_vheavy = thresholds.get("very_heavy", cfg.very_heavy) if thresholds else cfg.very_heavy
    th_extreme = thresholds.get("extreme", cfg.extreme) if thresholds else cfg.extreme

    conditions = [
        val_arr <= 0.1,
        (val_arr > 0.1) & (val_arr <= th_mod),
        (val_arr > th_mod) & (val_arr <= th_heavy),
        (val_arr > th_heavy) & (val_arr <= th_vheavy),
        (val_arr > th_vheavy) & (val_arr <= th_extreme),
        val_arr > th_extreme,
    ]
    labels = ["none", "light", "moderate", "heavy", "very_heavy", "extreme"]

    return np.select(conditions, labels, default="unknown")


def detect_extreme_rainfall(
    df: pd.DataFrame,
    value_col: str = "forecast_value",
    threshold: Optional[float] = None,
) -> pd.DataFrame:
    """
    Filter DataFrame for extreme rainfall events exceeding threshold.

    Parameters
    ----------
    df : pd.DataFrame
        Meteorological DataFrame containing rainfall records.
    value_col : str
        Column name for rainfall values.
    threshold : Optional[float]
        Rainfall threshold in mm. Defaults to 'heavy' threshold (35.5 mm).

    Returns
    -------
    pd.DataFrame
        Filtered records representing extreme rainfall events.
    """
    th = threshold if threshold is not None else get_config().extreme_event_thresholds.rainfall.heavy
    rain_mask = (df["variable"] == "rainfall") & (df[value_col] >= th)
    return df[rain_mask].copy().reset_index(drop=True)
