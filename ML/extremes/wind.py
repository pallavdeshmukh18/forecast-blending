"""
Extreme wind speed and gale detection module.
"""

from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd

from configs import get_config


def classify_wind_speed(
    values: Union[np.ndarray, pd.Series, list],
    thresholds: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    """
    Classify 10m wind speeds into Beaufort-aligned categories:
    - light_moderate: value < strong_breeze threshold
    - strong_breeze: strong_breeze <= value < gale threshold
    - gale: gale <= value < storm threshold
    - storm: value >= storm threshold

    Parameters
    ----------
    values : array-like
        Wind speed values in m/s.
    thresholds : Optional[Dict[str, float]]
        Custom thresholds with keys 'strong_breeze', 'gale', 'storm'.

    Returns
    -------
    np.ndarray
        Array of category strings.
    """
    val_arr = np.asarray(values, dtype=float)
    cfg = get_config().extreme_event_thresholds.wind

    th_breeze = thresholds.get("strong_breeze", cfg.strong_breeze) if thresholds else cfg.strong_breeze
    th_gale = thresholds.get("gale", cfg.gale) if thresholds else cfg.gale
    th_storm = thresholds.get("storm", cfg.storm) if thresholds else cfg.storm

    conditions = [
        val_arr < th_breeze,
        (val_arr >= th_breeze) & (val_arr < th_gale),
        (val_arr >= th_gale) & (val_arr < th_storm),
        val_arr >= th_storm,
    ]
    labels = ["light_moderate", "strong_breeze", "gale", "storm"]

    return np.select(conditions, labels, default="light_moderate")


def detect_wind_extremes(
    df: pd.DataFrame,
    value_col: str = "forecast_value",
    threshold: Optional[float] = None,
) -> pd.DataFrame:
    """
    Filter DataFrame for high wind / gale events exceeding threshold.

    Parameters
    ----------
    df : pd.DataFrame
        Meteorological DataFrame.
    value_col : str
        Column name for wind speed.
    threshold : Optional[float]
        Wind speed threshold in m/s. Defaults to 'gale' threshold (17.2 m/s).

    Returns
    -------
    pd.DataFrame
        Filtered records exceeding the wind speed threshold.
    """
    th = threshold if threshold is not None else get_config().extreme_event_thresholds.wind.gale
    wind_mask = (df["variable"] == "wind_speed") & (df[value_col] >= th)
    return df[wind_mask].copy().reset_index(drop=True)
