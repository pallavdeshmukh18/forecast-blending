"""
Feature engineering utilities for meteorological time series and blending models.
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from configs import AppConfig, get_config


def add_temporal_features(df: pd.DataFrame, time_col: str = "timestamp") -> pd.DataFrame:
    """
    Extract diurnal, cyclical, and seasonal calendar features from a timestamp column.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    time_col : str, default 'timestamp'
        Name of datetime column.

    Returns
    -------
    pd.DataFrame
        DataFrame with added temporal features: hour, month, day_of_year, diurnal_sin, diurnal_cos, seasonal_sin, seasonal_cos.
    """
    res = df.copy()
    ts = pd.to_datetime(res[time_col], utc=True)

    res["hour"] = ts.dt.hour
    res["month"] = ts.dt.month
    res["day_of_year"] = ts.dt.dayofyear

    # Cyclical encodings
    res["diurnal_sin"] = np.sin(2 * np.pi * res["hour"] / 24.0)
    res["diurnal_cos"] = np.cos(2 * np.pi * res["hour"] / 24.0)

    res["seasonal_sin"] = np.sin(2 * np.pi * res["day_of_year"] / 365.25)
    res["seasonal_cos"] = np.cos(2 * np.pi * res["day_of_year"] / 365.25)

    return res


def add_lead_time_bins(
    df: pd.DataFrame,
    lead_time_col: str = "lead_time",
    bins_config: Optional[Dict[str, List[int]]] = None,
) -> pd.DataFrame:
    """
    Assign lead time intervals to standardized operational bins without overlapping boundaries.

    Default Bins:
    - short: [0, 24] hours (0 <= lead <= 24)
    - medium: (24, 72] hours (24 < lead <= 72)
    - extended: (72, 120] hours (72 < lead <= 120)
    - long_range: > 120 hours (120 < lead <= 1000)

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    lead_time_col : str, default 'lead_time'
        Lead time column name.
    bins_config : Optional[Dict[str, List[int]]]
        Dictionary specifying bin boundaries.

    Returns
    -------
    pd.DataFrame
        DataFrame with 'lead_time_bin' column.
    """
    res = df.copy()
    if lead_time_col not in res.columns:
        return res

    leads = res[lead_time_col].astype(float)

    if bins_config is not None:
        conditions = []
        choices = []
        for bin_name, (low, high) in bins_config.items():
            if low == 0 or not conditions:
                cond = (leads >= low) & (leads <= high)
            else:
                cond = (leads > low) & (leads <= high)
            conditions.append(cond)
            choices.append(bin_name)

        res["lead_time_bin"] = np.select(conditions, choices, default="long_range")
    else:
        # Standard default assignment
        conditions = [
            (leads >= 0) & (leads <= 24),
            (leads > 24) & (leads <= 72),
            (leads > 72) & (leads <= 120),
            (leads > 120),
        ]
        choices = ["short", "medium", "extended", "long_range"]
        res["lead_time_bin"] = np.select(conditions, choices, default="long_range")

    return res


def add_lead_time_category(df: pd.DataFrame, lead_time_col: str = "lead_time") -> pd.DataFrame:
    """
    Categorize forecast lead time into standard operational horizons (legacy Phase 1 compatibility):
    - short_range: 0 - 24 hours
    - medium_range: 25 - 72 hours
    - extended_range: > 72 hours
    """
    res = df.copy()
    if lead_time_col not in res.columns:
        return res

    conditions = [
        res[lead_time_col] <= 24,
        (res[lead_time_col] > 24) & (res[lead_time_col] <= 72),
        res[lead_time_col] > 72,
    ]
    choices = ["short_range", "medium_range", "extended_range"]

    res["lead_time_category"] = np.select(conditions, choices, default="extended_range")
    return res


def add_season_feature(
    df: pd.DataFrame,
    time_col: str = "timestamp",
    seasons_config: Optional[Dict[str, List[int]]] = None,
) -> pd.DataFrame:
    """
    Assign climatological season to records based on verification month:
    - winter: DJF (12, 1, 2)
    - spring: MAM (3, 4, 5)
    - summer: JJA (6, 7, 8)
    - autumn: SON (9, 10, 11)

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with timestamp column.
    time_col : str, default 'timestamp'
        Datetime column name.
    seasons_config : Optional[Dict[str, List[int]]]
        Dictionary mapping season names to month numbers.

    Returns
    -------
    pd.DataFrame
        DataFrame with 'season' column.
    """
    res = df.copy()
    if time_col not in res.columns:
        return res

    ts = pd.to_datetime(res[time_col], utc=True)
    months = ts.dt.month

    default_seasons = {
        "winter": [12, 1, 2],
        "spring": [3, 4, 5],
        "summer": [6, 7, 8],
        "autumn": [9, 10, 11],
    }
    s_map = seasons_config or default_seasons

    conditions = []
    choices = []
    for s_name, s_months in s_map.items():
        conditions.append(months.isin(s_months))
        choices.append(s_name)

    res["season"] = np.select(conditions, choices, default="winter")
    return res
