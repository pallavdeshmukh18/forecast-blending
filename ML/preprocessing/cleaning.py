"""
Data cleaning and quality assurance for meteorological time series.
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from configs import AppConfig, get_config


def clip_to_physical_bounds(
    df: pd.DataFrame,
    value_col: str,
    variable_col: str = "variable",
    config: Optional[AppConfig] = None,
) -> pd.DataFrame:
    """
    Clip numerical values in a DataFrame to physical meteorological bounds defined in configuration.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    value_col : str
        Name of the column containing numerical values (e.g. 'forecast_value' or 'observed_value').
    variable_col : str, default 'variable'
        Column indicating the meteorological variable type.
    config : Optional[AppConfig]
        Application configuration containing physical bound rules.

    Returns
    -------
    pd.DataFrame
        DataFrame with bounded values.
    """
    cfg = config or get_config()
    cleaned = df.copy()

    for var_name, var_cfg in cfg.variables.items():
        mask = cleaned[variable_col] == var_name
        if mask.any():
            cleaned.loc[mask, value_col] = cleaned.loc[mask, value_col].clip(
                lower=var_cfg.min_physical_bound,
                upper=var_cfg.max_physical_bound,
            )

    return cleaned


def remove_invalid_records(
    df: pd.DataFrame,
    value_col: str,
    drop_nans: bool = True,
    drop_duplicates: bool = True,
    subset_keys: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Filter out invalid (NaN, Inf, duplicate) records from a meteorological DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    value_col : str
        Column to check for finiteness.
    drop_nans : bool, default True
        Whether to drop NaN/Inf rows.
    drop_duplicates : bool, default True
        Whether to drop duplicated key rows.
    subset_keys : Optional[List[str]]
        Keys for identifying duplicates.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame.
    """
    cleaned = df.copy()

    if drop_nans:
        finite_mask = np.isfinite(cleaned[value_col]) & cleaned[value_col].notna()
        cleaned = cleaned[finite_mask]

    if drop_duplicates:
        if subset_keys is not None:
            cleaned = cleaned.drop_duplicates(subset=subset_keys, keep="last")
        else:
            cleaned = cleaned.drop_duplicates(keep="last")

    return cleaned.reset_index(drop=True)
