"""
Spatiotemporal alignment routines for multi-model forecasts and ground-truth observations.
"""

from typing import List, Optional
import pandas as pd

from ingestion.schema import validate_forecast_dataframe, validate_observation_dataframe
from preprocessing.matching import match_forecasts_to_observations


def align_forecasts_and_observations(
    forecasts_df: pd.DataFrame,
    observations_df: pd.DataFrame,
    on_keys: Optional[List[str]] = None,
    allow_missing_obs: bool = False,
) -> pd.DataFrame:
    """
    Merge and align multi-model forecasts with corresponding ground-truth observations.

    Parameters
    ----------
    forecasts_df : pd.DataFrame
        DataFrame conforming to standard forecast schema.
    observations_df : pd.DataFrame
        DataFrame conforming to standard observation schema.
    on_keys : Optional[List[str]], default None
        Key columns for spatiotemporal matching. Defaults to ['timestamp', 'latitude', 'longitude', 'variable'].
    allow_missing_obs : bool, default False
        If False, performs inner join dropping unverified forecasts. If True, performs left join.

    Returns
    -------
    pd.DataFrame
        Aligned verification DataFrame containing both 'forecast_value' and 'observed_value'.
    """
    f_df = validate_forecast_dataframe(forecasts_df, allow_nans=True)
    o_df = validate_observation_dataframe(observations_df)

    keys = on_keys or ["timestamp", "latitude", "longitude", "variable"]

    # Select only required columns from observation df to avoid suffix collisions
    obs_cols = list(set(keys + ["observed_value", "station_id"]))
    obs_subset = o_df[[col for col in obs_cols if col in o_df.columns]].drop_duplicates(subset=keys)

    how = "left" if allow_missing_obs else "inner"
    aligned = pd.merge(f_df, obs_subset, on=keys, how=how)

    if not allow_missing_obs:
        aligned = aligned[aligned["observed_value"].notna()].reset_index(drop=True)

    return aligned


def pivot_models_wide(
    forecasts_df: pd.DataFrame,
    index_cols: Optional[List[str]] = None,
    model_col: str = "model_name",
    value_col: str = "forecast_value",
) -> pd.DataFrame:
    """
    Pivot forecast DataFrame so that each model occupies its own column.

    Parameters
    ----------
    forecasts_df : pd.DataFrame
        Forecast DataFrame in standard long format.
    index_cols : Optional[List[str]]
        Index columns for pivoting. Defaults to ['timestamp', 'latitude', 'longitude', 'lead_time', 'variable'].
    model_col : str
        Column containing model names.
    value_col : str
        Column containing numerical predictions.

    Returns
    -------
    pd.DataFrame
        Pivoted wide DataFrame with one column per model.
    """
    default_index = ["timestamp", "latitude", "longitude", "lead_time", "variable"]
    idx = index_cols or [c for c in default_index if c in forecasts_df.columns]

    pivoted = forecasts_df.pivot_table(
        index=idx,
        columns=model_col,
        values=value_col,
        aggfunc="first",
    ).reset_index()

    pivoted.columns.name = None
    return pivoted


__all__ = [
    "align_forecasts_and_observations",
    "pivot_models_wide",
    "match_forecasts_to_observations",
]
