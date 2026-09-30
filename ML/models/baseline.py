"""
Baseline multi-model forecasting methods: Equal-Weight Blending & Single-Model Selectors.
"""

from typing import List, Optional
import numpy as np
import pandas as pd

from ingestion.schema import validate_forecast_dataframe
from models.base import BaseBlender


def equal_weight_blend(
    forecasts_df: pd.DataFrame,
    models: Optional[List[str]] = None,
    min_valid_models: int = 1,
    output_model_name: str = "equal_weight_blend",
) -> pd.DataFrame:
    """
    Calculate an unweighted ensemble arithmetic mean across available forecast models:

    .. math::
        F_{blend} = \\frac{1}{N} \\sum_{i=1}^{N} F_i

    Handles missing forecasts safely by dynamically computing the mean over
    non-null predictions for each spatiotemporal point and lead time.

    Parameters
    ----------
    forecasts_df : pd.DataFrame
        DataFrame conforming to standard forecast schema.
    models : Optional[List[str]], default None
        Specific model names to include. If None, all models in the DataFrame are blended.
    min_valid_models : int, default 1
        Minimum number of non-null models required to produce a valid blend value.
        Points with fewer valid models will result in NaN.
    output_model_name : str, default 'equal_weight_blend'
        Identifier to assign to the blended model output.

    Returns
    -------
    pd.DataFrame
        DataFrame conforming to standard forecast schema containing blended predictions.
    """
    df = validate_forecast_dataframe(forecasts_df, allow_nans=True)

    if models is not None:
        df = df[df["model_name"].isin(models)]
        if df.empty:
            raise ValueError(f"No forecast records found for specified models: {models}")

    group_cols = ["timestamp", "latitude", "longitude", "lead_time", "variable"]
    meta_cols = ["run_time", "region", "season", "weather_regime"]
    active_meta = [c for c in meta_cols if c in df.columns]
    all_group_cols = group_cols + active_meta

    # Group and compute arithmetic mean, counting valid non-null models
    grouped = df.groupby(all_group_cols, as_index=False)
    agg_df = grouped.agg(
        forecast_value=("forecast_value", "mean"),
        valid_model_count=("forecast_value", "count"),
    )

    # Apply minimum valid model threshold
    invalid_mask = agg_df["valid_model_count"] < min_valid_models
    agg_df.loc[invalid_mask, "forecast_value"] = np.nan

    agg_df["model_name"] = output_model_name
    agg_df = agg_df.drop(columns=["valid_model_count"])

    # Drop any resulting NaNs if min_valid_models was not satisfied
    blended_clean = agg_df.dropna(subset=["forecast_value"]).reset_index(drop=True)

    return validate_forecast_dataframe(blended_clean)


def single_model_forecast(
    forecasts_df: pd.DataFrame,
    model_name: str,
) -> pd.DataFrame:
    """
    Extract single-model baseline forecast records from a multi-model dataset.

    Parameters
    ----------
    forecasts_df : pd.DataFrame
        Input multi-model forecast DataFrame.
    model_name : str
        Target model name to filter.

    Returns
    -------
    pd.DataFrame
        Filtered forecast DataFrame conforming to standard schema.
    """
    df = validate_forecast_dataframe(forecasts_df, allow_nans=True)
    filtered = df[df["model_name"] == model_name].copy().reset_index(drop=True)

    if filtered.empty:
        raise ValueError(f"Model '{model_name}' not found in forecasts DataFrame.")

    return validate_forecast_dataframe(filtered)


class EqualWeightBlender(BaseBlender):
    """
    Class wrapper for Equal-Weight Ensemble Mean Blender.
    """

    def __init__(
        self,
        models: Optional[List[str]] = None,
        min_valid_models: int = 1,
        name: str = "equal_weight_blend",
    ):
        super().__init__(name=name)
        self.models = models
        self.min_valid_models = min_valid_models
        self.is_fitted = True  # Equal weighting requires no training

    def predict(self, forecasts_df: pd.DataFrame) -> pd.DataFrame:
        return equal_weight_blend(
            forecasts_df=forecasts_df,
            models=self.models,
            min_valid_models=self.min_valid_models,
            output_model_name=self.name,
        )
