"""
Internal Standard Meteorological Data Schema.

This module establishes the unified contract that every forecast provider
(NWP, AI/ML, Ensemble, Statistical) and ground-truth observation source must
conform to before ingestion into the blending and evaluation pipeline.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, field_validator


# Required schema columns
REQUIRED_FORECAST_COLUMNS = [
    "timestamp",
    "latitude",
    "longitude",
    "lead_time",
    "model_name",
    "variable",
    "forecast_value",
]

REQUIRED_OBSERVATION_COLUMNS = [
    "timestamp",
    "latitude",
    "longitude",
    "variable",
    "observed_value",
]

OPTIONAL_FORECAST_COLUMNS = [
    "run_time",
    "ensemble_member",
    "region",
    "season",
    "weather_regime",
]

OPTIONAL_OBSERVATION_COLUMNS = [
    "station_id",
    "region",
    "elevation",
    "quality_flag",
]


class ForecastRecord(BaseModel):
    """
    Standard schema for a single point-forecast prediction.
    """
    timestamp: datetime = Field(..., description="UTC valid forecast verification time")
    latitude: float = Field(..., ge=-90.0, le=90.0, description="Latitude in decimal degrees")
    longitude: float = Field(..., ge=-180.0, le=180.0, description="Longitude in decimal degrees")
    lead_time: int = Field(..., ge=0, description="Forecast lead time horizon in hours")
    model_name: str = Field(..., min_length=1, description="Unique model identifier")
    variable: str = Field(..., min_length=1, description="Standard meteorological variable name")
    forecast_value: float = Field(..., description="Forecasted physical value in standard units")

    # Optional metadata
    run_time: Optional[datetime] = Field(None, description="UTC model initialization cycle time")
    ensemble_member: Optional[Union[int, str]] = Field(None, description="Ensemble member identifier")
    region: Optional[str] = Field(None, description="Geographic region tag")
    season: Optional[str] = Field(None, description="Climatological season tag")
    weather_regime: Optional[str] = Field(None, description="Synoptic weather regime tag")

    @field_validator("forecast_value")
    @classmethod
    def check_finite_value(cls, v: float) -> float:
        if np.isnan(v) or np.isinf(v):
            raise ValueError("forecast_value must be a finite numerical value")
        return float(v)


class ObservationRecord(BaseModel):
    """
    Standard schema for a single ground-truth meteorological observation.
    """
    timestamp: datetime = Field(..., description="UTC observation timestamp")
    latitude: float = Field(..., ge=-90.0, le=90.0, description="Latitude in decimal degrees")
    longitude: float = Field(..., ge=-180.0, le=180.0, description="Longitude in decimal degrees")
    variable: str = Field(..., min_length=1, description="Standard meteorological variable name")
    observed_value: float = Field(..., description="Observed physical value in standard units")

    # Optional metadata
    station_id: Optional[str] = Field(None, description="Station or sensor identifier")
    region: Optional[str] = Field(None, description="Geographic region tag")
    elevation: Optional[float] = Field(None, description="Station elevation above sea level in meters")
    quality_flag: Optional[int] = Field(1, description="Data quality control flag (1 = valid, 0 = suspect)")

    @field_validator("observed_value")
    @classmethod
    def check_finite_value(cls, v: float) -> float:
        if np.isnan(v) or np.isinf(v):
            raise ValueError("observed_value must be a finite numerical value")
        return float(v)


def validate_forecast_dataframe(
    df: pd.DataFrame,
    allow_nans: bool = False,
    enforce_utc: bool = True,
) -> pd.DataFrame:
    """
    Validate that a pandas DataFrame conforms to the internal standard forecast schema.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame to validate.
    allow_nans : bool, default False
        Whether to allow NaN forecast_values (e.g. for incomplete ensemble grids).
    enforce_utc : bool, default True
        Ensure timestamps are converted to UTC timezone-aware datetimes.

    Returns
    -------
    pd.DataFrame
        Validated DataFrame with standardized types.

    Raises
    ------
    ValueError
        If required columns are missing, data types are invalid, or values are out of physical bounds.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pandas DataFrame, got {type(df)}")

    missing = [col for col in REQUIRED_FORECAST_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Forecast DataFrame missing required columns: {missing}")

    validated_df = df.copy()

    # Convert timestamp to UTC datetime
    if enforce_utc:
        validated_df["timestamp"] = pd.to_datetime(validated_df["timestamp"], utc=True)
    else:
        if not pd.api.types.is_datetime64_any_dtype(validated_df["timestamp"]):
            validated_df["timestamp"] = pd.to_datetime(validated_df["timestamp"])

    # Convert run_time if present
    if "run_time" in validated_df.columns and validated_df["run_time"].notna().any():
        if enforce_utc:
            validated_df["run_time"] = pd.to_datetime(validated_df["run_time"], utc=True)
        else:
            if not pd.api.types.is_datetime64_any_dtype(validated_df["run_time"]):
                validated_df["run_time"] = pd.to_datetime(validated_df["run_time"])

    # Coordinate ranges
    if (validated_df["latitude"] < -90.0).any() or (validated_df["latitude"] > 90.0).any():
        raise ValueError("Latitude values must be between -90.0 and 90.0 degrees")

    if (validated_df["longitude"] < -180.0).any() or (validated_df["longitude"] > 180.0).any():
        raise ValueError("Longitude values must be between -180.0 and 180.0 degrees")

    # Lead time check
    if (validated_df["lead_time"] < 0).any():
        raise ValueError("lead_time must be non-negative (>= 0 hours)")

    # Model name and variable check
    if validated_df["model_name"].isna().any() or (validated_df["model_name"] == "").any():
        raise ValueError("model_name cannot be empty or null")

    if validated_df["variable"].isna().any() or (validated_df["variable"] == "").any():
        raise ValueError("variable cannot be empty or null")

    # Value finiteness
    if not allow_nans:
        if validated_df["forecast_value"].isna().any() or np.isinf(validated_df["forecast_value"]).any():
            raise ValueError("forecast_value contains NaN or Inf values while allow_nans is False")

    # Standardize types
    validated_df["latitude"] = validated_df["latitude"].astype(float)
    validated_df["longitude"] = validated_df["longitude"].astype(float)
    validated_df["lead_time"] = validated_df["lead_time"].astype(int)
    validated_df["model_name"] = validated_df["model_name"].astype(str)
    validated_df["variable"] = validated_df["variable"].astype(str)
    validated_df["forecast_value"] = validated_df["forecast_value"].astype(float)

    return validated_df


def validate_observation_dataframe(
    df: pd.DataFrame,
    enforce_utc: bool = True,
) -> pd.DataFrame:
    """
    Validate that a pandas DataFrame conforms to the internal standard observation schema.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame to validate.
    enforce_utc : bool, default True
        Ensure timestamps are converted to UTC timezone-aware datetimes.

    Returns
    -------
    pd.DataFrame
        Validated DataFrame with standardized types.

    Raises
    ------
    ValueError
        If required columns are missing or values are invalid.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pandas DataFrame, got {type(df)}")

    missing = [col for col in REQUIRED_OBSERVATION_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Observation DataFrame missing required columns: {missing}")

    validated_df = df.copy()

    # Convert timestamp to UTC datetime
    if enforce_utc:
        validated_df["timestamp"] = pd.to_datetime(validated_df["timestamp"], utc=True)
    else:
        if not pd.api.types.is_datetime64_any_dtype(validated_df["timestamp"]):
            validated_df["timestamp"] = pd.to_datetime(validated_df["timestamp"])

    # Coordinate ranges
    if (validated_df["latitude"] < -90.0).any() or (validated_df["latitude"] > 90.0).any():
        raise ValueError("Latitude values must be between -90.0 and 90.0 degrees")

    if (validated_df["longitude"] < -180.0).any() or (validated_df["longitude"] > 180.0).any():
        raise ValueError("Longitude values must be between -180.0 and 180.0 degrees")

    if validated_df["observed_value"].isna().any() or np.isinf(validated_df["observed_value"]).any():
        raise ValueError("observed_value contains NaN or Inf values")

    validated_df["latitude"] = validated_df["latitude"].astype(float)
    validated_df["longitude"] = validated_df["longitude"].astype(float)
    validated_df["variable"] = validated_df["variable"].astype(str)
    validated_df["observed_value"] = validated_df["observed_value"].astype(float)

    return validated_df
