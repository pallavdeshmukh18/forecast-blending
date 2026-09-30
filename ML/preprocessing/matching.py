"""
Core Forecast-Observation Verification Matching Engine.
Produces the standardized verified forecast-observation pair dataset.
"""

from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd

from configs import AppConfig, SpatialAlignmentConfig, TemporalAlignmentConfig, get_config
from ingestion.schema import validate_forecast_dataframe, validate_observation_dataframe
from preprocessing.spatial import match_spatial_nearest
from preprocessing.temporal import validate_and_compute_valid_time
from preprocessing.features import add_lead_time_bins, add_season_feature, add_temporal_features
from preprocessing.regimes import WeatherRegimeClassifier


def match_forecasts_to_observations(
    forecasts_df: pd.DataFrame,
    observations_df: pd.DataFrame,
    spatial_config: Optional[Union[SpatialAlignmentConfig, Dict]] = None,
    temporal_config: Optional[Union[TemporalAlignmentConfig, Dict]] = None,
    config: Optional[AppConfig] = None,
) -> pd.DataFrame:
    """
    Match multi-model forecast records with ground-truth observations to create verified pairs.
    Handles distinct model grid coordinates independently by finding the nearest grid point
    per model for each observation location within max_distance_km.

    Parameters
    ----------
    forecasts_df : pd.DataFrame
        Raw or standardized forecast DataFrame.
    observations_df : pd.DataFrame
        Raw or standardized observation DataFrame.
    spatial_config : Optional[SpatialAlignmentConfig | Dict]
        Spatial matching parameters (max_distance_km, method).
    temporal_config : Optional[TemporalAlignmentConfig | Dict]
        Temporal matching parameters (tolerance_minutes).
    config : Optional[AppConfig]
        Application configuration.

    Returns
    -------
    pd.DataFrame
        Verified forecast-observation dataset containing:
        - timestamp (UTC valid time)
        - latitude, longitude (observation coordinates)
        - variable
        - model_name
        - lead_time, lead_time_bin
        - forecast_value, observed_value
        - forecast_error (forecast_value - observed_value)
        - absolute_error (|forecast_error|)
        - squared_error (forecast_error^2)
        - spatial_distance_km
        - season, weather_regime
        - optional metadata (station_id, region, run_time)
    """
    cfg = config or get_config()

    # Extract spatial parameters
    if spatial_config is None:
        max_dist = cfg.spatial_alignment.max_distance_km if cfg else 50.0
    elif isinstance(spatial_config, dict):
        max_dist = spatial_config.get("max_distance_km", 50.0)
    else:
        max_dist = spatial_config.max_distance_km

    # Extract temporal parameters
    if temporal_config is None:
        tol_mins = cfg.temporal_alignment.tolerance_minutes if cfg else 30
    elif isinstance(temporal_config, dict):
        tol_mins = temporal_config.get("tolerance_minutes", 30)
    else:
        tol_mins = temporal_config.tolerance_minutes

    f_df = validate_forecast_dataframe(forecasts_df, allow_nans=True)
    o_df = validate_observation_dataframe(observations_df)

    # Ensure canonical valid verification timestamp
    f_df = validate_and_compute_valid_time(f_df)

    model_verified_list = []

    # Process each forecast model independently to accommodate differing spatial grids
    for m_name in f_df["model_name"].unique():
        f_sub = f_df[f_df["model_name"] == m_name].copy()

        # 1. Spatial Matching Mapping for this specific model's grid
        spatial_mapping = match_spatial_nearest(
            obs_coords=o_df[["latitude", "longitude"]],
            forecast_coords=f_sub[["latitude", "longitude"]],
            max_distance_km=max_dist,
        )

        valid_spatial = spatial_mapping[spatial_mapping["is_spatial_matched"]].copy()
        if valid_spatial.empty:
            continue

        # Attach matched forecast coordinates to observations
        o_matched = pd.merge(
            o_df,
            valid_spatial,
            left_on=["latitude", "longitude"],
            right_on=["obs_latitude", "obs_longitude"],
            how="inner",
        )

        f_sub["f_lat_round"] = f_sub["latitude"].round(4)
        f_sub["f_lon_round"] = f_sub["longitude"].round(4)
        o_matched["f_lat_round"] = o_matched["forecast_latitude"].round(4)
        o_matched["f_lon_round"] = o_matched["forecast_longitude"].round(4)

        # Merge on variable and spatial grid key
        join_keys = ["variable", "f_lat_round", "f_lon_round"]
        base_obs_cols = [
            "variable",
            "f_lat_round",
            "f_lon_round",
            "timestamp",
            "observed_value",
            "spatial_distance_km",
            "obs_latitude",
            "obs_longitude",
        ]
        opt_obs_cols = [c for c in ["station_id", "region", "elevation"] if c in o_matched.columns and c not in f_sub.columns]
        obs_select_cols = [c for c in base_obs_cols + opt_obs_cols if c in o_matched.columns]

        merged = pd.merge(
            f_sub,
            o_matched[obs_select_cols],
            on=join_keys,
            suffixes=("", "_obs"),
            how="inner",
        )

        if merged.empty:
            continue

        # 2. Temporal tolerance check
        time_diff_secs = (merged["timestamp"] - merged["timestamp_obs"]).abs().dt.total_seconds()
        time_diff_mins = time_diff_secs / 60.0
        temporal_mask = time_diff_mins <= tol_mins
        m_verified = merged[temporal_mask].copy()

        if m_verified.empty:
            continue

        # Use observation coordinates as the canonical geographic reference
        if "obs_latitude" in m_verified.columns:
            m_verified["latitude"] = m_verified["obs_latitude"]
            m_verified["longitude"] = m_verified["obs_longitude"]

        # Drop intermediate helper columns
        m_verified = m_verified.drop(
            columns=["f_lat_round", "f_lon_round", "timestamp_obs", "obs_latitude", "obs_longitude"],
            errors="ignore",
        )

        # Drop NaNs
        m_verified = m_verified[m_verified["forecast_value"].notna() & m_verified["observed_value"].notna()].copy()

        model_verified_list.append(m_verified)

    if not model_verified_list:
        cols = [
            "timestamp",
            "latitude",
            "longitude",
            "variable",
            "model_name",
            "lead_time",
            "forecast_value",
            "observed_value",
            "forecast_error",
            "absolute_error",
            "squared_error",
            "spatial_distance_km",
            "lead_time_bin",
            "season",
            "weather_regime",
        ]
        return pd.DataFrame(columns=cols)

    verified = pd.concat(model_verified_list, ignore_index=True)

    # 3. Error metrics calculation
    err = verified["forecast_value"] - verified["observed_value"]
    verified["forecast_error"] = np.round(err, 4)
    verified["absolute_error"] = np.round(np.abs(err), 4)
    verified["squared_error"] = np.round(err**2, 4)

    # 4. Lead-time binning, Season, Temporal features, Weather Regime
    bins_cfg = cfg.lead_time_bins if cfg else None
    seasons_cfg = cfg.seasons if cfg else None

    verified = add_lead_time_bins(verified, lead_time_col="lead_time", bins_config=bins_cfg)
    verified = add_season_feature(verified, time_col="timestamp", seasons_config=seasons_cfg)
    verified = add_temporal_features(verified, time_col="timestamp")

    classifier = WeatherRegimeClassifier()
    verified = classifier.classify(verified)

    # 5. Preserve strict chronological order to avoid temporal leakage
    sort_cols = [c for c in ["timestamp", "lead_time", "variable", "model_name"] if c in verified.columns]
    verified = verified.sort_values(by=sort_cols).reset_index(drop=True)

    return verified
