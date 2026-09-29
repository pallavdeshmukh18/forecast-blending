"""
Unit tests for data alignment and pivot routines.
"""

from datetime import datetime
import pandas as pd
from preprocessing.alignment import align_forecasts_and_observations, pivot_models_wide


def test_align_forecasts_and_observations():
    ts = datetime(2026, 1, 1, 12, 0)
    forecasts = pd.DataFrame([
        {
            "timestamp": ts,
            "latitude": 20.0,
            "longitude": 75.0,
            "lead_time": 24,
            "model_name": "nwp_model_a",
            "variable": "temperature",
            "forecast_value": 22.0,
        },
        {
            "timestamp": ts,
            "latitude": 20.0,
            "longitude": 75.0,
            "lead_time": 24,
            "model_name": "nwp_model_b",
            "variable": "temperature",
            "forecast_value": 24.0,
        },
    ])

    observations = pd.DataFrame([
        {
            "timestamp": ts,
            "latitude": 20.0,
            "longitude": 75.0,
            "variable": "temperature",
            "observed_value": 23.0,
            "station_id": "STN_01",
        }
    ])

    aligned = align_forecasts_and_observations(forecasts, observations)
    assert len(aligned) == 2
    assert "observed_value" in aligned.columns
    assert "forecast_value" in aligned.columns
    assert (aligned["observed_value"] == 23.0).all()


def test_pivot_models_wide():
    ts = datetime(2026, 1, 1, 12, 0)
    forecasts = pd.DataFrame([
        {
            "timestamp": ts,
            "latitude": 20.0,
            "longitude": 75.0,
            "lead_time": 24,
            "model_name": "nwp_model_a",
            "variable": "temperature",
            "forecast_value": 22.0,
        },
        {
            "timestamp": ts,
            "latitude": 20.0,
            "longitude": 75.0,
            "lead_time": 24,
            "model_name": "nwp_model_b",
            "variable": "temperature",
            "forecast_value": 24.0,
        },
    ])

    wide = pivot_models_wide(forecasts)
    assert "nwp_model_a" in wide.columns
    assert "nwp_model_b" in wide.columns
    assert wide["nwp_model_a"].iloc[0] == 22.0
    assert wide["nwp_model_b"].iloc[0] == 24.0
