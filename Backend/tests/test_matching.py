"""
Unit tests for the forecast-observation verification matching engine.
"""

import pytest
import pandas as pd
from preprocessing.matching import match_forecasts_to_observations


def test_match_forecasts_to_observations_error_calculation():
    ts = "2026-01-01T12:00:00Z"
    forecasts = pd.DataFrame([
        {
            "timestamp": ts,
            "latitude": 20.0,
            "longitude": 75.0,
            "lead_time": 24,
            "model_name": "nwp_model_a",
            "variable": "temperature",
            "forecast_value": 25.0,
        },
        {
            "timestamp": ts,
            "latitude": 20.0,
            "longitude": 75.0,
            "lead_time": 24,
            "model_name": "ai_model_c",
            "variable": "temperature",
            "forecast_value": 21.0,
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

    verified = match_forecasts_to_observations(forecasts, observations)
    assert len(verified) == 2

    # Verify model A: forecast=25, observed=23 -> error = +2.0, abs_err = 2.0, sq_err = 4.0
    rec_a = verified[verified["model_name"] == "nwp_model_a"].iloc[0]
    assert rec_a["forecast_error"] == pytest.approx(2.0, abs=1e-4)
    assert rec_a["absolute_error"] == pytest.approx(2.0, abs=1e-4)
    assert rec_a["squared_error"] == pytest.approx(4.0, abs=1e-4)

    # Verify model C: forecast=21, observed=23 -> error = -2.0, abs_err = 2.0, sq_err = 4.0
    rec_c = verified[verified["model_name"] == "ai_model_c"].iloc[0]
    assert rec_c["forecast_error"] == pytest.approx(-2.0, abs=1e-4)
    assert rec_c["absolute_error"] == pytest.approx(2.0, abs=1e-4)
    assert rec_c["squared_error"] == pytest.approx(4.0, abs=1e-4)

    # Required columns check
    for req_col in [
        "timestamp", "latitude", "longitude", "variable", "model_name", "lead_time",
        "forecast_value", "observed_value", "forecast_error", "absolute_error", "squared_error",
        "spatial_distance_km", "lead_time_bin", "season", "weather_regime"
    ]:
        assert req_col in verified.columns


def test_match_forecasts_to_observations_missing_obs():
    # Forecast exists without matching observation
    forecasts = pd.DataFrame([
        {
            "timestamp": "2026-01-01T12:00:00Z",
            "latitude": 20.0,
            "longitude": 75.0,
            "lead_time": 24,
            "model_name": "nwp_model_a",
            "variable": "temperature",
            "forecast_value": 25.0,
        }
    ])
    observations = pd.DataFrame([
        {
            "timestamp": "2026-01-05T12:00:00Z",  # Different date
            "latitude": 20.0,
            "longitude": 75.0,
            "variable": "temperature",
            "observed_value": 23.0,
        }
    ])
    verified = match_forecasts_to_observations(forecasts, observations)
    assert len(verified) == 0


def test_match_forecasts_to_observations_missing_forecasts():
    # Observation exists without matching forecast
    forecasts = pd.DataFrame(columns=[
        "timestamp", "latitude", "longitude", "lead_time", "model_name", "variable", "forecast_value"
    ])
    observations = pd.DataFrame([
        {
            "timestamp": "2026-01-01T12:00:00Z",
            "latitude": 20.0,
            "longitude": 75.0,
            "variable": "temperature",
            "observed_value": 23.0,
        }
    ])
    verified = match_forecasts_to_observations(forecasts, observations)
    assert len(verified) == 0
