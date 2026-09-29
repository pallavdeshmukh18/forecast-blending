"""
Unit tests for temporal alignment and verification timestamp validation.
"""

from datetime import datetime, timezone
import pandas as pd
from preprocessing.temporal import validate_and_compute_valid_time, match_temporal_tolerance


def test_validate_and_compute_valid_time_from_run_and_lead():
    df = pd.DataFrame([
        {
            "run_time": "2026-01-01T00:00:00Z",
            "lead_time": 24,
            "latitude": 20.0,
            "longitude": 75.0,
            "model_name": "nwp_model_a",
            "variable": "temperature",
            "forecast_value": 25.0,
        }
    ])
    res = validate_and_compute_valid_time(df)
    assert "timestamp" in res.columns
    assert res["timestamp"].iloc[0] == pd.Timestamp("2026-01-02T00:00:00Z")


def test_match_temporal_tolerance_valid():
    fcs = pd.DataFrame([
        {"timestamp": "2026-01-01T12:00:00Z", "forecast_value": 22.0},
    ])
    obs = pd.DataFrame([
        {"timestamp": "2026-01-01T12:15:00Z", "observed_value": 21.8},
    ])
    # 15 minutes difference <= 30 minutes tolerance
    matched = match_temporal_tolerance(fcs, obs, tolerance_minutes=30)
    assert len(matched) == 1
    assert matched["observed_value"].notna().all()


def test_match_temporal_tolerance_rejection():
    fcs = pd.DataFrame([
        {"timestamp": "2026-01-01T12:00:00Z", "forecast_value": 22.0},
    ])
    obs = pd.DataFrame([
        {"timestamp": "2026-01-01T13:30:00Z", "observed_value": 21.8},
    ])
    # 90 minutes difference > 30 minutes tolerance
    matched = match_temporal_tolerance(fcs, obs, tolerance_minutes=30)
    # Merged observation value should be NaN since it falls outside tolerance
    assert matched["observed_value"].isna().all()
