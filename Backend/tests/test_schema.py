"""
Unit tests for internal meteorological schema validation.
"""

from datetime import datetime
import pytest
import pandas as pd
from pydantic import ValidationError

from ingestion.schema import (
    ForecastRecord,
    ObservationRecord,
    validate_forecast_dataframe,
    validate_observation_dataframe,
)


def test_forecast_record_valid():
    rec = ForecastRecord(
        timestamp=datetime(2026, 1, 1, 12, 0),
        latitude=25.5,
        longitude=80.2,
        lead_time=24,
        model_name="nwp_model_a",
        variable="temperature",
        forecast_value=22.5,
        region="plains",
    )
    assert rec.latitude == 25.5
    assert rec.lead_time == 24
    assert rec.forecast_value == 22.5


def test_forecast_record_invalid_latitude():
    with pytest.raises(ValidationError):
        ForecastRecord(
            timestamp=datetime(2026, 1, 1, 12, 0),
            latitude=95.0,  # Invalid > 90
            longitude=80.0,
            lead_time=24,
            model_name="nwp_model_a",
            variable="temperature",
            forecast_value=22.5,
        )


def test_forecast_record_negative_lead_time():
    with pytest.raises(ValidationError):
        ForecastRecord(
            timestamp=datetime(2026, 1, 1, 12, 0),
            latitude=20.0,
            longitude=80.0,
            lead_time=-6,  # Invalid < 0
            model_name="nwp_model_a",
            variable="temperature",
            forecast_value=22.5,
        )


def test_observation_record_valid():
    obs = ObservationRecord(
        timestamp=datetime(2026, 1, 1, 12, 0),
        latitude=18.5,
        longitude=72.8,
        variable="rainfall",
        observed_value=12.4,
        station_id="STN_COASTAL_01",
    )
    assert obs.variable == "rainfall"
    assert obs.observed_value == 12.4


def test_validate_forecast_dataframe_missing_column():
    df = pd.DataFrame({
        "timestamp": [datetime(2026, 1, 1, 12, 0)],
        "latitude": [25.0],
        "longitude": [80.0],
        # missing lead_time, model_name, variable, forecast_value
    })
    with pytest.raises(ValueError, match="missing required columns"):
        validate_forecast_dataframe(df)


def test_validate_forecast_dataframe_success():
    df = pd.DataFrame({
        "timestamp": ["2026-01-01 12:00:00"],
        "latitude": [25.0],
        "longitude": [80.0],
        "lead_time": [24],
        "model_name": ["nwp_model_a"],
        "variable": ["temperature"],
        "forecast_value": [21.5],
    })
    validated = validate_forecast_dataframe(df)
    assert pd.api.types.is_datetime64_any_dtype(validated["timestamp"])
    assert validated["lead_time"].dtype == int
