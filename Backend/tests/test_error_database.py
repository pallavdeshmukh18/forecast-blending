"""
Unit tests for Historical Error Database storage, metrics calculation, and queries.
"""

from pathlib import Path
import pytest
import pandas as pd
from evaluation.error_database import HistoricalErrorDatabase


def test_historical_error_database_add_and_queries(tmp_path: Path):
    df = pd.DataFrame([
        {
            "timestamp": "2026-01-01T12:00:00Z",
            "latitude": 20.0,
            "longitude": 75.0,
            "variable": "temperature",
            "model_name": "nwp_model_a",
            "lead_time": 24,
            "forecast_value": 25.0,
            "observed_value": 22.0,
            "region": "plains",
            "season": "winter",
            "weather_regime": "standard",
        },
        {
            "timestamp": "2026-01-02T12:00:00Z",
            "latitude": 20.0,
            "longitude": 75.0,
            "variable": "temperature",
            "model_name": "nwp_model_a",
            "lead_time": 24,
            "forecast_value": 24.0,
            "observed_value": 22.0,
            "region": "plains",
            "season": "winter",
            "weather_regime": "standard",
        },
    ])

    db = HistoricalErrorDatabase()
    db.add_records(df)

    assert len(db) == 2

    # Query RMSE for NWP model A at 24h lead time: errors are +3.0 and +2.0 -> RMSE = sqrt((9+4)/2) = sqrt(6.5) = ~2.5495
    rmse = db.query_metric(model_name="nwp_model_a", variable="temperature", metric="rmse", lead_time=24)
    assert rmse == pytest.approx(2.5495, abs=1e-3)

    # Query MAE: (3.0 + 2.0)/2 = 2.5
    mae = db.query_metric(model_name="nwp_model_a", variable="temperature", metric="mae", lead_time=24)
    assert mae == pytest.approx(2.5, abs=1e-3)

    # Test Save & Load
    save_path = tmp_path / "error_db.parquet"
    db.save(save_path)
    loaded_db = HistoricalErrorDatabase.load(save_path)
    assert len(loaded_db) == 2
