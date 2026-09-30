"""
Unit tests for the Single Operational Forecast Pipeline Orchestrator.
"""

import pytest
import pandas as pd
import numpy as np

from pipeline.forecast_pipeline import (
    ForecastResult,
    ForecastPipelineOrchestrator,
    run_forecast_pipeline,
)


def test_forecast_pipeline_synthetic_run():
    res = run_forecast_pipeline(
        mode="synthetic",
        variable="rainfall",
        lead_time=24,
        blender_method="regime_blender",
    )

    assert isinstance(res, ForecastResult)
    assert res.pipeline_version == "0.6.0"
    assert res.mode == "synthetic"
    assert res.target_variable == "rainfall"
    assert res.target_lead_time == 24
    assert res.forecast_count > 0
    assert len(res.forecast) > 0
    assert "dominant_regime" in res.regime
    assert "mean_spread" in res.uncertainty
    assert len(res.probabilities) > 0
    assert res.verification is not None
    assert "mae" in res.verification
    assert "rmse" in res.verification


def test_forecast_pipeline_custom_inputs():
    f_df = pd.DataFrame({
        "timestamp": ["2026-01-01T12:00:00Z", "2026-01-01T12:00:00Z"],
        "latitude": [20.0, 20.0],
        "longitude": [75.0, 75.0],
        "lead_time": [24, 24],
        "variable": ["temperature", "temperature"],
        "model_name": ["model_1", "model_2"],
        "forecast_value": [22.0, 24.0],
    })
    o_df = pd.DataFrame({
        "timestamp": ["2026-01-01T12:00:00Z"],
        "latitude": [20.0],
        "longitude": [75.0],
        "variable": ["temperature"],
        "observed_value": [23.0],
    })

    res = run_forecast_pipeline(
        mode="synthetic",
        variable="temperature",
        lead_time=24,
        forecast_df=f_df,
        observation_df=o_df,
        blender_method="equal_weight",
    )

    assert res.forecast_count == 1
    # Equal weight of 22 and 24 = 23.0
    assert res.forecast[0]["blend_value"] == pytest.approx(23.0, abs=1e-3)
    assert res.verification is not None
    assert res.verification["mae"] == pytest.approx(0.0, abs=1e-3)
