"""
Unit tests for baseline multi-model blending methods.
"""

from datetime import datetime
import numpy as np
import pandas as pd
import pytest

from models.baseline import equal_weight_blend, single_model_forecast, EqualWeightBlender


def test_equal_weight_blend_exact_math():
    df = pd.DataFrame([
        {
            "timestamp": datetime(2026, 1, 1, 12, 0),
            "latitude": 20.0,
            "longitude": 75.0,
            "lead_time": 24,
            "model_name": "nwp_model_a",
            "variable": "temperature",
            "forecast_value": 20.0,
        },
        {
            "timestamp": datetime(2026, 1, 1, 12, 0),
            "latitude": 20.0,
            "longitude": 75.0,
            "lead_time": 24,
            "model_name": "nwp_model_b",
            "variable": "temperature",
            "forecast_value": 26.0,
        },
        {
            "timestamp": datetime(2026, 1, 1, 12, 0),
            "latitude": 20.0,
            "longitude": 75.0,
            "lead_time": 24,
            "model_name": "ai_model_c",
            "variable": "temperature",
            "forecast_value": 23.0,
        },
    ])

    blended = equal_weight_blend(df)
    assert len(blended) == 1
    # Expected: (20.0 + 26.0 + 23.0) / 3 = 23.0
    assert pytest.approx(blended["forecast_value"].iloc[0], 0.001) == 23.0
    assert blended["model_name"].iloc[0] == "equal_weight_blend"


def test_equal_weight_blend_missing_model_handling():
    """Test when one model is missing / NaN for a point, the blend safely averages remaining valid models."""
    df = pd.DataFrame([
        {
            "timestamp": datetime(2026, 1, 1, 12, 0),
            "latitude": 20.0,
            "longitude": 75.0,
            "lead_time": 24,
            "model_name": "nwp_model_a",
            "variable": "temperature",
            "forecast_value": 20.0,
        },
        {
            "timestamp": datetime(2026, 1, 1, 12, 0),
            "latitude": 20.0,
            "longitude": 75.0,
            "lead_time": 24,
            "model_name": "nwp_model_b",
            "variable": "temperature",
            "forecast_value": np.nan,  # Missing
        },
        {
            "timestamp": datetime(2026, 1, 1, 12, 0),
            "latitude": 20.0,
            "longitude": 75.0,
            "lead_time": 24,
            "model_name": "ai_model_c",
            "variable": "temperature",
            "forecast_value": 24.0,
        },
    ])

    blended = equal_weight_blend(df, min_valid_models=1)
    assert len(blended) == 1
    # Expected: (20.0 + 24.0) / 2 = 22.0
    assert pytest.approx(blended["forecast_value"].iloc[0], 0.001) == 22.0


def test_equal_weight_blend_min_valid_threshold():
    """When valid models are fewer than min_valid_models, point is safely dropped."""
    df = pd.DataFrame([
        {
            "timestamp": datetime(2026, 1, 1, 12, 0),
            "latitude": 20.0,
            "longitude": 75.0,
            "lead_time": 24,
            "model_name": "nwp_model_a",
            "variable": "temperature",
            "forecast_value": 20.0,
        },
        {
            "timestamp": datetime(2026, 1, 1, 12, 0),
            "latitude": 20.0,
            "longitude": 75.0,
            "lead_time": 24,
            "model_name": "nwp_model_b",
            "variable": "temperature",
            "forecast_value": np.nan,
        },
    ])

    blended = equal_weight_blend(df, min_valid_models=2)
    assert len(blended) == 0  # Only 1 valid model, requires 2


def test_single_model_forecast():
    df = pd.DataFrame([
        {
            "timestamp": datetime(2026, 1, 1, 12, 0),
            "latitude": 20.0,
            "longitude": 75.0,
            "lead_time": 24,
            "model_name": "nwp_model_a",
            "variable": "temperature",
            "forecast_value": 20.0,
        },
        {
            "timestamp": datetime(2026, 1, 1, 12, 0),
            "latitude": 20.0,
            "longitude": 75.0,
            "lead_time": 24,
            "model_name": "nwp_model_b",
            "variable": "temperature",
            "forecast_value": 26.0,
        },
    ])

    sub_a = single_model_forecast(df, "nwp_model_a")
    assert len(sub_a) == 1
    assert sub_a["model_name"].iloc[0] == "nwp_model_a"

    with pytest.raises(ValueError, match="not found"):
        single_model_forecast(df, "nonexistent_model")


def test_equal_weight_blender_class():
    df = pd.DataFrame([
        {
            "timestamp": datetime(2026, 1, 1, 12, 0),
            "latitude": 20.0,
            "longitude": 75.0,
            "lead_time": 24,
            "model_name": "nwp_model_a",
            "variable": "temperature",
            "forecast_value": 10.0,
        },
        {
            "timestamp": datetime(2026, 1, 1, 12, 0),
            "latitude": 20.0,
            "longitude": 75.0,
            "lead_time": 24,
            "model_name": "nwp_model_b",
            "variable": "temperature",
            "forecast_value": 20.0,
        },
    ])

    blender = EqualWeightBlender()
    out = blender.predict(df)
    assert out["forecast_value"].iloc[0] == 15.0
