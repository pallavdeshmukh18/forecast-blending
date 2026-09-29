"""
Robustness and Fault-Tolerance Simulation Tests for Phase 6.
Evaluates missing models, corrupted/biased models, extreme model disagreement, and missing verification observations.
"""

import numpy as np
import pandas as pd
import pytest

from pipeline.forecast_pipeline import run_forecast_pipeline
from models.skill_weighting import InverseVarianceSkillBlender
from models.baseline import equal_weight_blend


def test_missing_model_renormalization():
    # Only 2 models provided out of 3
    df = pd.DataFrame({
        "timestamp": ["2026-01-01T12:00:00Z", "2026-01-01T12:00:00Z"],
        "latitude": [20.0, 20.0],
        "longitude": [75.0, 75.0],
        "model_name": ["nwp_model_a", "ai_model_c"],  # nwp_model_b is missing
        "lead_time": [24, 24],
        "variable": ["rainfall", "rainfall"],
        "forecast_value": [15.0, 25.0],
    })

    blended = equal_weight_blend(df, models=["nwp_model_a", "ai_model_c"])
    assert len(blended) == 1
    # Equal weight of 2 models: (15 + 25) / 2 = 20.0
    assert blended["forecast_value"].iloc[0] == pytest.approx(20.0, abs=1e-3)


def test_corrupted_model_bias_downweighting():
    # Simulate historical error DB where model_b is heavily biased/corrupted
    records = []
    for _ in range(30):
        records.append({
            "timestamp": "2026-01-01T12:00:00Z",
            "latitude": 20.0,
            "longitude": 75.0,
            "lead_time": 24,
            "variable": "temperature",
            "model_name": "model_good",
            "forecast_value": 25.0,
            "observed_value": 25.2,
            "absolute_error": 0.2,
            "squared_error": 0.04,
        })
        records.append({
            "timestamp": "2026-01-01T12:00:00Z",
            "latitude": 20.0,
            "longitude": 75.0,
            "lead_time": 24,
            "variable": "temperature",
            "model_name": "model_corrupted",
            "forecast_value": 75.0,  # heavily biased
            "observed_value": 25.2,
            "absolute_error": 49.8,
            "squared_error": 2480.0,
        })

    hist_df = pd.DataFrame(records)
    blender = InverseVarianceSkillBlender()
    blender.fit(hist_df)
    weights = blender.get_weights(variable="temperature", lead_time=24)

    assert "model_good" in weights
    assert "model_corrupted" in weights
    # Good model should receive dominant weight compared to corrupted model
    assert weights["model_good"] > 5.0 * weights["model_corrupted"]
    assert pytest.approx(sum(weights.values()), abs=1e-4) == 1.0


def test_high_disagreement_uncertainty_response():
    # Pipeline with high spread across models
    f_df = pd.DataFrame({
        "timestamp": ["2026-01-01T12:00:00Z"] * 3,
        "latitude": [20.0] * 3,
        "longitude": [75.0] * 3,
        "lead_time": [24] * 3,
        "variable": ["rainfall"] * 3,
        "model_name": ["model_a", "model_b", "model_c"],
        "forecast_value": [5.0, 50.0, 95.0],  # high disagreement
    })

    res = run_forecast_pipeline(
        mode="synthetic",
        variable="rainfall",
        lead_time=24,
        blender_method="equal_weight",
        forecast_df=f_df,
        observation_df=pd.DataFrame(),  # No observations
    )

    assert res.uncertainty["mean_spread"] > 10.0


def test_missing_observations_safety():
    # When no observations are available, verification must be None without raising errors
    res = run_forecast_pipeline(
        mode="synthetic",
        variable="rainfall",
        lead_time=24,
        forecast_df=pd.DataFrame({
            "timestamp": ["2026-01-01T12:00:00Z"] * 2,
            "latitude": [20.0] * 2,
            "longitude": [75.0] * 2,
            "lead_time": [24] * 2,
            "variable": ["rainfall"] * 2,
            "model_name": ["m1", "m2"],
            "forecast_value": [10.0, 12.0],
        }),
        observation_df=pd.DataFrame(),
    )

    assert res.verification is None
    assert res.forecast_count == 1
