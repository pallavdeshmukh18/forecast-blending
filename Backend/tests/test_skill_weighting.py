"""
Unit and integration tests for Skill-Weighted Blenders (Inverse-Variance, Inverse-MAE).
Tests mathematical properties, weight normalization, smoothing, hierarchical fallback,
and missing model handling.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from configs import get_config
from models.skill_weighting import SkillWeightedBlender, InverseVarianceSkillBlender, InverseMAESkillBlender


@pytest.fixture
def sample_verified_history():
    """Create controlled verified dataset with known model error variances."""
    records = []
    base_time = datetime(2026, 1, 1, 0, 0)

    # Model A: High error on plains (sigma=2.0), low error on coastal (sigma=0.5)
    # Model B: Low error on plains (sigma=0.5), high error on coastal (sigma=2.0)
    # Model C: Moderate error everywhere (sigma=1.0)
    rng = np.random.default_rng(42)

    for i in range(30):
        t = base_time + timedelta(hours=6 * i)
        for lt in [6, 24]:
            # Coastal
            records.append({"timestamp": t, "model_name": "model_a", "variable": "temperature", "lead_time": lt, "region": "coastal", "forecast_value": 25.0 + rng.normal(0, 0.5), "observed_value": 25.0})
            records.append({"timestamp": t, "model_name": "model_b", "variable": "temperature", "lead_time": lt, "region": "coastal", "forecast_value": 25.0 + rng.normal(0, 2.0), "observed_value": 25.0})
            records.append({"timestamp": t, "model_name": "model_c", "variable": "temperature", "lead_time": lt, "region": "coastal", "forecast_value": 25.0 + rng.normal(0, 1.0), "observed_value": 25.0})

            # Plains
            records.append({"timestamp": t, "model_name": "model_a", "variable": "temperature", "lead_time": lt, "region": "plains", "forecast_value": 30.0 + rng.normal(0, 2.0), "observed_value": 30.0})
            records.append({"timestamp": t, "model_name": "model_b", "variable": "temperature", "lead_time": lt, "region": "plains", "forecast_value": 30.0 + rng.normal(0, 0.5), "observed_value": 30.0})
            records.append({"timestamp": t, "model_name": "model_c", "variable": "temperature", "lead_time": lt, "region": "plains", "forecast_value": 30.0 + rng.normal(0, 1.0), "observed_value": 30.0})

    df = pd.DataFrame(records)
    df["error"] = df["forecast_value"] - df["observed_value"]
    df["forecast_error"] = df["error"]
    return df


def test_inverse_variance_weights_sum_to_one(sample_verified_history):
    blender = InverseVarianceSkillBlender(weight_smoothing=0.0, minimum_samples=10)
    blender.fit_verification_data(sample_verified_history)

    weights_coastal = blender.get_weights(variable="temperature", lead_time=6, region="coastal")
    assert len(weights_coastal) == 3
    assert abs(sum(weights_coastal.values()) - 1.0) < 1e-6
    for w in weights_coastal.values():
        assert 0.0 <= w <= 1.0


def test_inverse_variance_region_conditioning(sample_verified_history):
    """Verify Model A has higher weight in coastal and Model B has higher weight in plains."""
    blender = InverseVarianceSkillBlender(weight_smoothing=0.0, minimum_samples=10)
    blender.fit_verification_data(sample_verified_history)

    w_coastal = blender.get_weights(variable="temperature", lead_time=24, region="coastal")
    w_plains = blender.get_weights(variable="temperature", lead_time=24, region="plains")

    # In coastal, Model A (sigma=0.5) must have higher weight than Model B (sigma=2.0)
    assert w_coastal["model_a"] > w_coastal["model_b"]
    # In plains, Model B (sigma=0.5) must have higher weight than Model A (sigma=2.0)
    assert w_plains["model_b"] > w_plains["model_a"]


def test_inverse_mae_weights(sample_verified_history):
    blender = InverseMAESkillBlender(weight_smoothing=0.0, minimum_samples=10)
    blender.fit_verification_data(sample_verified_history)

    weights = blender.get_weights(variable="temperature", lead_time=6, region="coastal")
    assert abs(sum(weights.values()) - 1.0) < 1e-6
    assert weights["model_a"] > weights["model_b"]


def test_weight_smoothing_regularization(sample_verified_history):
    """Smoothing alpha should shrink weights toward equal weight (1/3 = ~0.333)."""
    blender_raw = InverseVarianceSkillBlender(weight_smoothing=0.0, minimum_samples=10)
    blender_raw.fit_verification_data(sample_verified_history)
    w_raw = blender_raw.get_weights(variable="temperature", lead_time=6, region="coastal")

    blender_smooth = InverseVarianceSkillBlender(weight_smoothing=0.50, minimum_samples=10)
    blender_smooth.fit_verification_data(sample_verified_history)
    w_smooth = blender_smooth.get_weights(variable="temperature", lead_time=6, region="coastal")

    # The dominant weight should be lower under smoothing
    max_raw_model = max(w_raw, key=w_raw.get)
    assert w_smooth[max_raw_model] < w_raw[max_raw_model]
    # Smoothed weights must still sum to 1.0
    assert abs(sum(w_smooth.values()) - 1.0) < 1e-6


def test_hierarchical_fallback_sparse_data():
    """Verify fallback from scope 4 -> scope 3 -> scope 2 -> scope 1 when data is sparse."""
    # Create dataset with only 3 samples for mountain region at 72h
    records = []
    base_time = datetime(2026, 1, 1, 0, 0)
    for i in range(3):
        t = base_time + timedelta(hours=6 * i)
        records.append({"timestamp": t, "model_name": "model_a", "variable": "temperature", "lead_time": 72, "region": "mountain", "forecast_value": 15.0, "observed_value": 15.0, "error": 0.1})
        records.append({"timestamp": t, "model_name": "model_b", "variable": "temperature", "lead_time": 72, "region": "mountain", "forecast_value": 15.0, "observed_value": 15.0, "error": 0.2})

    # Add 25 samples for global temperature
    for i in range(25):
        t = base_time + timedelta(hours=6 * i)
        records.append({"timestamp": t, "model_name": "model_a", "variable": "temperature", "lead_time": 24, "region": "plains", "forecast_value": 20.0, "observed_value": 20.0, "error": 0.5})
        records.append({"timestamp": t, "model_name": "model_b", "variable": "temperature", "lead_time": 24, "region": "plains", "forecast_value": 20.0, "observed_value": 20.0, "error": 1.0})

    df = pd.DataFrame(records)
    blender = InverseVarianceSkillBlender(minimum_samples=10)
    blender.fit_verification_data(df)

    # For mountain at 72h, only 3 samples exist (< minimum 10)
    meta = blender.get_weight_metadata(variable="temperature", lead_time=72, region="mountain")
    # Should fall back to model_variable scope (which has 3 + 25 = 28 samples >= 10)
    assert meta["models"]["model_a"]["active_skill_scope"] == "model_variable"
    assert meta["models"]["model_a"]["historical_sample_count"] == 28


def test_missing_model_renormalization(sample_verified_history):
    """When a model is missing at inference time, remaining weights must renormalize to 1.0."""
    blender = InverseVarianceSkillBlender(minimum_samples=10)
    blender.fit_verification_data(sample_verified_history)

    # Ask for weights when model_c is missing
    w_avail = blender.get_weights(
        variable="temperature",
        lead_time=6,
        region="coastal",
        available_models=["model_a", "model_b"],
    )
    assert len(w_avail) == 2
    assert "model_c" not in w_avail
    assert abs(sum(w_avail.values()) - 1.0) < 1e-6


def test_predict_produces_uncertainty_spread(sample_verified_history):
    blender = InverseVarianceSkillBlender(minimum_samples=10)
    blender.fit_verification_data(sample_verified_history)

    # Create input forecast dataframe with 3 models
    fcs = pd.DataFrame([
        {"timestamp": datetime(2026, 1, 10, 0, 0), "run_time": datetime(2026, 1, 9, 0, 0), "latitude": 18.95, "longitude": 72.82, "lead_time": 24, "model_name": "model_a", "variable": "temperature", "forecast_value": 24.0, "region": "coastal"},
        {"timestamp": datetime(2026, 1, 10, 0, 0), "run_time": datetime(2026, 1, 9, 0, 0), "latitude": 18.95, "longitude": 72.82, "lead_time": 24, "model_name": "model_b", "variable": "temperature", "forecast_value": 26.0, "region": "coastal"},
        {"timestamp": datetime(2026, 1, 10, 0, 0), "run_time": datetime(2026, 1, 9, 0, 0), "latitude": 18.95, "longitude": 72.82, "lead_time": 24, "model_name": "model_c", "variable": "temperature", "forecast_value": 25.0, "region": "coastal"},
    ])

    pred = blender.predict(fcs)
    assert len(pred) == 1
    assert "forecast_spread" in pred.columns
    assert "historical_residual_std" in pred.columns
    assert pred["forecast_spread"].iloc[0] > 0.0
    assert 24.0 <= pred["forecast_value"].iloc[0] <= 26.0
