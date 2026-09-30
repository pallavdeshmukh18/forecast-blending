"""
Unit and integration tests for Adaptive Ridge Blending.
Tests Mode A (unconstrained Ridge) and Mode B (constrained simplex w >= 0, sum(w) = 1),
intercept handling, and missing model renormalization.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from models.adaptive_blending import RidgeBlender


@pytest.fixture
def synthetic_linear_history():
    """Create verification dataset where True = 0.6 * ModelA + 0.4 * ModelB + 0.0 * ModelC + 1.5."""
    rng = np.random.default_rng(123)
    records = []
    base_time = datetime(2026, 1, 1, 0, 0)

    for i in range(40):
        t = base_time + timedelta(hours=6 * i)
        val_a = 20.0 + rng.normal(0, 2.0)
        val_b = 20.0 + rng.normal(0, 2.0)
        val_c = 20.0 + rng.normal(0, 5.0)  # Uninformative noisy model

        # Ground truth observation
        obs_val = 1.5 + 0.60 * val_a + 0.40 * val_b + rng.normal(0, 0.2)

        records.append({"timestamp": t, "model_name": "model_a", "variable": "temperature", "lead_time": 24, "region": "plains", "forecast_value": val_a, "observed_value": obs_val, "error": val_a - obs_val})
        records.append({"timestamp": t, "model_name": "model_b", "variable": "temperature", "lead_time": 24, "region": "plains", "forecast_value": val_b, "observed_value": obs_val, "error": val_b - obs_val})
        records.append({"timestamp": t, "model_name": "model_c", "variable": "temperature", "lead_time": 24, "region": "plains", "forecast_value": val_c, "observed_value": obs_val, "error": val_c - obs_val})

    return pd.DataFrame(records)


def test_ridge_constrained_simplex_weights(synthetic_linear_history):
    """Mode B: strictly non-negative weights summing to 1.0."""
    blender = RidgeBlender(alpha=0.1, fit_intercept=True, constrained_weights=True, minimum_samples=10)
    blender.fit_verification_data(synthetic_linear_history)

    weights = blender.get_weights(variable="temperature", lead_time=24, region="plains")
    assert len(weights) == 3
    # Check weights sum to 1.0
    assert abs(sum(weights.values()) - 1.0) < 1e-4
    # Check all weights non-negative
    for m, w in weights.items():
        assert w >= -1e-6
    # Model C (noise) should receive lowest weight
    assert weights["model_a"] > weights["model_c"]
    assert weights["model_b"] > weights["model_c"]


def test_ridge_unconstrained_mode(synthetic_linear_history):
    """Mode A: standard unconstrained Ridge regression."""
    blender = RidgeBlender(alpha=0.1, fit_intercept=True, constrained_weights=False, minimum_samples=10)
    blender.fit_verification_data(synthetic_linear_history)

    meta = blender.get_weight_metadata(variable="temperature", lead_time=24, region="plains")
    assert "intercept" in meta
    assert "weights" in meta
    assert len(meta["weights"]) == 3
    # Intercept should be estimated near 1.5
    assert abs(meta["intercept"] - 1.5) < 3.0


def test_ridge_no_intercept(synthetic_linear_history):
    """Test Ridge regression with fit_intercept=False."""
    blender = RidgeBlender(alpha=0.1, fit_intercept=False, constrained_weights=True, minimum_samples=10)
    blender.fit_verification_data(synthetic_linear_history)

    meta = blender.get_weight_metadata(variable="temperature", lead_time=24, region="plains")
    assert meta["intercept"] == 0.0


def test_ridge_predict_pipeline(synthetic_linear_history):
    blender = RidgeBlender(alpha=0.1, fit_intercept=True, constrained_weights=True, minimum_samples=10)
    blender.fit_verification_data(synthetic_linear_history)

    test_fcs = pd.DataFrame([
        {"timestamp": datetime(2026, 1, 15, 0, 0), "run_time": datetime(2026, 1, 14, 0, 0), "latitude": 28.61, "longitude": 77.20, "lead_time": 24, "model_name": "model_a", "variable": "temperature", "forecast_value": 22.0, "region": "plains"},
        {"timestamp": datetime(2026, 1, 15, 0, 0), "run_time": datetime(2026, 1, 14, 0, 0), "latitude": 28.61, "longitude": 77.20, "lead_time": 24, "model_name": "model_b", "variable": "temperature", "forecast_value": 23.0, "region": "plains"},
        {"timestamp": datetime(2026, 1, 15, 0, 0), "run_time": datetime(2026, 1, 14, 0, 0), "latitude": 28.61, "longitude": 77.20, "lead_time": 24, "model_name": "model_c", "variable": "temperature", "forecast_value": 21.0, "region": "plains"},
    ])

    pred = blender.predict(test_fcs)
    assert len(pred) == 1
    assert "forecast_value" in pred.columns
    assert "historical_residual_std" in pred.columns
    assert 20.0 <= pred["forecast_value"].iloc[0] <= 26.0
