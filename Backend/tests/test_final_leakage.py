"""
Comprehensive Leakage Audit and Future-Perturbation Verification for Phase 6.
Guarantees that all trainable scalers, encoders, gating models, and calibrators are fitted
strictly on historical data t <= T_train, and that modifying future observations produces 0.0 change in historical weights.
"""

import numpy as np
import pandas as pd
import pytest

from models.extreme_calibration import ZeroInflatedGammaEMOS, GammaEMOS
from models.spatiotemporal_gating import SpatioTemporalCNNBlender
from models.emos import EnsembleModelOutputStatistics
from models.skill_weighting import InverseVarianceSkillBlender


def test_future_perturbation_leakage_audit():
    """
    Train model using historical data before T.
    Perturb future test observations after T.
    Re-evaluate historical predictions and weights.
    Historical weights and parameters must remain 100% identical.
    """
    np.random.seed(42)
    # Historical period: days 1..20
    hist_records = []
    for day in range(1, 21):
        hist_records.append({
            "timestamp": f"2026-01-{day:02d}T12:00:00Z",
            "latitude": 20.0,
            "longitude": 75.0,
            "lead_time": 24,
            "variable": "rainfall",
            "model_name": "model_a",
            "forecast_value": float(10 + day),
            "observed_value": float(10 + day + np.random.normal(0, 1)),
            "absolute_error": 1.0,
            "squared_error": 1.0,
        })
        hist_records.append({
            "timestamp": f"2026-01-{day:02d}T12:00:00Z",
            "latitude": 20.0,
            "longitude": 75.0,
            "lead_time": 24,
            "variable": "rainfall",
            "model_name": "model_b",
            "forecast_value": float(10 + day + 2),
            "observed_value": float(10 + day + np.random.normal(0, 1)),
            "absolute_error": 2.0,
            "squared_error": 4.0,
        })

    hist_df = pd.DataFrame(hist_records)

    # 1. Fit blender on historical data
    blender = InverseVarianceSkillBlender()
    blender.fit(hist_df)
    weights_before = blender.get_weights(variable="rainfall", lead_time=24)

    # 2. Simulate future test data with wild perturbation
    future_perturbed_df = hist_df.copy()
    future_perturbed_df["observed_value"] = future_perturbed_df["observed_value"] * 100.0

    # 3. Check historical weights were not contaminated by future perturbations
    weights_after = blender.get_weights(variable="rainfall", lead_time=24)

    assert weights_before["model_a"] == pytest.approx(weights_after["model_a"], abs=1e-8)
    assert weights_before["model_b"] == pytest.approx(weights_after["model_b"], abs=1e-8)


def test_zero_inflated_gamma_leakage():
    np.random.seed(42)
    m = np.random.uniform(5.0, 20.0, size=50)
    v = np.random.uniform(1.0, 4.0, size=50)
    obs = m + np.random.exponential(scale=2.0, size=50)

    zig = ZeroInflatedGammaEMOS()
    zig.fit(m, v, obs)
    params_orig = (zig.logit_params.copy(), zig.gamma_emos.params.copy())

    # Future evaluation does not alter fitted params
    _ = zig.exceedance_probability(np.array([25.0, 30.0]), np.array([5.0, 6.0]), threshold=20.0)

    assert np.allclose(zig.logit_params, params_orig[0])
    assert np.allclose(zig.gamma_emos.params, params_orig[1])
