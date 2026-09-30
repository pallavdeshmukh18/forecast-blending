"""
Tests for Chronological Walk-Forward Validation and Temporal Leakage Prevention.
Verifies that model skill estimation strictly avoids future data leakage.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from evaluation.walk_forward import (
    WalkForwardBlenderEvaluator,
    compute_paired_bootstrap_ci,
    compute_relative_improvements,
)
from models.skill_weighting import InverseVarianceSkillBlender
from models.baseline import EqualWeightBlender


def test_temporal_leakage_safeguard():
    """
    Critical Test: Ensure future observations at/after time T cannot influence weights at time T.
    Construct history where:
    - In Period 1 (T1..T10): Model A is accurate (error ~ 0), Model B has high error (~ 5).
    - In Period 2 (T11..T20): Future observations reverse this (Model A error ~ 10, Model B error ~ 0).
    Verify that walk-forward forecast at T10 strictly uses Period 1 errors (assigning Model A high weight).
    """
    base_time = datetime(2026, 1, 1, 0, 0)
    records_fcs = []
    records_obs = []

    for i in range(20):
        t = base_time + timedelta(hours=6 * i)
        obs_val = 20.0

        if i < 10:
            # Period 1: Model A is perfect, Model B is bad
            fcs_a = 20.0
            fcs_b = 25.0
        else:
            # Period 2: Future reversal! Model A is terrible, Model B is perfect
            fcs_a = 30.0
            fcs_b = 20.0

        records_obs.append({"timestamp": t, "latitude": 20.0, "longitude": 75.0, "variable": "temperature", "observed_value": obs_val, "region": "plains"})
        records_fcs.append({"timestamp": t, "run_time": t - timedelta(hours=24), "latitude": 20.0, "longitude": 75.0, "lead_time": 24, "model_name": "model_a", "variable": "temperature", "forecast_value": fcs_a, "region": "plains"})
        records_fcs.append({"timestamp": t, "run_time": t - timedelta(hours=24), "latitude": 20.0, "longitude": 75.0, "lead_time": 24, "model_name": "model_b", "variable": "temperature", "forecast_value": fcs_b, "region": "plains"})

    obs_df = pd.DataFrame(records_obs)
    fcs_df = pd.DataFrame(records_fcs)

    blender = InverseVarianceSkillBlender(weight_smoothing=0.0, minimum_samples=5)
    evaluator = WalkForwardBlenderEvaluator(
        blenders=[blender],
        mode="expanding",
        min_train_steps=5,
    )

    results = evaluator.run(fcs_df, obs_df)
    assert not results.empty

    # At step T9 (time_step 9, which is the last step of Period 1),
    # the blender forecast should have followed Model A (20.0) rather than the future Period 2
    t9 = pd.to_datetime(base_time + timedelta(hours=6 * 9), utc=True)
    res_t9 = results[(results["timestamp"] == t9) & (results["model_name"] == "inverse_variance_blend")]
    assert len(res_t9) == 1
    # Model A had 0 error in history, Model B had error 5 -> blend should be close to 20.0 (Model A)
    assert abs(res_t9["forecast_value"].iloc[0] - 20.0) < 0.2


def test_paired_bootstrap_confidence_interval():
    rng = np.random.default_rng(42)
    # Baseline has higher error than blend
    e_base = rng.normal(0, 2.0, size=100)
    e_blend = rng.normal(0, 1.0, size=100)

    res = compute_paired_bootstrap_ci(e_base, e_blend, metric="rmse", n_bootstrap=500)
    assert "mean_diff" in res
    assert "ci_lower" in res
    assert "ci_upper" in res
    assert "p_value_superiority" in res
    # Blend has lower RMSE, so diff = blend - base should be negative
    assert res["mean_diff"] < 0
    assert res["ci_upper"] < 0
    assert res["p_value_superiority"] > 0.95


def test_relative_improvements_calculation():
    comp_df = pd.DataFrame([
        {"variable": "temperature", "lead_time": 24, "model_name": "equal_weight_blend", "rmse": 2.0, "mae": 1.5, "correlation": 0.8},
        {"variable": "temperature", "lead_time": 24, "model_name": "inverse_variance_blend", "rmse": 1.6, "mae": 1.2, "correlation": 0.9},
    ])

    imp_df = compute_relative_improvements(comp_df, baseline_model="equal_weight_blend", metric="rmse")
    assert "rmse_rel_improvement_vs_equal_weight_blend" in imp_df.columns
    iv_row = imp_df[imp_df["model_name"] == "inverse_variance_blend"].iloc[0]
    # (2.0 - 1.6) / 2.0 * 100 = 20.0%
    assert abs(iv_row["rmse_rel_improvement_vs_equal_weight_blend"] - 20.0) < 1e-2
