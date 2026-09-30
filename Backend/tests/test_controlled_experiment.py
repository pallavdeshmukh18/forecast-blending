"""
Critical Synthetic Experiment Test (Requirement 30).
Validates that differential error profiles across lead times and variables dynamically emerge:
- Short lead time (+6h) -> Model A receives higher weight
- Long lead time (+120h) -> Model B receives higher weight
- Rainfall variable -> Model C receives higher weight
All weights are derived dynamically from historical forecast-observation verification.
"""

import pytest
import numpy as np
import pandas as pd

from scripts.run_phase3_experiment import generate_controlled_experiment_data
from models.skill_weighting import InverseVarianceSkillBlender
from models.adaptive_blending import RidgeBlender


def test_controlled_differential_skill_emergence():
    """
    Run controlled experiment and verify that historical skill estimation dynamically discovers:
    1. Short horizons (+6h) favor Model A
    2. Extended horizons (+120h) favor Model B
    3. Convective rainfall favors Model C
    """
    obs_df, fcs_df = generate_controlled_experiment_data(seed=42)

    blender = InverseVarianceSkillBlender(weight_smoothing=0.05, minimum_samples=10)
    blender.fit(fcs_df, obs_df)

    # 1. Check Temperature @ +6h vs +120h
    w_temp_6h = blender.get_weights(variable="temperature", lead_time=6, region="plains")
    w_temp_120h = blender.get_weights(variable="temperature", lead_time=120, region="plains")

    # At +6h, Model A should have significantly higher weight than Model B
    assert w_temp_6h["nwp_model_a"] > w_temp_6h["nwp_model_b"]

    # At +120h, Model B should have significantly higher weight than Model A
    assert w_temp_120h["nwp_model_b"] > w_temp_120h["nwp_model_a"]

    # 2. Check Rainfall Variable vs Model Specialization
    w_rain = blender.get_weights(variable="rainfall", lead_time=24, region="plains")
    # AI Model C has superior precipitation skill, so it should receive higher weight than Model A & B
    assert w_rain["ai_model_c"] > w_rain["nwp_model_a"]
    assert w_rain["ai_model_c"] > w_rain["nwp_model_b"]


def test_ridge_controlled_differential_skill():
    """Verify that Constrained Ridge Blender also allocates weights reflecting differential skill."""
    obs_df, fcs_df = generate_controlled_experiment_data(seed=42)

    blender = RidgeBlender(alpha=0.1, fit_intercept=True, constrained_weights=True, minimum_samples=10)
    blender.fit(fcs_df, obs_df)

    w_temp_6h = blender.get_weights(variable="temperature", lead_time=6, region="plains")
    w_temp_120h = blender.get_weights(variable="temperature", lead_time=120, region="plains")
    w_rain = blender.get_weights(variable="rainfall", lead_time=24, region="plains")

    # Model A higher at 6h, Model B higher at 120h, Model C higher on rain
    assert w_temp_6h["nwp_model_a"] > w_temp_6h["nwp_model_b"]
    assert w_temp_120h["nwp_model_b"] > w_temp_120h["nwp_model_a"]
    assert w_rain["ai_model_c"] > w_rain["nwp_model_a"]
