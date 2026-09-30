"""
Unit tests for deterministic synthetic data generation.
"""

import pandas as pd
import numpy as np
from configs import get_config
from ingestion.synthetic import SyntheticDataGenerator


def test_synthetic_data_determinism():
    gen = SyntheticDataGenerator()
    obs1, fcast1 = gen.generate(seed=42)
    obs2, fcast2 = gen.generate(seed=42)

    pd.testing.assert_frame_equal(obs1, obs2)
    pd.testing.assert_frame_equal(fcast1, fcast2)


def test_synthetic_data_different_seed():
    gen = SyntheticDataGenerator()
    obs1, fcast1 = gen.generate(seed=42)
    obs2, fcast2 = gen.generate(seed=99)

    assert not obs1["observed_value"].equals(obs2["observed_value"])
    assert not fcast1["forecast_value"].equals(fcast2["forecast_value"])


def test_synthetic_data_schema_and_contents():
    cfg = get_config()
    gen = SyntheticDataGenerator(cfg)
    obs_df, forecasts_df = gen.generate(seed=42)

    assert not obs_df.empty
    assert not forecasts_df.empty

    # Verify models
    expected_models = set(cfg.models.keys())
    actual_models = set(forecasts_df["model_name"].unique())
    assert expected_models == actual_models

    # Verify variables
    expected_vars = set(cfg.variables.keys())
    assert expected_vars == set(obs_df["variable"].unique())
    assert expected_vars == set(forecasts_df["variable"].unique())

    # Verify lead times
    expected_lts = set(cfg.lead_times_hours)
    assert expected_lts == set(forecasts_df["lead_time"].unique())


def test_synthetic_model_error_profiles():
    """Verify that models exhibit designed error tendencies (e.g. Model A short-range vs long-range)."""
    gen = SyntheticDataGenerator()
    obs_df, forecasts_df = gen.generate(seed=42)

    # Merge forecasts and observations
    m = pd.merge(forecasts_df, obs_df, on=["timestamp", "latitude", "longitude", "variable"])
    m_temp = m[m["variable"] == "temperature"]

    # Model A error at 6h vs 120h
    a_short = m_temp[(m_temp["model_name"] == "nwp_model_a") & (m_temp["lead_time"] == 6)]
    a_long = m_temp[(m_temp["model_name"] == "nwp_model_a") & (m_temp["lead_time"] == 120)]

    mae_a_short = (a_short["forecast_value"] - a_short["observed_value"]).abs().mean()
    mae_a_long = (a_long["forecast_value"] - a_long["observed_value"]).abs().mean()

    # Error should grow with lead time
    assert mae_a_long > mae_a_short
