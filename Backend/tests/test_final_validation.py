"""
Unit tests for Final Multi-Dimensional Holdout Validation and Statistical Comparison.
"""

import numpy as np
import pandas as pd
import pytest

from evaluation.final_validation import MultiDimensionalHoldoutValidator
from evaluation.model_comparison import (
    compute_effective_sample_size,
    block_bootstrap_ci,
    evaluate_paired_model_comparison,
)


def test_chronological_split():
    validator = MultiDimensionalHoldoutValidator()
    dates = pd.date_range("2026-01-01", periods=20, freq="D").astype(str).tolist()
    df = pd.DataFrame({
        "timestamp": dates,
        "observed_value": np.random.randn(20),
    })

    train_df, val_df, test_df = validator.split_chronological(df, train_frac=0.6, val_frac=0.2)

    assert len(train_df) == 12
    assert len(val_df) == 4
    assert len(test_df) == 4

    # Verify temporal order is strictly preserved
    assert max(train_df["timestamp"]) < min(val_df["timestamp"])
    assert max(val_df["timestamp"]) < min(test_df["timestamp"])


def test_block_bootstrap_confidence_interval():
    np.random.seed(42)
    # Differences where candidate model is consistently better (negative difference)
    diffs = np.random.normal(loc=-1.5, scale=0.5, size=100)
    mean_diff, ci_low, ci_high = block_bootstrap_ci(diffs, block_size=4, n_bootstraps=500)

    assert ci_low <= mean_diff <= ci_high
    assert ci_high < 0.0  # Confirms candidate model is statistically superior


def test_effective_sample_size_autocorrelation():
    # Independent sequence
    np.random.seed(42)
    iid_series = np.random.randn(100)
    n_eff_iid = compute_effective_sample_size(iid_series)
    assert 70.0 <= n_eff_iid <= 100.0

    # Highly autocorrelated sequence
    ar_series = np.zeros(100)
    for t in range(1, 100):
        ar_series[t] = 0.85 * ar_series[t - 1] + np.random.randn()
    n_eff_ar = compute_effective_sample_size(ar_series)
    assert n_eff_ar < n_eff_iid


def test_extreme_contingency_metrics():
    validator = MultiDimensionalHoldoutValidator()
    obs = np.array([5.0, 15.0, 30.0, 45.0, 0.0, 12.0, 60.0])
    probs = np.array([0.1, 0.3, 0.8, 0.9, 0.05, 0.2, 0.95])
    threshold = 25.0

    metrics = validator.evaluate_extreme_contingency(obs, probs, threshold=threshold, prob_threshold=0.5)

    assert metrics["true_positives"] == 3
    assert metrics["false_positives"] == 0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1_score"] == 1.0
    assert metrics["csi_threat_score"] == 1.0
