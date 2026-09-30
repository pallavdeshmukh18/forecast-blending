"""
Tests for Probability Calibration & Reliability Analysis (Phase 5).
"""

import numpy as np
import pytest

from evaluation.calibration import (
    brier_score,
    brier_skill_score,
    compute_reliability_curve,
    evaluate_event_calibration,
)
from models.probabilistic import ProbabilisticForecast


def test_brier_score():
    obs = np.array([1, 1, 0, 0])
    # Perfect forecast
    p_perf = np.array([1.0, 1.0, 0.0, 0.0])
    assert brier_score(obs, p_perf) == 0.0

    # Worst forecast
    p_worst = np.array([0.0, 0.0, 1.0, 1.0])
    assert brier_score(obs, p_worst) == 1.0

    # Skill score
    assert brier_skill_score(0.1, 0.2) == 0.5


def test_compute_reliability_curve():
    obs = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    probs = np.array([0.05, 0.08, 0.12, 0.15, 0.85, 0.88, 0.92, 0.95])

    diag = compute_reliability_curve(obs, probs, n_bins=5)
    assert "bin_edges" in diag
    assert "observed_frequencies" in diag
    assert diag["brier_score"] < 0.05
    assert diag["sample_count"] == 8


def test_evaluate_event_calibration():
    obs_values = np.array([10.0, 20.0, 30.0, 40.0])
    pf = ProbabilisticForecast(
        mean=np.array([10.0, 20.0, 30.0, 40.0]),
        std=np.array([2.0, 2.0, 2.0, 2.0]),
    )
    diag = evaluate_event_calibration(obs_values, pf, threshold=25.0, operator=">=")
    assert diag["threshold"] == 25.0
    assert diag["sample_count"] == 4
