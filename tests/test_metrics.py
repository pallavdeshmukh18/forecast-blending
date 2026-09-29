"""
Unit tests for meteorological verification metrics.
"""

import numpy as np
import pytest

from evaluation.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    mean_bias_error,
    pearson_correlation,
    compute_all_metrics,
)


def test_mean_absolute_error():
    obs = np.array([10.0, 20.0, 30.0])
    pred = np.array([12.0, 18.0, 33.0])
    # |2| + |-2| + |3| = 7 / 3 = 2.3333...
    assert pytest.approx(mean_absolute_error(obs, pred), 0.001) == 7.0 / 3.0


def test_root_mean_squared_error():
    obs = np.array([0.0, 3.0])
    pred = np.array([4.0, 0.0])
    # 4^2 + (-3)^2 = 16 + 9 = 25 -> mean = 12.5 -> sqrt(12.5) = 3.5355...
    assert pytest.approx(root_mean_squared_error(obs, pred), 0.001) == np.sqrt(12.5)


def test_mean_bias_error():
    obs = np.array([10.0, 20.0, 30.0])
    pred = np.array([12.0, 22.0, 32.0])
    # (+2 + 2 + 2) / 3 = +2.0
    assert pytest.approx(mean_bias_error(obs, pred), 0.001) == 2.0


def test_pearson_correlation_perfect():
    obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    pred = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
    assert pytest.approx(pearson_correlation(obs, pred), 0.001) == 1.0


def test_pearson_correlation_inverse():
    obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    pred = np.array([10.0, 8.0, 6.0, 4.0, 2.0])
    assert pytest.approx(pearson_correlation(obs, pred), 0.001) == -1.0


def test_pearson_correlation_constant():
    """Test constant arrays where standard deviation is zero."""
    obs = np.array([5.0, 5.0, 5.0])
    pred = np.array([5.0, 5.0, 5.0])
    assert pearson_correlation(obs, pred) == 1.0


def test_metrics_mismatched_length():
    with pytest.raises(ValueError, match="equal length"):
        mean_absolute_error([1, 2], [1, 2, 3])


def test_compute_all_metrics_dictionary():
    obs = [10.0, 20.0, 30.0]
    pred = [10.0, 20.0, 30.0]
    res = compute_all_metrics(obs, pred)
    assert res["mae"] == 0.0
    assert res["rmse"] == 0.0
    assert res["bias"] == 0.0
    assert res["correlation"] == 1.0
    assert res["sample_count"] == 3
