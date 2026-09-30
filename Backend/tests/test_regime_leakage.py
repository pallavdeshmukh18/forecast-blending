"""
Automated unit tests for Temporal Isolation and Data Leakage Prevention in Weather Regimes & Gating.
"""

import pytest
import numpy as np
import pandas as pd

from preprocessing.regime_clustering import KMeansRegimeClassifier
from models.regime_blending import RegimeInverseVarianceBlender
from models.gating_model import WeatherRegimeGatingNetwork


@pytest.fixture
def chronological_split_data():
    """Create chronological dataset divided cleanly into historical (train) and future (test)."""
    rng = np.random.default_rng(42)
    n_train = 60
    n_test = 30

    train_dates = pd.date_range("2026-01-01", periods=n_train, freq="6h", tz="UTC")
    test_dates = pd.date_range("2026-01-16", periods=n_test, freq="6h", tz="UTC")

    # Historical data (temperatures around 20 C)
    df_train = pd.DataFrame({
        "timestamp": train_dates,
        "temperature": rng.normal(20, 2, n_train),
        "rainfall": rng.exponential(5, n_train),
        "wind_speed": rng.normal(5, 1, n_train),
        "latitude": 28.5,
        "longitude": 77.2,
    })

    # Future test data
    df_test = pd.DataFrame({
        "timestamp": test_dates,
        "temperature": rng.normal(22, 2, n_test),
        "rainfall": rng.exponential(6, n_test),
        "wind_speed": rng.normal(6, 1, n_test),
        "latitude": 28.5,
        "longitude": 77.2,
    })

    return df_train, df_test


def test_kmeans_scaler_fitted_only_on_train_period(chronological_split_data):
    """
    Verify KMeans StandardScaler is fitted exclusively on historical training data.
    Perturbing future test observations drastically must NOT change training scaler mean/variance.
    """
    df_train, df_test = chronological_split_data

    clf = KMeansRegimeClassifier(n_clusters=3, random_state=42)
    clf.fit(df_train)

    train_mean_before = clf.scaler.mean_.copy()
    train_scale_before = clf.scaler.scale_.copy()

    # Predict on unseen test data
    pred_test_1 = clf.predict(df_test)

    # Now simulate a massive perturbation in future data (e.g. +100 C temperature spike in test data)
    df_test_perturbed = df_test.copy()
    df_test_perturbed["temperature"] = df_test_perturbed["temperature"] + 100.0

    # Predict on perturbed future data
    pred_test_2 = clf.predict(df_test_perturbed)

    # Verify training scaler statistics were completely untouched
    assert np.allclose(clf.scaler.mean_, train_mean_before)
    assert np.allclose(clf.scaler.scale_, train_scale_before)


def test_future_observation_perturbation_regime_weights(chronological_split_data):
    """
    Verify historical regime skill weights at T_0 are invariant to changes or additions in future data.
    """
    df_train, _ = chronological_split_data

    # Create verification records for train period
    records = []
    for _, row in df_train.iterrows():
        records.append({
            "model_name": "model_a",
            "timestamp": row["timestamp"],
            "variable": "temperature",
            "lead_time": 24,
            "region": "north",
            "weather_regime": "stable",
            "forecast_value": 20.0,
            "observed_value": 20.0,
            "error": 0.5,
        })
        records.append({
            "model_name": "model_b",
            "timestamp": row["timestamp"],
            "variable": "temperature",
            "lead_time": 24,
            "region": "north",
            "weather_regime": "stable",
            "forecast_value": 25.0,
            "observed_value": 20.0,
            "error": 5.0,
        })
    verified_train = pd.DataFrame(records)

    blender = RegimeInverseVarianceBlender(minimum_samples=5)
    blender.fit_verification_data(verified_train)

    w_before = blender.get_weights(variable="temperature", lead_time=24, region="north", weather_regime="stable")

    # Add a massive future error batch
    future_records = [
        {
            "model_name": "model_a",
            "timestamp": "2026-02-01T00:00:00Z",
            "variable": "temperature",
            "lead_time": 24,
            "region": "north",
            "weather_regime": "stable",
            "forecast_value": 999.0,
            "observed_value": 20.0,
            "error": 979.0,
        }
    ]

    # Re-evaluating historical weights with only historical training data must yield identical weights
    blender_isolated = RegimeInverseVarianceBlender(minimum_samples=5)
    blender_isolated.fit_verification_data(verified_train)
    w_after = blender_isolated.get_weights(variable="temperature", lead_time=24, region="north", weather_regime="stable")

    assert np.isclose(w_before["model_a"], w_after["model_a"])
    assert np.isclose(w_before["model_b"], w_after["model_b"])
