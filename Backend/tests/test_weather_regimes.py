"""
Automated unit tests for atmospheric feature extraction and rule-based weather regime classification.
"""

import pytest
import numpy as np
import pandas as pd

from configs import get_config
from preprocessing.weather_regimes import (
    extract_atmospheric_features,
    classify_rule_based_regime,
    RuleBasedRegimeClassifier,
    REGIME_TO_ID,
    ID_TO_REGIME,
)


def test_extract_atmospheric_features_cyclical_time():
    """Verify hour/month/day of year extraction and cyclical sine/cosine transforms."""
    df = pd.DataFrame({
        "timestamp": ["2026-06-15T00:00:00Z", "2026-06-15T06:00:00Z", "2026-06-15T12:00:00Z"],
        "temperature": [20.0, 22.0, 28.0],
    })
    feat_df = extract_atmospheric_features(df)

    assert "hour" in feat_df.columns
    assert "diurnal_sin" in feat_df.columns
    assert "diurnal_cos" in feat_df.columns
    assert "seasonal_sin" in feat_df.columns
    assert "seasonal_cos" in feat_df.columns

    # Check diurnal sin/cos at 00:00 and 06:00
    assert np.isclose(feat_df.loc[0, "diurnal_sin"], 0.0, atol=1e-4)
    assert np.isclose(feat_df.loc[0, "diurnal_cos"], 1.0, atol=1e-4)
    assert np.isclose(feat_df.loc[1, "diurnal_sin"], 1.0, atol=1e-4)
    assert np.isclose(feat_df.loc[1, "diurnal_cos"], 0.0, atol=1e-4)


def test_rule_based_regime_classification_heavy_rain():
    """Verify heavy rain threshold triggers 'heavy_rain' regime."""
    df = pd.DataFrame({
        "timestamp": ["2026-07-01T12:00:00Z"],
        "rainfall": [65.0],
        "temperature": [24.0],
        "wind_speed": [5.0],
    })
    regime = classify_rule_based_regime(df.iloc[0])
    assert regime == "heavy_rain"


def test_rule_based_regime_classification_convective():
    """Verify moderate rain with high temperature or humidity triggers 'convective' regime."""
    df = pd.DataFrame({
        "timestamp": ["2026-07-01T12:00:00Z"],
        "rainfall": [15.0],
        "temperature": [30.0],
        "wind_speed": [4.0],
    })
    regime = classify_rule_based_regime(df.iloc[0])
    assert regime == "convective"


def test_rule_based_regime_classification_high_wind():
    """Verify wind above threshold triggers 'high_wind' regime."""
    df = pd.DataFrame({
        "timestamp": ["2026-07-01T12:00:00Z"],
        "rainfall": [0.0],
        "temperature": [20.0],
        "wind_speed": [18.0],
    })
    regime = classify_rule_based_regime(df.iloc[0])
    assert regime == "high_wind"


def test_rule_based_regime_classification_temperature_extremes():
    """Verify hot and cold extremes trigger their respective regimes."""
    df_hot = pd.DataFrame({
        "timestamp": ["2026-07-01T12:00:00Z"],
        "rainfall": [0.0],
        "temperature": [42.0],
        "wind_speed": [2.0],
    })
    assert classify_rule_based_regime(df_hot.iloc[0]) == "hot_extreme"

    df_cold = pd.DataFrame({
        "timestamp": ["2026-01-01T12:00:00Z"],
        "rainfall": [0.0],
        "temperature": [-5.0],
        "wind_speed": [2.0],
    })
    assert classify_rule_based_regime(df_cold.iloc[0]) == "cold_extreme"


def test_rule_based_regime_priority_order():
    """
    Verify deterministic priority when multiple extreme conditions are satisfied simultaneously:
    heavy_rain takes precedence over high_wind and hot_extreme.
    """
    df_compound = pd.DataFrame({
        "timestamp": ["2026-07-01T12:00:00Z"],
        "rainfall": [75.0],
        "wind_speed": [25.0],
        "temperature": [38.0],
    })
    regime = classify_rule_based_regime(df_compound.iloc[0])
    assert regime == "heavy_rain"


def test_rule_based_regime_classifier_dataframe_pipeline():
    """Verify RuleBasedRegimeClassifier transforms a full DataFrame and appends regime_id."""
    df = pd.DataFrame({
        "timestamp": ["2026-07-01T00:00:00Z", "2026-07-01T06:00:00Z", "2026-07-01T12:00:00Z"],
        "rainfall": [0.0, 60.0, 0.0],
        "wind_speed": [3.0, 10.0, 20.0],
        "temperature": [22.0, 24.0, 25.0],
    })
    clf = RuleBasedRegimeClassifier()
    res = clf.classify(df)

    assert "weather_regime" in res.columns
    assert "regime_id" in res.columns
    assert res.loc[0, "weather_regime"] == "stable"
    assert res.loc[1, "weather_regime"] == "heavy_rain"
    assert res.loc[2, "weather_regime"] == "high_wind"
    assert res.loc[1, "regime_id"] == REGIME_TO_ID["heavy_rain"]
    assert res.loc[0, "regime_confidence"] is None or pd.isna(res.loc[0, "regime_confidence"])
