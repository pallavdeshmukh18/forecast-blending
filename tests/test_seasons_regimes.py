"""
Unit tests for climatological seasons and weather regime classification.
"""

import pandas as pd
from preprocessing.features import add_season_feature, add_temporal_features
from preprocessing.regimes import WeatherRegimeClassifier


def test_add_season_feature():
    df = pd.DataFrame({
        "timestamp": [
            "2026-01-15T12:00:00Z",  # Jan -> winter
            "2026-04-15T12:00:00Z",  # Apr -> spring
            "2026-07-15T12:00:00Z",  # Jul -> summer
            "2026-10-15T12:00:00Z",  # Oct -> autumn
            "2026-12-15T12:00:00Z",  # Dec -> winter
        ]
    })
    res = add_season_feature(df)
    assert list(res["season"]) == ["winter", "spring", "summer", "autumn", "winter"]


def test_add_temporal_features():
    df = pd.DataFrame({
        "timestamp": ["2026-03-15T06:00:00Z"]
    })
    res = add_temporal_features(df)
    assert res["hour"].iloc[0] == 6
    assert res["month"].iloc[0] == 3
    assert res["day_of_year"].iloc[0] == 74
    assert "diurnal_sin" in res.columns
    assert "seasonal_cos" in res.columns


def test_weather_regime_classifier_interface():
    df = pd.DataFrame([
        {"variable": "rainfall", "forecast_value": 50.0},
        {"variable": "wind_speed", "forecast_value": 25.0},
        {"variable": "temperature", "forecast_value": 42.0},
        {"variable": "temperature", "forecast_value": 22.0},
    ])
    classifier = WeatherRegimeClassifier()
    classified = classifier.classify(df)
    assert "weather_regime" in classified.columns
    assert classified["weather_regime"].iloc[0] == "convective_storm"
    assert classified["weather_regime"].iloc[1] == "synoptic_gale"
    assert classified["weather_regime"].iloc[2] == "extreme_heat"
    assert classified["weather_regime"].iloc[3] == "standard"
