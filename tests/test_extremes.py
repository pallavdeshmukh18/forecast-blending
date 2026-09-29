"""
Unit tests for extreme weather event categorization and alerts.
"""

import numpy as np
import pandas as pd
from datetime import datetime

from extremes.rainfall import classify_rainfall_intensity, detect_extreme_rainfall
from extremes.heatwave import classify_heatwave, detect_heatwave_events
from extremes.wind import classify_wind_speed, detect_wind_extremes


def test_classify_rainfall_intensity():
    values = [0.0, 5.0, 20.0, 50.0, 80.0, 150.0]
    cats = classify_rainfall_intensity(values)
    assert cats[0] == "none"
    assert cats[1] == "light"
    assert cats[2] == "moderate"
    assert cats[3] == "heavy"
    assert cats[4] == "very_heavy"
    assert cats[5] == "extreme"


def test_detect_extreme_rainfall():
    df = pd.DataFrame([
        {"variable": "rainfall", "forecast_value": 10.0},
        {"variable": "rainfall", "forecast_value": 45.0},
        {"variable": "temperature", "forecast_value": 45.0},
    ])
    ext = detect_extreme_rainfall(df, threshold=35.5)
    assert len(ext) == 1
    assert ext["forecast_value"].iloc[0] == 45.0


def test_classify_heatwave():
    temps = [32.0, 41.5, 46.2]
    cats = classify_heatwave(temps)
    assert cats[0] == "normal"
    assert cats[1] == "mild_heatwave"
    assert cats[2] == "severe_heatwave"


def test_classify_wind_speed():
    winds = [5.0, 12.0, 19.0, 26.0]
    cats = classify_wind_speed(winds)
    assert cats[0] == "light_moderate"
    assert cats[1] == "strong_breeze"
    assert cats[2] == "gale"
    assert cats[3] == "storm"
