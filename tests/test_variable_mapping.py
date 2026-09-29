"""
Unit tests for variable standardization and provider alias mapping.
"""

import pandas as pd
from ingestion.variable_mapping import (
    standardize_variable_name,
    standardize_dataframe_variables,
)


def test_standardize_temperature_aliases():
    for alias in ["temperature", "t2m", "T2M", "2m_temperature", "temp", "tmp", "TMP", "t_2m", "air_temperature"]:
        assert standardize_variable_name(alias) == "temperature"


def test_standardize_rainfall_aliases():
    for alias in ["rainfall", "rain", "tp", "TP", "precipitation", "precip", "total_precipitation", "apcp", "APCP", "rain_mm"]:
        assert standardize_variable_name(alias) == "rainfall"


def test_standardize_wind_aliases():
    for alias in ["wind_speed", "wind", "ws10", "WS10", "10m_wind", "10si", "si10", "wspd", "windspeed", "speed_10m"]:
        assert standardize_variable_name(alias) == "wind_speed"


def test_standardize_dataframe_variables():
    df = pd.DataFrame({
        "variable": ["t2m", "TP", "ws10", "2m_temperature", "apcp"],
        "forecast_value": [25.0, 10.0, 5.0, 22.0, 15.0],
    })
    standardized = standardize_dataframe_variables(df)
    assert list(standardized["variable"]) == ["temperature", "rainfall", "wind_speed", "temperature", "rainfall"]
