"""
Unit tests for meteorological unit conversion and normalization layer.
"""

import pytest
import numpy as np
import pandas as pd

from ingestion.units import (
    convert_temperature,
    convert_rainfall,
    convert_wind_speed,
    convert_units,
    normalize_dataframe_units,
)


def test_convert_temperature_kelvin_to_celsius():
    assert convert_temperature(273.15, from_unit="K", to_unit="celsius") == pytest.approx(0.0, abs=1e-4)
    assert convert_temperature(300.0, from_unit="kelvin", to_unit="celsius") == pytest.approx(26.85, abs=1e-4)


def test_convert_temperature_fahrenheit_to_celsius():
    assert convert_temperature(32.0, from_unit="F", to_unit="celsius") == pytest.approx(0.0, abs=1e-4)
    assert convert_temperature(212.0, from_unit="fahrenheit", to_unit="celsius") == pytest.approx(100.0, abs=1e-4)


def test_convert_temperature_unsupported_unit():
    with pytest.raises(ValueError, match="Unsupported temperature unit"):
        convert_temperature(100.0, from_unit="rankine")


def test_convert_rainfall_meters_to_mm():
    assert convert_rainfall(0.015, from_unit="m", to_unit="mm") == pytest.approx(15.0, abs=1e-4)
    assert convert_rainfall(0.1245, from_unit="meters", to_unit="mm") == pytest.approx(124.5, abs=1e-4)


def test_convert_rainfall_inches_to_mm():
    assert convert_rainfall(1.0, from_unit="in", to_unit="mm") == pytest.approx(25.4, abs=1e-4)
    assert convert_rainfall(2.5, from_unit="inches", to_unit="mm") == pytest.approx(63.5, abs=1e-4)


def test_convert_rainfall_kg_per_sq_meter():
    # 1 kg/m^2 liquid water = 1 mm
    assert convert_rainfall(35.0, from_unit="kg/m^2", to_unit="mm") == pytest.approx(35.0, abs=1e-4)


def test_convert_wind_speed_kmh_to_mps():
    assert convert_wind_speed(36.0, from_unit="km/h", to_unit="m/s") == pytest.approx(10.0, abs=1e-4)
    assert convert_wind_speed(108.0, from_unit="kmh", to_unit="m/s") == pytest.approx(30.0, abs=1e-4)


def test_convert_wind_speed_knots_to_mps():
    assert convert_wind_speed(10.0, from_unit="kts", to_unit="m/s") == pytest.approx(5.14444, abs=1e-4)


def test_convert_wind_speed_mph_to_mps():
    assert convert_wind_speed(10.0, from_unit="mph", to_unit="m/s") == pytest.approx(4.4704, abs=1e-4)


def test_convert_units_universal_dispatcher():
    assert convert_units(298.15, from_unit="K", variable="temperature") == pytest.approx(25.0, abs=1e-4)
    assert convert_units(0.02, from_unit="m", variable="rainfall") == pytest.approx(20.0, abs=1e-4)
    assert convert_units(72.0, from_unit="km/h", variable="wind_speed") == pytest.approx(20.0, abs=1e-4)


def test_normalize_dataframe_units_by_variable_dict():
    df = pd.DataFrame({
        "variable": ["temperature", "rainfall", "wind_speed"],
        "forecast_value": [300.15, 0.05, 36.0],
    })
    source_units = {
        "temperature": "kelvin",
        "rainfall": "m",
        "wind_speed": "km/h",
    }
    norm = normalize_dataframe_units(df, source_units=source_units)
    assert norm["forecast_value"].iloc[0] == pytest.approx(27.0, abs=1e-4)
    assert norm["forecast_value"].iloc[1] == pytest.approx(50.0, abs=1e-4)
    assert norm["forecast_value"].iloc[2] == pytest.approx(10.0, abs=1e-4)
