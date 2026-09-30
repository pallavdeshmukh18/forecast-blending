"""
Unit tests for data provider adapters and format readers.
"""

from pathlib import Path
import pytest
import pandas as pd

from ingestion.providers import (
    GFSProvider,
    ECMWFProvider,
    WRFProvider,
    AIProvider,
    ERA5Provider,
    StationProvider,
    GenericForecastProvider,
    GenericObservationProvider,
)
from ingestion.readers import load_weather_file


def test_gfs_provider_load_from_csv(tmp_path: Path):
    csv_file = tmp_path / "gfs_raw.csv"
    df_raw = pd.DataFrame({
        "timestamp": ["2026-01-01T12:00:00Z"],
        "lat": [20.0],
        "lon": [75.0],
        "lead": [24],
        "var": ["TMP"],      # GFS alias for temperature
        "val": [295.15],     # Kelvin
    })
    df_raw.to_csv(csv_file, index=False)

    provider = GFSProvider()
    standardized = provider.load_from_file(csv_file)

    assert standardized["variable"].iloc[0] == "temperature"
    # 295.15 K -> 22.0 C
    assert standardized["forecast_value"].iloc[0] == pytest.approx(22.0, abs=1e-4)
    assert standardized["model_name"].iloc[0] == "nwp_model_a"
    assert standardized["lead_time"].iloc[0] == 24


def test_ecmwf_provider_load_from_csv(tmp_path: Path):
    csv_file = tmp_path / "ecmwf_raw.csv"
    df_raw = pd.DataFrame({
        "time": ["2026-01-01T12:00:00Z"],
        "lat": [25.0],
        "lon": [80.0],
        "step": [48],
        "param": ["tp"],     # ECMWF alias for total precipitation
        "value": [0.035],     # 0.035 m
    })
    df_raw.to_csv(csv_file, index=False)

    provider = ECMWFProvider()
    standardized = provider.load_from_file(csv_file)

    assert standardized["variable"].iloc[0] == "rainfall"
    # 0.035 m -> 35.0 mm
    assert standardized["forecast_value"].iloc[0] == pytest.approx(35.0, abs=1e-4)
    assert standardized["model_name"].iloc[0] == "nwp_model_b"


def test_ai_provider_metadata():
    provider = AIProvider(architecture="graph_neural_network")
    assert provider.architecture == "graph_neural_network"
    assert provider.model_type == "ai"
    assert provider.default_model_name == "ai_model_c"


def test_station_provider_load_from_csv(tmp_path: Path):
    csv_file = tmp_path / "station_obs.csv"
    df_raw = pd.DataFrame({
        "time": ["2026-01-01T12:00:00Z"],
        "lat": [19.0],
        "lon": [72.8],
        "param": ["ws10"],
        "value": [8.5],
        "station": ["STN_COASTAL_01"],
    })
    df_raw.to_csv(csv_file, index=False)

    provider = StationProvider()
    standardized = provider.load_from_file(csv_file)

    assert standardized["variable"].iloc[0] == "wind_speed"
    assert standardized["observed_value"].iloc[0] == 8.5
    assert standardized["station_id"].iloc[0] == "STN_COASTAL_01"


def test_load_weather_file_unsupported_format(tmp_path: Path):
    fake_file = tmp_path / "data.xyz"
    fake_file.write_text("dummy")
    with pytest.raises(ValueError, match="Unsupported weather file format"):
        load_weather_file(fake_file)
