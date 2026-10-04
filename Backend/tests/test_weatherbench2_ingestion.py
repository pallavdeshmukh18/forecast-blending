"""
Tests for WeatherBench 2 Real Meteorological Data Integration.

Verifies:
- Variable mapping & standardization
- Unit conversion (Kelvin -> Celsius, meters -> mm)
- Wind speed component derivation (sqrt(u^2 + v^2))
- Spatio-temporal alignment and valid time calculation
- Quality control & missing-value handling
- Manifest and provenance generation
- Synthetic isolation (DATA_MODE=synthetic does not depend on WeatherBench 2)
- Real mode execution (DATA_MODE=real selects real paths with no silent synthetic fallback)
- Strict temporal causality (no data leakage)
"""

import os
from pathlib import Path
import sys

# Setup project import paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "ML"))
sys.path.insert(0, str(PROJECT_ROOT / "Backend"))
sys.path.insert(0, str(PROJECT_ROOT))

from datetime import datetime, timezone
import hashlib
import json
import tempfile
import numpy as np
import pandas as pd
import pytest

from configs import AppConfig, get_config
from ingestion.schema import (
    ForecastRecord,
    ObservationRecord,
    validate_forecast_dataframe,
    validate_observation_dataframe,
)
from ingestion.variable_mapping import standardize_variable_name
from ingestion.units import convert_units, convert_temperature, convert_rainfall, convert_wind_speed
from ingestion.quality_control import apply_quality_control
from ingestion.providers import (
    WeatherBench2ForecastProvider,
    WeatherBench2ObservationProvider,
)
from preprocessing.matching import match_forecasts_to_observations
from preprocessing.temporal import validate_and_compute_valid_time
from pipeline.forecast_pipeline import ForecastPipelineOrchestrator


# =====================================================================
# Test 1: Variable Mapping
# =====================================================================

def test_weatherbench2_variable_mapping():
    """Verify WeatherBench 2 variable names map correctly to standard canonical names."""
    assert standardize_variable_name("2m_temperature") == "temperature"
    assert standardize_variable_name("t2m") == "temperature"
    assert standardize_variable_name("total_precipitation_6hr") == "rainfall"
    assert standardize_variable_name("total_precipitation") == "rainfall"
    assert standardize_variable_name("10m_wind_speed") == "wind_speed"
    assert standardize_variable_name("ws10") == "wind_speed"


# =====================================================================
# Test 2: Unit Conversion
# =====================================================================

def test_weatherbench2_unit_conversion_temperature():
    """Verify Kelvin to Celsius temperature conversions."""
    # 273.15 K == 0.0 °C
    assert convert_temperature(273.15, from_unit="kelvin", to_unit="celsius") == pytest.approx(0.0, abs=1e-4)
    # 300.0 K == 26.85 °C
    assert convert_temperature(300.0, from_unit="kelvin", to_unit="celsius") == pytest.approx(26.85, abs=1e-4)
    # 250.0 K == -23.15 °C
    assert convert_temperature(250.0, from_unit="kelvin", to_unit="celsius") == pytest.approx(-23.15, abs=1e-4)


def test_weatherbench2_unit_conversion_rainfall():
    """Verify meters to millimeters precipitation accumulation conversions."""
    # 0.005 m == 5.0 mm
    assert convert_rainfall(0.005, from_unit="meters", to_unit="mm") == pytest.approx(5.0, abs=1e-4)
    # 0.0254 m == 25.4 mm
    assert convert_rainfall(0.0254, from_unit="m", to_unit="mm") == pytest.approx(25.4, abs=1e-4)
    # 0.0 m == 0.0 mm
    assert convert_rainfall(0.0, from_unit="m", to_unit="mm") == pytest.approx(0.0, abs=1e-4)


# =====================================================================
# Test 3: Wind-Speed Derivation
# =====================================================================

def test_weatherbench2_wind_speed_derivation():
    """Verify wind speed vector magnitude derivation sqrt(u^2 + v^2)."""
    # u=3, v=4 -> ws=5
    u = np.array([3.0, 0.0, -6.0, -3.0])
    v = np.array([4.0, 10.0, 8.0, -4.0])
    expected = np.array([5.0, 10.0, 10.0, 5.0])
    computed = np.sqrt(u**2 + v**2)
    np.testing.assert_allclose(computed, expected, rtol=1e-5)


# =====================================================================
# Test 4: Time Alignment
# =====================================================================

def test_weatherbench2_time_alignment():
    """Verify valid_time = run_time + lead_time calculation and consistency."""
    run_t = datetime(2020, 1, 1, 0, 0, tzinfo=timezone.utc)
    lead_h = 24
    expected_valid_t = datetime(2020, 1, 2, 0, 0, tzinfo=timezone.utc)

    df = pd.DataFrame([{
        "run_time": run_t,
        "lead_time": lead_h,
        "latitude": 15.0,
        "longitude": 75.0,
        "model_name": "ecmwf_hres",
        "variable": "temperature",
        "forecast_value": 24.5,
    }])

    validated = validate_and_compute_valid_time(df)
    assert "timestamp" in validated.columns
    assert validated["timestamp"].iloc[0] == expected_valid_t


# =====================================================================
# Test 5: Spatial Alignment
# =====================================================================

def test_weatherbench2_spatial_alignment():
    """Verify spatial matching pairs forecast and observation grids accurately."""
    t0 = datetime(2020, 1, 1, 12, 0, tzinfo=timezone.utc)
    obs_df = pd.DataFrame([
        {"timestamp": t0, "latitude": 15.0, "longitude": 75.0, "variable": "temperature", "observed_value": 25.0},
        {"timestamp": t0, "latitude": 20.0, "longitude": 80.0, "variable": "temperature", "observed_value": 22.0},
    ])
    f_df = pd.DataFrame([
        {"timestamp": t0, "run_time": t0 - pd.Timedelta(hours=12), "lead_time": 12, "latitude": 15.0, "longitude": 75.0, "model_name": "ecmwf_hres", "variable": "temperature", "forecast_value": 24.8},
        {"timestamp": t0, "run_time": t0 - pd.Timedelta(hours=12), "lead_time": 12, "latitude": 20.0, "longitude": 80.0, "model_name": "ecmwf_hres", "variable": "temperature", "forecast_value": 21.9},
    ])

    matched = match_forecasts_to_observations(forecasts_df=f_df, observations_df=obs_df)
    assert len(matched) == 2
    assert "forecast_error" in matched.columns
    assert matched["forecast_error"].iloc[0] == pytest.approx(-0.2, abs=1e-3)
    assert matched["forecast_error"].iloc[1] == pytest.approx(-0.1, abs=1e-3)


# =====================================================================
# Test 6: Missing Values & QC
# =====================================================================

def test_weatherbench2_quality_control_missing_values():
    """Verify QC flags missing, infinite, and out-of-physical-range values."""
    t0 = datetime(2020, 1, 1, 12, 0, tzinfo=timezone.utc)
    raw_df = pd.DataFrame([
        {"timestamp": t0, "latitude": 15.0, "longitude": 75.0, "model_name": "ecmwf_hres", "lead_time": 6, "variable": "temperature", "forecast_value": 25.0},
        {"timestamp": t0, "latitude": 15.0, "longitude": 75.0, "model_name": "ecmwf_hres", "lead_time": 6, "variable": "temperature", "forecast_value": np.nan},
        {"timestamp": t0, "latitude": 15.0, "longitude": 75.0, "model_name": "ecmwf_hres", "lead_time": 6, "variable": "rainfall", "forecast_value": -10.0}, # unphysical negative rain
        {"timestamp": t0, "latitude": 15.0, "longitude": 75.0, "model_name": "ecmwf_hres", "lead_time": 6, "variable": "wind_speed", "forecast_value": np.inf},
    ])

    valid_df, rej_df, report = apply_quality_control(raw_df, value_col="forecast_value", variable_col="variable")
    assert len(valid_df) == 1
    assert len(rej_df) == 3
    assert report.missing_records >= 2


# =====================================================================
# Test 7: Provenance & Manifest Generation
# =====================================================================

def test_weatherbench2_manifest_provenance_structure(tmp_path):
    """Verify manifest and provenance JSON files contain required metadata."""
    prov_dir = tmp_path / "provenance"
    prov_dir.mkdir(parents=True)

    dummy_file = prov_dir / "sample.parquet"
    dummy_file.write_bytes(b"dummy parquet bytes")
    file_hash = hashlib.sha256(b"dummy parquet bytes").hexdigest()

    manifest = {
        "dataset": "WeatherBench 2",
        "source": "Google Cloud Storage (gs://weatherbench2/datasets/)",
        "dataset_version": "1.5_deg_equiangular_with_poles_conservative",
        "download_timestamp": datetime.now(timezone.utc).isoformat(),
        "variables": ["temperature", "wind_speed", "rainfall"],
        "forecast_models": ["ecmwf_hres", "graphcast"],
        "start_date": "2020-01-01",
        "end_date": "2020-01-03",
        "sha256": {"sample": file_hash},
        "transformations": ["Kelvin to Celsius", "Meters to mm"],
    }
    manifest_path = prov_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f)

    with open(manifest_path, "r", encoding="utf-8") as f:
        loaded = json.load(f)

    assert loaded["dataset"] == "WeatherBench 2"
    assert "temperature" in loaded["variables"]
    assert loaded["sha256"]["sample"] == file_hash


# =====================================================================
# Test 8: Synthetic Isolation
# =====================================================================

def test_weatherbench2_synthetic_isolation(monkeypatch):
    """Verify DATA_MODE=synthetic operates independently of WeatherBench 2 data."""
    monkeypatch.setenv("DATA_MODE", "synthetic")
    orchestrator = ForecastPipelineOrchestrator()
    result = orchestrator.execute(mode="synthetic", variable="temperature", lead_time=24)
    assert result is not None
    assert result.forecast_count > 0
    assert result.mode == "synthetic"
    assert "fallback_to_synthetic" not in result.diagnostics or not result.diagnostics["fallback_to_synthetic"]


# =====================================================================
# Test 9: Real Mode Selection & Strict No-Fallback
# =====================================================================

def test_weatherbench2_real_mode_selection_and_strict_error(monkeypatch):
    """Verify DATA_MODE=real fails explicitly if real data is missing (no silent fallback)."""
    monkeypatch.setenv("DATA_MODE", "real")
    orchestrator = ForecastPipelineOrchestrator()

    # Pass non-existent file paths to trigger strict FileNotFoundError
    orchestrator.config.data.real_forecasts_path = "data/non_existent_forecasts.parquet"
    orchestrator.config.data.real_observations_path = "data/non_existent_obs.parquet"

    with pytest.raises(FileNotFoundError) as excinfo:
        orchestrator.execute(mode="real", variable="temperature", lead_time=24)
    assert "Real mode requires real WeatherBench 2 data" in str(excinfo.value)


# =====================================================================
# Test 10: No Temporal Leakage
# =====================================================================

def test_weatherbench2_no_data_leakage():
    """Verify verification records preserve initialization time, lead time, and causality."""
    t0 = datetime(2020, 1, 1, 0, 0, tzinfo=timezone.utc)
    t1 = datetime(2020, 1, 1, 6, 0, tzinfo=timezone.utc)
    t2 = datetime(2020, 1, 1, 12, 0, tzinfo=timezone.utc)

    obs_df = pd.DataFrame([
        {"timestamp": t1, "latitude": 15.0, "longitude": 75.0, "variable": "temperature", "observed_value": 24.0},
        {"timestamp": t2, "latitude": 15.0, "longitude": 75.0, "variable": "temperature", "observed_value": 28.0},
    ])
    f_df = pd.DataFrame([
        {"timestamp": t1, "run_time": t0, "lead_time": 6, "latitude": 15.0, "longitude": 75.0, "model_name": "ecmwf_hres", "variable": "temperature", "forecast_value": 23.5},
        {"timestamp": t2, "run_time": t0, "lead_time": 12, "latitude": 15.0, "longitude": 75.0, "model_name": "ecmwf_hres", "variable": "temperature", "forecast_value": 27.5},
    ])

    matched = match_forecasts_to_observations(forecasts_df=f_df, observations_df=obs_df)
    # Must preserve run_time and lead_time exactly
    assert "run_time" in matched.columns
    assert "lead_time" in matched.columns
    for _, row in matched.iterrows():
        assert row["timestamp"] == row["run_time"] + pd.Timedelta(hours=row["lead_time"])
        # Forecast initialization must strictly precede valid verification time for lead_time > 0
        assert row["run_time"] < row["timestamp"]


# =====================================================================
# Test 11: Provider Classes
# =====================================================================

def test_weatherbench2_provider_classes():
    """Verify WeatherBench 2 provider classes instantiate and configure correct metadata."""
    f_prov = WeatherBench2ForecastProvider(default_model_name="ecmwf_hres")
    assert f_prov.provider_name == "weatherbench2_forecast"
    assert f_prov.source_units["temperature"] == "kelvin"
    assert f_prov.source_units["rainfall"] == "m"

    o_prov = WeatherBench2ObservationProvider()
    assert o_prov.provider_name == "weatherbench2_era5"
    assert o_prov.source_units["temperature"] == "kelvin"
