"""
Unit tests for Provider-Neutral Real Data Pipeline, Data Provenance, and Quality Control.
"""

from pathlib import Path
import json
import pytest
import pandas as pd
import numpy as np

from data.provenance import DataProvenance, ProvenanceTracker, get_provenance_tracker
from ingestion.providers import (
    ForecastProvider,
    ObservationProvider,
    NWPForecastProvider,
    SurfaceStationObservationProvider,
    ReanalysisObservationProvider,
    SatellitePrecipitationProvider,
    RadarPrecipitationProvider,
)
from ingestion.quality_control import apply_quality_control, DataQualityReport, save_quality_report


def test_data_provenance_tracking(tmp_path: Path):
    prov_file = tmp_path / "test_provenance.json"
    tracker = ProvenanceTracker(storage_path=prov_file)

    rec = tracker.register_dataset(
        source="ECMWF_HRES",
        dataset_name="ecmwf_temperature_025",
        provider="ECMWF",
        variable="temperature",
        units="celsius",
        spatial_resolution="0.25 deg",
        temporal_resolution="6h",
    )

    assert rec.source == "ECMWF_HRES"
    assert rec.variable == "temperature"
    assert prov_file.exists()

    with open(prov_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        assert len(data) == 1
        assert data[0]["provider"] == "ECMWF"


def test_nwp_forecast_provider(tmp_path: Path):
    csv_file = tmp_path / "nwp_sample.csv"
    df = pd.DataFrame({
        "time": ["2026-01-01T12:00:00Z", "2026-01-01T18:00:00Z"],
        "lat": [20.0, 20.0],
        "lon": [75.0, 75.0],
        "lead_time": [24, 30],
        "variable": ["rainfall", "rainfall"],
        "forecast_value": [12.5, 25.0],
    })
    df.to_csv(csv_file, index=False)

    provider = NWPForecastProvider(provider_name="TestNWP", default_file_path=csv_file)
    res_df = provider.fetch_forecast(lead_times=[24])

    assert len(res_df) == 1
    assert res_df["lead_time"].iloc[0] == 24
    assert res_df["variable"].iloc[0] == "rainfall"
    assert res_df["model_name"].iloc[0] == "TestNWP"


def test_surface_station_observation_provider(tmp_path: Path):
    csv_file = tmp_path / "obs_sample.csv"
    df = pd.DataFrame({
        "timestamp": ["2026-01-01T12:00:00Z", "2026-01-01T18:00:00Z"],
        "latitude": [19.0, 19.0],
        "longitude": [72.8, 72.8],
        "variable": ["wind_speed", "wind_speed"],
        "observed_value": [7.5, 9.2],
        "station_id": ["STN_001", "STN_001"],
    })
    df.to_csv(csv_file, index=False)

    provider = SurfaceStationObservationProvider(default_file_path=csv_file)
    stations = provider.list_available_stations()
    assert "STN_001" in stations

    obs_df = provider.fetch_observations(variables=["wind_speed"])
    assert len(obs_df) == 2
    assert obs_df["observed_value"].iloc[0] == 7.5


def test_quality_control_report_json(tmp_path: Path):
    report_file = tmp_path / "data_quality.json"
    df = pd.DataFrame({
        "timestamp": ["2026-01-01T12:00:00Z", "2026-01-01T12:00:00Z", "2026-01-01T18:00:00Z"],
        "latitude": [20.0, 20.0, 20.0],
        "longitude": [75.0, 75.0, 75.0],
        "model_name": ["model_a", "model_a", "model_a"],
        "lead_time": [24, 24, 24],
        "variable": ["temperature", "temperature", "temperature"],
        "forecast_value": [25.0, 25.0, -100.0],  # row 2 is duplicate, row 3 violates physical temperature bound
    })

    valid_df, rej_df, report = apply_quality_control(
        df, value_col="forecast_value", save_report_path=report_file
    )

    assert len(valid_df) == 1
    assert len(rej_df) == 2
    assert report_file.exists()

    with open(report_file, "r", encoding="utf-8") as f:
        rep_json = json.load(f)
        assert rep_json["duplicate_records"] >= 1
        assert rep_json["out_of_range_records"] >= 1
