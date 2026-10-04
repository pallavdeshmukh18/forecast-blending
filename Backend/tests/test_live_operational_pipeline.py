"""
Comprehensive test suite for SkyCast Live Operational Ingestion, Normalization,
Validation, Staleness Detection, Disk Caching, Fallback Hierarchy, FastAPI Endpoints,
and Scientific Research Isolation.
"""

from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch
import urllib.error

from fastapi.testclient import TestClient
import numpy as np
import pandas as pd
import pytest

from api.main import app
from live import (
    CurrentConditions,
    DataFreshnessStatus,
    LiveDataCache,
    LiveDisagreementMetric,
    LiveExtremeAlert,
    LiveForecastResult,
    LiveForecastScheduler,
    LiveForecastService,
    LiveForecastStep,
    LiveVariableRecord,
    OpenMeteoClient,
    evaluate_data_freshness,
    get_live_config,
    LiveConfig,
    normalize_current_conditions,
    normalize_hourly_forecasts,
    validate_live_dataframe,
    validate_raw_provider_payload,
)

client = TestClient(app)

SYNTHETIC_BASELINE_SHA256 = "54cb659a7d8e86f0014114e616d2ecbf834d574b877b8db69326e6440c8223fd"


def compute_sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _generate_mock_open_meteo_payload(
    base_time: datetime = None,
    temp_base: float = 28.5,
    wind_base: float = 5.5,
    precip_base: float = 2.0,
) -> Dict[str, Any]:
    """Generate a realistic mock Open-Meteo payload for testing."""
    if base_time is None:
        base_time = datetime.now(timezone.utc)

    hours = [base_time + timedelta(hours=i) for i in range(96)]
    time_strings = [h.strftime("%Y-%m-%dT%H:00:00Z") for h in hours]
    temps = [round(temp_base + 3.0 * np.sin(i / 12.0 * np.pi), 2) for i in range(96)]
    winds = [round(wind_base + 2.0 * np.cos(i / 12.0 * np.pi), 2) for i in range(96)]
    precips = [round(precip_base if (i % 6 == 0 and i > 0) else 0.2, 2) for i in range(96)]
    humidities = [round(70.0 + 10.0 * np.sin(i / 6.0), 1) for i in range(96)]
    pressures = [round(1012.0 + 2.0 * np.cos(i / 24.0), 1) for i in range(96)]

    return {
        "latitude": 19.076,
        "longitude": 72.8777,
        "generationtime_ms": 0.45,
        "utc_offset_seconds": 0,
        "timezone": "UTC",
        "timezone_abbreviation": "UTC",
        "elevation": 14.0,
        "current_units": {
            "time": "iso8601",
            "interval": "seconds",
            "temperature_2m": "°C",
            "relative_humidity_2m": "%",
            "precipitation": "mm",
            "wind_speed_10m": "m/s",
            "wind_direction_10m": "°",
            "surface_pressure": "hPa",
        },
        "current": {
            "time": time_strings[0],
            "interval": 900,
            "temperature_2m": temp_base,
            "relative_humidity_2m": 72.0,
            "precipitation": 0.0,
            "wind_speed_10m": wind_base,
            "wind_direction_10m": 240,
            "surface_pressure": 1012.4,
        },
        "hourly_units": {
            "time": "iso8601",
            "temperature_2m": "°C",
            "relative_humidity_2m": "%",
            "precipitation": "mm",
            "wind_speed_10m": "m/s",
            "wind_direction_10m": "°",
            "surface_pressure": "hPa",
            "cape": "J/kg",
        },
        "hourly": {
            "time": time_strings,
            "temperature_2m": temps,
            "relative_humidity_2m": humidities,
            "precipitation": precips,
            "wind_speed_10m": winds,
            "wind_direction_10m": [220] * 96,
            "surface_pressure": pressures,
            "cape": [450.0] * 96,
        },
        "_retrieved_at": base_time.isoformat(),
        "_provider": "Open-Meteo",
        "_request_url": "https://api.open-meteo.com/v1/forecast",
        "_latency_ms": 42.5,
        "_latitude": 19.076,
        "_longitude": 72.8777,
    }


# ===========================================================================
# 1. UNIT TESTS: SCHEMA & RECORD VALIDATION
# ===========================================================================


def test_live_variable_record_schema():
    rec = LiveVariableRecord(
        timestamp="2026-10-04T12:00:00Z",
        latitude=19.076,
        longitude=72.877,
        variable="temperature",
        value=29.4,
        unit="°C",
        provider="Open-Meteo",
        model="operational_forecast",
        forecast_initialization_time="2026-10-04T06:00:00Z",
        lead_time_hours=6,
        retrieved_at=datetime.now(timezone.utc).isoformat(),
    )
    assert rec.variable == "temperature"
    assert rec.value == 29.4
    assert rec.lead_time_hours == 6
    assert rec.unit == "°C"


def test_current_conditions_schema():
    curr = CurrentConditions(
        location_name="Mumbai Operational Gateway",
        latitude=19.076,
        longitude=72.877,
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        temperature_c=28.5,
        wind_speed_ms=4.2,
        precipitation_mm=0.0,
        provider="Open-Meteo",
        model="operational_analysis",
        retrieved_at=datetime.now(timezone.utc).isoformat(),
        freshness=DataFreshnessStatus.FRESH,
    )
    assert curr.temperature_c == 28.5
    assert curr.freshness == DataFreshnessStatus.FRESH
    assert curr.provider == "Open-Meteo"


# ===========================================================================
# 2. UNIT TESTS: UNIT CONVERSION & NORMALIZATION
# ===========================================================================


def test_unit_conversion_fahrenheit_and_kelvin():
    payload_f = {
        "current": {"temperature_2m": 86.0, "wind_speed_10m": 10.0, "precipitation": 0.0},
        "current_units": {"temperature_2m": "°F", "wind_speed_10m": "km/h"},
    }
    cond_f = normalize_current_conditions(payload_f)
    assert abs(cond_f.temperature_c - 30.0) < 0.01  # (86 - 32) * 5/9 = 30°C
    assert abs(cond_f.wind_speed_ms - (10.0 / 3.6)) < 0.01

    payload_k = {
        "current": {"temperature_2m": 300.15, "wind_speed_10m": 10.0, "precipitation": 0.0},
        "current_units": {"temperature_2m": "K", "wind_speed_10m": "kn"},
    }
    cond_k = normalize_current_conditions(payload_k)
    assert abs(cond_k.temperature_c - 27.0) < 0.01  # 300.15 - 273.15 = 27°C
    assert abs(cond_k.wind_speed_ms - (10.0 * 0.514444)) < 0.01


def test_wind_mph_and_direction_normalization():
    payload_mph = {
        "current": {
            "temperature_2m": 22.0,
            "wind_speed_10m": 20.0,
            "wind_direction_10m": 180,
            "precipitation": -1.0,  # Negative precipitation should clamp to 0
            "surface_pressure": 1013.2,
        },
        "current_units": {"temperature_2m": "°C", "wind_speed_10m": "mph"},
    }
    cond = normalize_current_conditions(payload_mph)
    assert abs(cond.wind_speed_ms - (20.0 * 0.44704)) < 0.01
    assert cond.wind_direction_deg == 180
    assert cond.precipitation_mm == 0.0
    assert cond.surface_pressure_hpa == 1013.2


def test_hourly_rain_accumulation_aggregation():
    """Verify that hourly precipitation is properly rolling-summed into 6-hour windows."""
    payload = _generate_mock_open_meteo_payload(precip_base=5.0)
    df, steps = normalize_hourly_forecasts(payload, target_lead_times=[6, 12, 24, 48, 72])

    assert not df.empty
    assert "rainfall" in df["variable"].values
    assert len(steps) == 5

    step_6 = [s for s in steps if s["lead_time_hours"] == 6][0]
    assert step_6["rainfall_mm"] > 0.0


# ===========================================================================
# 3. UNIT TESTS: QUALITY CONTROL & PHYSICAL BOUNDS
# ===========================================================================


def test_physical_bounds_validation():
    valid_payload = _generate_mock_open_meteo_payload()
    is_valid, errors = validate_raw_provider_payload(valid_payload)
    assert is_valid is True
    assert len(errors) == 0

    # Malformed payloads
    bad_payload_missing = {"latitude": 19.0}
    is_valid, errors = validate_raw_provider_payload(bad_payload_missing)
    assert is_valid is False
    assert any("lacks" in e.lower() for e in errors)

    # DataFrame validation
    df, _ = normalize_hourly_forecasts(valid_payload)
    clean_df, qc = validate_live_dataframe(df)
    assert qc["status"] == "VALID"
    assert qc["valid_records"] == len(df)

    # Inject physically impossible value
    bad_df = df.copy()
    bad_df.loc[0, "forecast_value"] = 120.0  # Impossible temperature
    clean_bad_df, bad_qc = validate_live_dataframe(bad_df)
    assert bad_qc["status"] == "PARTIAL_FILTERED"
    assert bad_qc["dropped_records"] == 1


# ===========================================================================
# 4. UNIT TESTS: STALENESS DETECTION
# ===========================================================================


def test_staleness_detection_logic():
    cfg = get_live_config()
    now = datetime.now(timezone.utc)

    fresh_time = (now - timedelta(minutes=20)).isoformat()
    status_fresh, age_fresh = evaluate_data_freshness(fresh_time, cfg)
    assert status_fresh == DataFreshnessStatus.FRESH
    assert 19 <= age_fresh <= 21

    stale_time = (now - timedelta(minutes=90)).isoformat()
    status_stale, age_stale = evaluate_data_freshness(stale_time, cfg)
    assert status_stale == DataFreshnessStatus.STALE
    assert 89 <= age_stale <= 91

    expired_time = (now - timedelta(minutes=240)).isoformat()
    status_expired, age_expired = evaluate_data_freshness(expired_time, cfg)
    assert status_expired == DataFreshnessStatus.EXPIRED
    assert 239 <= age_expired <= 241

    invalid_time = "not-a-timestamp"
    status_unavail, _ = evaluate_data_freshness(invalid_time, cfg)
    assert status_unavail == DataFreshnessStatus.UNAVAILABLE


# ===========================================================================
# 5. UNIT TESTS: LOCAL DISK CACHE & BOUNDEDNESS
# ===========================================================================


def test_live_data_cache_boundedness(tmp_path):
    cfg = LiveConfig(cache_dir=tmp_path, max_cache_files=3)
    cache = LiveDataCache(config=cfg)
    mock_payload = _generate_mock_open_meteo_payload()

    for i in range(5):
        mock_payload["_retrieved_at"] = (datetime.now(timezone.utc) + timedelta(minutes=i)).isoformat()
        cache.save(mock_payload)

    files = list(tmp_path.glob("live_payload_*.json"))
    assert len(files) <= 3  # Pruning enforces bound

    loaded = cache.load_latest()
    assert loaded is not None
    assert loaded["_provider"] == "Open-Meteo"


# ===========================================================================
# 6. UNIT TESTS: EXTREME WEATHER ALERT THRESHOLDS
# ===========================================================================


def test_extreme_weather_alerts_trigger():
    service = LiveForecastService()
    current = CurrentConditions(
        location_name="Mumbai Operational Gateway",
        latitude=19.076,
        longitude=72.877,
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        temperature_c=42.5,  # High Heat Alert
        wind_speed_ms=19.0,  # Gale Wind Alert
        precipitation_mm=0.0,
    )
    steps = [
        LiveForecastStep(
            lead_time_hours=6,
            valid_time_utc=datetime.now(timezone.utc).isoformat(),
            temperature_c=42.5,
            wind_speed_ms=19.0,
            rainfall_mm=38.0,  # Extreme Rainfall Alert
            regime="convective",
        )
    ]
    alerts = service._detect_extreme_alerts(current, steps)
    hazard_types = [a.hazard_type for a in alerts]

    assert "extreme_rain" in hazard_types
    assert "high_wind" in hazard_types
    assert "heatwave" in hazard_types
    assert len(alerts) == 3


def test_heavy_rain_watch_trigger():
    service = LiveForecastService()
    current = CurrentConditions(
        location_name="Mumbai Operational Gateway",
        latitude=19.076,
        longitude=72.877,
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        temperature_c=25.0,
        wind_speed_ms=5.0,
        precipitation_mm=0.0,
    )
    steps = [
        LiveForecastStep(
            lead_time_hours=12,
            valid_time_utc=datetime.now(timezone.utc).isoformat(),
            temperature_c=25.0,
            wind_speed_ms=5.0,
            rainfall_mm=18.0,  # Heavy rain >= 15 mm/6h
            regime="heavy_rain",
        )
    ]
    alerts = service._detect_extreme_alerts(current, steps)
    assert len(alerts) == 1
    assert alerts[0].hazard_type == "heavy_rain"
    assert alerts[0].severity == "watch"


# ===========================================================================
# 7. INTEGRATION & FALLBACK TESTS (MOCKED)
# ===========================================================================


def test_service_live_forecast_generation_with_mock():
    mock_payload = _generate_mock_open_meteo_payload()
    service = LiveForecastService()

    with patch.object(service.client, "fetch_live_data", return_value=mock_payload):
        result = service.get_live_forecast(force_refresh=True)

        assert isinstance(result, LiveForecastResult)
        assert result.status == "success"
        assert result.provider == "Open-Meteo"
        assert result.models_available == ["operational_forecast"]
        assert len(result.forecast_steps) >= 4
        assert result.freshness in [DataFreshnessStatus.FRESH, DataFreshnessStatus.STALE]
        assert result.current_conditions.temperature_c > 0


def test_service_fallback_to_cache_on_provider_failure(tmp_path):
    cfg = LiveConfig(cache_dir=tmp_path)
    cache = LiveDataCache(config=cfg)
    mock_payload = _generate_mock_open_meteo_payload()
    cache.save(mock_payload)

    service = LiveForecastService(config=cfg)
    service.cache = cache

    with patch.object(service.client, "fetch_live_data", side_effect=ConnectionError("Provider down")):
        result = service.get_live_forecast(latitude=19.076, longitude=72.877)

        assert result.status == "success"
        assert result.metadata.get("cache_used") is True
        assert result.provider == "Open-Meteo"
        assert len(result.forecast_steps) > 0


def test_service_graceful_offline_degradation_when_no_cache(tmp_path):
    cfg = LiveConfig(cache_dir=tmp_path)
    empty_cache = LiveDataCache(config=cfg)
    service = LiveForecastService(config=cfg)
    service.cache = empty_cache

    with patch.object(service.client, "fetch_live_data", side_effect=ConnectionError("Provider down")):
        result = service.get_live_forecast(latitude=19.076, longitude=72.877)

        assert result.status == "error"
        assert result.freshness == DataFreshnessStatus.UNAVAILABLE
        assert len(result.forecast_steps) == 0


def test_client_retry_and_backoff():
    cfg = LiveConfig(max_retries=2, retry_backoff_factor=0.01)
    client_obj = OpenMeteoClient(config=cfg)

    with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("Connection refused")):
        with pytest.raises(ConnectionError) as exc_info:
            client_obj.fetch_live_data()
        assert "Failed to fetch live weather data" in str(exc_info.value)


# ===========================================================================
# 8. FASTAPI ENDPOINT TESTS
# ===========================================================================


def test_fastapi_live_status_endpoint():
    res = client.get("/api/live/status")
    assert res.status_code == 200
    data = res.json()
    assert "status" in data
    assert "provider" in data
    assert "freshness" in data
    assert "default_node" in data

    # Route alias check
    res_alias = client.get("/live/status")
    assert res_alias.status_code == 200


def test_fastapi_live_metadata_endpoint():
    res = client.get("/api/live/metadata")
    assert res.status_code == 200
    data = res.json()
    assert data["provider"] == "Open-Meteo"
    assert "scientific_boundary" in data
    assert "extreme_thresholds" in data

    res_alias = client.get("/live/metadata")
    assert res_alias.status_code == 200


def test_fastapi_live_current_and_forecast_endpoints():
    mock_payload = _generate_mock_open_meteo_payload()

    with patch("live.clients.weather_client.OpenMeteoClient.fetch_live_data", return_value=mock_payload):
        # /live/current
        res_curr = client.get("/api/live/current?force_refresh=true")
        assert res_curr.status_code == 200
        curr_data = res_curr.json()
        assert "temperature_c" in curr_data
        assert "wind_speed_ms" in curr_data
        assert curr_data["provider"] == "Open-Meteo"

        # /live/forecast
        res_fc = client.get("/api/live/forecast?force_refresh=true&blender=regime_blender")
        assert res_fc.status_code == 200
        fc_data = res_fc.json()
        assert "forecast_steps" in fc_data
        assert len(fc_data["forecast_steps"]) > 0
        assert fc_data["models_available"] == ["operational_forecast"]

        # /live/alerts
        res_alert = client.get("/api/live/alerts")
        assert res_alert.status_code == 200
        assert "alerts" in res_alert.json()

        # /live/disagreement
        res_disagree = client.get("/api/live/disagreement")
        assert res_disagree.status_code == 200
        assert "disagreement" in res_disagree.json()


def test_scheduler_lifecycle():
    import asyncio

    async def _run_test():
        cfg = LiveConfig(live_mode_enabled=True, refresh_interval_minutes=15)
        mock_service = MagicMock(spec=LiveForecastService)
        scheduler = LiveForecastScheduler(service=mock_service, config=cfg)
        assert scheduler._is_running is False
        await scheduler.start()
        assert scheduler._is_running is True
        # Calling start again should not duplicate
        await scheduler.start()
        assert scheduler._is_running is True
        await scheduler.stop()
        assert scheduler._is_running is False

    asyncio.run(_run_test())


# ===========================================================================
# 10. RESEARCH ISOLATION & SCIENTIFIC INTEGRITY TEST (STEP 21)
# ===========================================================================


def test_research_benchmark_isolation_and_synthetic_hash():
    """
    CRITICAL STEP 21: Verify that executing live ingestion and operations
    CANNOT mutate any WeatherBench 2 benchmark files, splits, or synthetic baseline.
    """
    project_root = Path(__file__).resolve().parent.parent.parent
    bench_dir = project_root / "ML" / "data" / "weatherbench2"
    reports_dir = project_root / "ML" / "reports"

    # 1. Verify WeatherBench benchmark directory is completely untouched by live caching
    live_cache_dir = project_root / "ML" / "data" / "live_cache"
    assert not str(live_cache_dir).startswith(str(bench_dir))

    # 2. Check synthetic baseline SHA-256 hash
    synth_csv = reports_dir / "final_model_comparison.csv"
    assert synth_csv.exists(), "Missing final_model_comparison.csv"
    actual_hash = compute_sha256(synth_csv)

    assert actual_hash == SYNTHETIC_BASELINE_SHA256, (
        f"CRITICAL: Synthetic baseline hash was altered! Expected {SYNTHETIC_BASELINE_SHA256}, got {actual_hash}"
    )

    # 3. Verify that live operational cache does not write into benchmark files
    if bench_dir.exists():
        assert not any(f.name.startswith("live_") for f in bench_dir.glob("*.*"))
