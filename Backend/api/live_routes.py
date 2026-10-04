"""
FastAPI router for SkyCast Live Operational Forecasting endpoints.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Query, HTTPException

_project_root = Path(__file__).resolve().parent.parent.parent
_ml_dir = _project_root / "ML"
for _p in [str(_ml_dir), str(_project_root)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from live import (
    LiveConfig,
    get_live_config,
    LiveForecastService,
    LiveForecastResult,
    CurrentConditions,
    LiveExtremeAlert,
    LiveDisagreementMetric,
    DataFreshnessStatus,
)

router = APIRouter(prefix="", tags=["Live Operational Forecasting"])
_live_service = LiveForecastService()


@router.get("/live/status", summary="Get live operational connection status, freshness, and latency")
@router.get("/api/live/status", summary="Get live operational connection status, freshness, and latency (API alias)")
def get_live_status():
    """
    Returns the real-time operational status, active weather provider, cache health, and data age.
    """
    cfg = get_live_config()
    cached = _live_service.cache.load_latest()
    
    if cached and "_retrieved_at" in cached:
        from live.staleness import evaluate_data_freshness
        freshness, age = evaluate_data_freshness(cached["_retrieved_at"], cfg)
        last_retrieved = cached["_retrieved_at"]
        latency_ms = cached.get("_latency_ms", 0.0)
    else:
        freshness = DataFreshnessStatus.UNAVAILABLE
        age = 9999.0
        last_retrieved = None
        latency_ms = 0.0

    return {
        "status": "operational" if freshness != DataFreshnessStatus.UNAVAILABLE else "degraded",
        "mode": "live",
        "provider": cfg.provider_name,
        "api_endpoint": cfg.api_base_url,
        "freshness": freshness.value,
        "data_age_minutes": age,
        "last_retrieved_utc": last_retrieved,
        "latency_ms": latency_ms,
        "cache_files_count": len(list(cfg.cache_dir.glob("live_payload_*.json"))),
        "default_node": {
            "name": cfg.default_location_name,
            "latitude": cfg.default_latitude,
            "longitude": cfg.default_longitude,
        },
    }


@router.get("/live/current", summary="Get current surface meteorological conditions")
@router.get("/api/live/current", summary="Get current surface meteorological conditions (API alias)")
def get_live_current(
    latitude: Optional[float] = Query(None, description="Latitude (defaults to configured node)"),
    longitude: Optional[float] = Query(None, description="Longitude (defaults to configured node)"),
    force_refresh: bool = Query(False, description="Bypass cache and query live provider directly"),
):
    """
    Returns latest real-time surface observation / analysis (Temperature, Wind, Precipitation, Humidity, Pressure).
    """
    result = _live_service.get_live_forecast(
        latitude=latitude,
        longitude=longitude,
        force_refresh=force_refresh,
    )
    return result.current_conditions


@router.get("/live/forecast", summary="Get multi-horizon operational live forecast with adaptive blending")
@router.get("/api/live/forecast", summary="Get multi-horizon operational live forecast (API alias)")
def get_live_forecast(
    latitude: Optional[float] = Query(None, description="Latitude (defaults to configured node)"),
    longitude: Optional[float] = Query(None, description="Longitude (defaults to configured node)"),
    blender: str = Query("regime_blender", description="Blending algorithm ('regime_blender', 'inverse_variance', 'equal_weight', 'spatial_cnn')"),
    force_refresh: bool = Query(False, description="Bypass cache and query live provider directly"),
):
    """
    Executes live operational forecasting pipeline across horizons (+6h, +12h, +24h, +48h, +72h).
    Returns point forecasts, calibrated 90% confidence bounds, regimes, and extreme alerts.
    """
    result = _live_service.get_live_forecast(
        latitude=latitude,
        longitude=longitude,
        blender_method=blender,
        force_refresh=force_refresh,
    )
    return result


@router.get("/live/alerts", summary="Get active operational extreme weather alerts")
@router.get("/api/live/alerts", summary="Get active operational extreme weather alerts (API alias)")
def get_live_alerts(
    latitude: Optional[float] = Query(None, description="Latitude"),
    longitude: Optional[float] = Query(None, description="Longitude"),
):
    """
    Returns active extreme weather warnings/watches/advisories (heavy rain, gale wind, extreme heat, cold waves).
    """
    result = _live_service.get_live_forecast(latitude=latitude, longitude=longitude)
    return {
        "location": result.location,
        "count": len(result.active_alerts),
        "alerts": result.active_alerts,
        "timestamp_utc": result.retrieved_at_utc,
    }


@router.get("/live/disagreement", summary="Get model disagreement diagnostics")
@router.get("/api/live/disagreement", summary="Get model disagreement diagnostics (API alias)")
def get_live_disagreement(
    latitude: Optional[float] = Query(None, description="Latitude"),
    longitude: Optional[float] = Query(None, description="Longitude"),
):
    """
    Returns physical-AI model disagreement diagnostics across operational forecast members.
    """
    result = _live_service.get_live_forecast(latitude=latitude, longitude=longitude)
    return {
        "location": result.location,
        "models_available": result.models_available,
        "disagreement": result.disagreement,
    }


@router.get("/live/metadata", summary="Get live operational layer metadata and scientific boundaries")
@router.get("/api/live/metadata", summary="Get live operational layer metadata (API alias)")
def get_live_metadata():
    """
    Returns configuration, supported variables, lead horizons, update frequency, and scientific limitations.
    """
    cfg = get_live_config()
    return {
        "operational_layer": "SkyCast Live Operational Ingestion & Blending",
        "version": "0.6.0",
        "provider": cfg.provider_name,
        "provider_endpoint": cfg.api_base_url,
        "supported_variables": cfg.supported_variables,
        "supported_lead_times_hours": cfg.supported_lead_times_hours,
        "refresh_interval_minutes": cfg.refresh_interval_minutes,
        "freshness_thresholds": {
            "fresh_minutes": cfg.fresh_threshold_minutes,
            "stale_minutes": cfg.stale_threshold_minutes,
            "expired_minutes": cfg.cache_max_age_minutes,
        },
        "extreme_thresholds": {
            "heavy_rain_6h_mm": cfg.heavy_rain_threshold_6h_mm,
            "extreme_rain_6h_mm": cfg.extreme_rain_threshold_6h_mm,
            "high_wind_ms": cfg.high_wind_threshold_ms,
            "high_temp_c": cfg.high_temp_threshold_c,
            "cold_temp_c": cfg.cold_temp_threshold_c,
        },
        "scientific_boundary": (
            "Live mode uses newly ingested operational forecasts from Open-Meteo and is strictly "
            "isolated from the historical WeatherBench 2 research benchmark. Operational live forecasts "
            "are not equivalent to the audited historical benchmark and represent a real-time deployment gateway."
        ),
    }
