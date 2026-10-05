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


def _multi_model_disagreement(lat: float, lon: float, cfg) -> Optional[Dict[str, Any]]:
    """Mean |ECMWF IFS - GFS| over the next 24 h for temperature, wind and precipitation (Open-Meteo)."""
    import urllib.parse, urllib.request, json

    series: Dict[str, Dict[str, List[Optional[float]]]] = {}
    for model in ("ecmwf_ifs025", "gfs_seamless"):
        params = {
            "latitude": f"{lat:.4f}", "longitude": f"{lon:.4f}",
            "hourly": "temperature_2m,wind_speed_10m,precipitation",
            "wind_speed_unit": "ms", "timezone": "UTC", "forecast_days": 2, "models": model,
        }
        req = urllib.request.Request(
            f"{cfg.api_base_url}?{urllib.parse.urlencode(params)}",
            headers={"User-Agent": "SkyCast-Weather-Intelligence/0.6.0", "Accept": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=cfg.request_timeout_seconds) as resp:
            hourly = json.loads(resp.read().decode("utf-8")).get("hourly", {})
        series[model] = {k: hourly.get(k, [])[:24] for k in ("temperature_2m", "wind_speed_10m", "precipitation")}

    thresholds = {  # (low_max, moderate_max)
        "temperature": ("temperature_2m", 1.0, 2.5, "degC"),
        "wind_speed": ("wind_speed_10m", 1.5, 3.5, "m/s"),
        "rainfall": ("precipitation", 0.3, 1.0, "mm/h"),
    }
    out = []
    for var, (key, lo, mid, unit) in thresholds.items():
        a, b = series["ecmwf_ifs025"][key], series["gfs_seamless"][key]
        diffs = [abs(x - y) for x, y in zip(a, b) if x is not None and y is not None]
        if not diffs:
            continue
        mad = sum(diffs) / len(diffs)
        level = "low" if mad < lo else "moderate" if mad < mid else "high"
        out.append({
            "variable": var,
            "lead_time_hours": 24,
            "mean_disagreement": round(mad, 3),
            "disagreement_level": level,
            "models_compared": ["ecmwf_ifs025", "gfs_seamless"],
            "interpretation": f"Mean |ECMWF IFS - GFS| over the next 24 h = {mad:.2f} {unit} ({level}). "
                              "GraphCast is not part of the live feed.",
        })
    return {"models": ["ecmwf_ifs025", "gfs_seamless"], "disagreement": out} if out else None


@router.get("/live/disagreement", summary="Get model disagreement diagnostics")
@router.get("/api/live/disagreement", summary="Get model disagreement diagnostics (API alias)")
def get_live_disagreement(
    latitude: Optional[float] = Query(None, description="Latitude"),
    longitude: Optional[float] = Query(None, description="Longitude"),
):
    """
    Returns model disagreement diagnostics. When the provider is reachable this is the real
    ECMWF IFS vs GFS mean absolute difference over the next 24 h; otherwise it falls back to the
    single-feed 'unavailable' diagnostics from the live service.
    """
    cfg = get_live_config()
    result = _live_service.get_live_forecast(latitude=latitude, longitude=longitude)
    lat = result.location.get("latitude", cfg.default_latitude)
    lon = result.location.get("longitude", cfg.default_longitude)
    try:
        multi = _multi_model_disagreement(lat, lon, cfg)
    except Exception as exc:
        import logging
        logging.getLogger(__name__).warning('Multi-model disagreement unavailable: %s', exc)
        multi = None
    if multi:
        return {
            "location": result.location,
            "models_available": multi["models"],
            "disagreement": multi["disagreement"],
        }
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


# ---------------------------------------------------------------------------
# Live spatial field (wind vectors, temperature, precipitation, disagreement)
# ---------------------------------------------------------------------------
import json as _json
import math as _math
import threading as _threading
import time as _time
import urllib.parse as _urlparse
import urllib.request as _urlrequest
from datetime import datetime as _dt, timedelta as _timedelta, timezone as _tz

_GRID_LAT = (6.0, 36.0)
_GRID_LON = (67.0, 98.0)
_GRID_N = 8
_grid_cache: Dict[str, Any] = {}
_grid_lock = _threading.Lock()


def _linspace(a: float, b: float, n: int) -> List[float]:
    return [round(a + (b - a) * i / (n - 1), 4) for i in range(n)]


def _fetch_grid_points(
    lats: List[float],
    lons: List[float],
    variables: str,
    model: Optional[str],
    cfg,
    target_hour: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Fetch one value per variable per grid point. target_hour (ISO 'YYYY-MM-DDTHH:00') selects a
    forecast hour; otherwise the provider's current analysis is used. Result is normalised to {'current': {...}}."""
    params = {
        "latitude": ",".join(f"{v:.3f}" for v in lats),
        "longitude": ",".join(f"{v:.3f}" for v in lons),
        "wind_speed_unit": "ms",
        "precipitation_unit": "mm",
        "timezone": "UTC",
    }
    if target_hour:
        params["hourly"] = variables
        params["start_hour"] = target_hour
        params["end_hour"] = target_hour
    else:
        params["current"] = variables
    if model:
        params["models"] = model
    url = f"{cfg.api_base_url}?{_urlparse.urlencode(params)}"
    req = _urlrequest.Request(url, headers={"User-Agent": "SkyCast-Weather-Intelligence/0.6.0", "Accept": "application/json"})
    with _urlrequest.urlopen(req, timeout=cfg.request_timeout_seconds) as resp:
        data = _json.loads(resp.read().decode("utf-8"))
    data = data if isinstance(data, list) else [data]
    if target_hour:
        for d in data:
            hourly = d.get("hourly") or {}
            d["current"] = {k: (v[0] if v else None) for k, v in hourly.items() if k != "time"}
            d["current"]["time"] = (hourly.get("time") or [target_hour])[0]
    return data


_GRID_VARS = "temperature_2m,precipitation,wind_speed_10m,wind_direction_10m,relative_humidity_2m,pressure_msl,dew_point_2m,cloud_cover"


@router.get("/live/wind-grid", summary="Get live gridded wind vectors (u10, v10), temperature, precipitation and optional model disagreement")
@router.get("/api/live/wind-grid", summary="Live gridded surface field over South Asia (API alias)")
def get_live_wind_grid(
    include_disagreement: bool = Query(False, description="Also compute |ECMWF IFS - GFS| wind/temperature disagreement"),
    force_refresh: bool = Query(False, description="Bypass in-memory cache"),
    hour: int = Query(0, ge=0, le=72, description="Forecast hour ahead of now (0 = current analysis)"),
):
    """
    Returns a coarse regular grid over 6-36N, 67-98E. Wind components follow the meteorological
    convention: u10 = -speed*sin(dir), v10 = -speed*cos(dir) (direction wind blows FROM).
    Data comes straight from the live provider; if it is unreachable a 503 is returned (no synthetic fallback).
    """
    cfg = get_live_config()
    ttl = cfg.refresh_interval_minutes * 60
    key = f"{'disagreement' if include_disagreement else 'base'}:{hour}"
    target_hour = None
    if hour > 0:
        target_hour = (_dt.now(_tz.utc).replace(minute=0, second=0, microsecond=0) + _timedelta(hours=hour)).strftime("%Y-%m-%dT%H:00")

    with _grid_lock:
        hit = _grid_cache.get(key)
        if hit and not force_refresh and _time.time() - hit["ts"] < ttl:
            return hit["payload"]

    lat_axis = _linspace(*_GRID_LAT, _GRID_N)
    lon_axis = _linspace(*_GRID_LON, _GRID_N)
    pts = [(la, lo) for la in lat_axis for lo in lon_axis]
    lats = [p[0] for p in pts]
    lons = [p[1] for p in pts]

    try:
        base = _fetch_grid_points(lats, lons, _GRID_VARS, None, cfg, target_hour)
        models: Dict[str, List[Dict[str, Any]]] = {}
        if include_disagreement:
            for m in ("ecmwf_ifs025", "gfs_seamless"):
                models[m] = _fetch_grid_points(lats, lons, "temperature_2m,precipitation,wind_speed_10m", m, cfg, target_hour)
    except Exception as exc:  # network / provider failure
        stale = _grid_cache.get(key)
        if stale:
            payload = dict(stale["payload"])
            payload["freshness"] = "stale_cache"
            return payload
        raise HTTPException(status_code=503, detail=f"Live provider unavailable: {exc}")

    cells = []
    for i, (la, lo) in enumerate(pts):
        cur = (base[i] or {}).get("current") or {}
        speed = cur.get("wind_speed_10m")
        wdir = cur.get("wind_direction_10m")
        u = v = None
        if speed is not None and wdir is not None:
            rad = _math.radians(wdir)
            u = round(-speed * _math.sin(rad), 3)
            v = round(-speed * _math.cos(rad), 3)
        cell = {
            "lat": la,
            "lng": lo,
            "u10": u,
            "v10": v,
            "wind_speed_ms": speed,
            "wind_direction_deg": wdir,
            "temperature_c": cur.get("temperature_2m"),
            "precipitation_mm": cur.get("precipitation"),
            "humidity_pct": cur.get("relative_humidity_2m"),
            "pressure_hpa": cur.get("pressure_msl"),
            "dew_point_c": cur.get("dew_point_2m"),
            "cloud_cover_pct": cur.get("cloud_cover"),
        }
        if include_disagreement:
            e = (models["ecmwf_ifs025"][i] or {}).get("current") or {}
            g = (models["gfs_seamless"][i] or {}).get("current") or {}

            def _diff(key_: str):
                return None if e.get(key_) is None or g.get(key_) is None else round(abs(e[key_] - g[key_]), 3)

            cell["disagreement"] = {
                "temperature_c": _diff("temperature_2m"),
                "wind_speed_ms": _diff("wind_speed_10m"),
                "precipitation_mm": _diff("precipitation"),
            }
        cells.append(cell)

    payload = {
        "bounds": {"lat_min": _GRID_LAT[0], "lat_max": _GRID_LAT[1], "lon_min": _GRID_LON[0], "lon_max": _GRID_LON[1]},
        "lat_axis": lat_axis,
        "lon_axis": lon_axis,
        "provider": cfg.provider_name,
        "disagreement_models": ["ecmwf_ifs025", "gfs_seamless"] if include_disagreement else [],
        "hour": hour,
        "valid_time_utc": ((base[0] or {}).get("current") or {}).get("time"),
        "retrieved_at_utc": _dt.now(_tz.utc).isoformat(),
        "freshness": "fresh",
        "cells": cells,
    }
    with _grid_lock:
        _grid_cache[key] = {"ts": _time.time(), "payload": payload}
    return payload
