"""
Normalization and parsing routines for raw live meteorological data feeds.
"""

from datetime import datetime, timezone
import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from .schemas import CurrentConditions, LiveForecastStep, LiveVariableRecord, DataFreshnessStatus


def parse_iso_timestamp(ts_str: str) -> datetime:
    """Parse ISO timestamp ensuring UTC timezone."""
    # Handle timestamps ending with 'Z' or offset
    ts_clean = ts_str.replace("Z", "+00:00")
    dt = datetime.fromisoformat(ts_clean)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt


def normalize_current_conditions(raw_data: Dict[str, Any], location_name: str = "Configured Node") -> CurrentConditions:
    """Extract and normalize CurrentConditions from raw provider response."""
    current = raw_data.get("current", {})
    units = raw_data.get("current_units", {})
    
    # Temperature normalization (°C)
    temp_val = float(current.get("temperature_2m", 0.0))
    if units.get("temperature_2m") == "K":
        temp_val -= 273.15
    elif units.get("temperature_2m") == "°F":
        temp_val = (temp_val - 32) * 5 / 9
        
    # Wind speed normalization (m/s)
    wind_val = float(current.get("wind_speed_10m", 0.0))
    if units.get("wind_speed_10m") in ["km/h", "kmh"]:
        wind_val /= 3.6
    elif units.get("wind_speed_10m") == "kn":
        wind_val *= 0.514444
    elif units.get("wind_speed_10m") == "mph":
        wind_val *= 0.44704
        
    precip_val = max(0.0, float(current.get("precipitation", 0.0)))
    
    ts_str = current.get("time") or datetime.now(timezone.utc).isoformat()
    dt_utc = parse_iso_timestamp(ts_str)
    
    return CurrentConditions(
        location_name=location_name,
        latitude=float(raw_data.get("_latitude", raw_data.get("latitude", 0.0))),
        longitude=float(raw_data.get("_longitude", raw_data.get("longitude", 0.0))),
        timestamp_utc=dt_utc.isoformat(),
        temperature_c=round(temp_val, 2),
        wind_speed_ms=round(wind_val, 2),
        wind_direction_deg=current.get("wind_direction_10m"),
        precipitation_mm=round(precip_val, 2),
        relative_humidity_pct=current.get("relative_humidity_2m"),
        surface_pressure_hpa=current.get("surface_pressure"),
        provider=raw_data.get("_provider", "Open-Meteo"),
        model="operational_analysis",
        retrieved_at=raw_data.get("_retrieved_at", datetime.now(timezone.utc).isoformat()),
        freshness=DataFreshnessStatus.FRESH,
    )


def normalize_hourly_forecasts(
    raw_data: Dict[str, Any],
    target_lead_times: Optional[List[int]] = None,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """
    Extract and structure hourly forecast records and calculate 6-hour accumulations.
    
    Returns:
        (standard_forecast_df, list_of_step_dicts)
    """
    hourly = raw_data.get("hourly", {})
    if not hourly or "time" not in hourly:
        return pd.DataFrame(), []
        
    times = [parse_iso_timestamp(t) for t in hourly["time"]]
    temps = [float(v) if v is not None else np.nan for v in hourly.get("temperature_2m", [])]
    winds = [float(v) if v is not None else np.nan for v in hourly.get("wind_speed_10m", [])]
    precips = [max(0.0, float(v)) if v is not None else 0.0 for v in hourly.get("precipitation", [])]
    
    lat = float(raw_data.get("_latitude", raw_data.get("latitude", 0.0)))
    lon = float(raw_data.get("_longitude", raw_data.get("longitude", 0.0)))
    provider = raw_data.get("_provider", "Open-Meteo")
    model_name = "operational_forecast"
    
    init_time = times[0] if times else datetime.now(timezone.utc)
    leads = target_lead_times or [6, 12, 24, 48, 72]
    
    records = []
    step_summaries = []
    
    for lead in leads:
        if lead < len(times):
            v_time = times[lead]
            t_val = temps[lead]
            w_val = winds[lead]
            
            # 6-hour precipitation accumulation: sum of the preceding 6 hourly precipitation values
            start_idx = max(0, lead - 5)
            r_val_6h = sum(precips[start_idx : lead + 1])
            
            # Standard DataFrame rows
            records.append({
                "timestamp": v_time,
                "latitude": lat,
                "longitude": lon,
                "variable": "temperature",
                "forecast_value": t_val,
                "model_name": model_name,
                "lead_time": lead,
                "run_time": init_time,
                "provider": provider,
            })
            records.append({
                "timestamp": v_time,
                "latitude": lat,
                "longitude": lon,
                "variable": "wind_speed",
                "forecast_value": w_val,
                "model_name": model_name,
                "lead_time": lead,
                "run_time": init_time,
                "provider": provider,
            })
            records.append({
                "timestamp": v_time,
                "latitude": lat,
                "longitude": lon,
                "variable": "rainfall",
                "forecast_value": r_val_6h,
                "model_name": model_name,
                "lead_time": lead,
                "run_time": init_time,
                "provider": provider,
            })
            
            step_summaries.append({
                "lead_time_hours": lead,
                "valid_time_utc": v_time.isoformat(),
                "temperature_c": round(t_val, 2) if not np.isnan(t_val) else 0.0,
                "wind_speed_ms": round(w_val, 2) if not np.isnan(w_val) else 0.0,
                "rainfall_mm": round(r_val_6h, 2),
            })
            
    df = pd.DataFrame(records)
    return df, step_summaries
