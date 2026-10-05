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
    Supports both standard single-feed and multi-model operational feeds.
    
    Returns:
        (standard_forecast_df, list_of_step_dicts)
    """
    hourly = raw_data.get("hourly", {})
    if not hourly or "time" not in hourly:
        return pd.DataFrame(), []
        
    times = [parse_iso_timestamp(t) for t in hourly["time"]]
    lat = float(raw_data.get("_latitude", raw_data.get("latitude", 0.0)))
    lon = float(raw_data.get("_longitude", raw_data.get("longitude", 0.0)))
    provider = raw_data.get("_provider", "Open-Meteo")
    
    # Check for multi-model keys in hourly
    model_keys: Dict[str, Dict[str, str]] = {}
    for k in hourly.keys():
        if k.startswith("temperature_2m_"):
            m_name = k.replace("temperature_2m_", "")
            model_keys[m_name] = {
                "temp": k,
                "wind": f"wind_speed_10m_{m_name}",
                "precip": f"precipitation_{m_name}",
            }

    # Fallback to single operational forecast if no multi-model suffixes
    if not model_keys:
        model_keys["operational_forecast"] = {
            "temp": "temperature_2m",
            "wind": "wind_speed_10m",
            "precip": "precipitation",
        }

    init_time = times[0] if times else datetime.now(timezone.utc)
    leads = target_lead_times or [6, 12, 24, 48, 72]
    
    records = []
    step_summaries = []

    for lead in leads:
        if lead < len(times):
            v_time = times[lead]
            start_idx = max(0, lead - 5)
            
            lead_model_fcs = {}
            for m_name, cols in model_keys.items():
                m_temps = hourly.get(cols["temp"], hourly.get("temperature_2m", []))
                m_winds = hourly.get(cols["wind"], hourly.get("wind_speed_10m", []))
                m_precips = hourly.get(cols["precip"], hourly.get("precipitation", []))
                
                t_val = float(m_temps[lead]) if lead < len(m_temps) and m_temps[lead] is not None else np.nan
                w_val = float(m_winds[lead]) if lead < len(m_winds) and m_winds[lead] is not None else np.nan
                r_val_6h = sum([max(0.0, float(v)) for v in m_precips[start_idx : lead + 1] if v is not None]) if len(m_precips) > lead else 0.0
                
                clean_m = "ecmwf" if "ecmwf" in m_name else "gfs" if "gfs" in m_name else "icon" if "icon" in m_name else m_name

                lead_model_fcs[clean_m] = {
                    "temperature_c": round(t_val, 2) if not np.isnan(t_val) else 0.0,
                    "wind_speed_ms": round(w_val, 2) if not np.isnan(w_val) else 0.0,
                    "rainfall_mm": round(r_val_6h, 2),
                }

                # Standard DataFrame rows
                records.append({
                    "timestamp": v_time,
                    "latitude": lat,
                    "longitude": lon,
                    "variable": "temperature",
                    "forecast_value": t_val,
                    "model_name": clean_m,
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
                    "model_name": clean_m,
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
                    "model_name": clean_m,
                    "lead_time": lead,
                    "run_time": init_time,
                    "provider": provider,
                })

            all_temps = [f["temperature_c"] for f in lead_model_fcs.values() if not np.isnan(f["temperature_c"])]
            all_winds = [f["wind_speed_ms"] for f in lead_model_fcs.values() if not np.isnan(f["wind_speed_ms"])]
            all_rains = [f["rainfall_mm"] for f in lead_model_fcs.values() if not np.isnan(f["rainfall_mm"])]
            
            mean_t = float(np.mean(all_temps)) if all_temps else 0.0
            mean_w = float(np.mean(all_winds)) if all_winds else 0.0
            mean_r = float(np.mean(all_rains)) if all_rains else 0.0

            step_summaries.append({
                "lead_time_hours": lead,
                "valid_time_utc": v_time.isoformat(),
                "temperature_c": round(mean_t, 2),
                "wind_speed_ms": round(mean_w, 2),
                "rainfall_mm": round(mean_r, 2),
                "models": lead_model_fcs,
            })
            
    df = pd.DataFrame(records)
    return df, step_summaries
