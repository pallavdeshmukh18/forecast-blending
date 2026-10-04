"""
Configuration for the Live Operational Forecasting Layer (SkyCast Live).
"""

import os
from pathlib import Path
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class LiveConfig(BaseModel):
    """
    Configuration parameters for live weather data ingestion, caching, and operational blending.
    """
    # Operational mode
    live_mode_enabled: bool = Field(default=True, description="Enable live operational forecasting")
    provider_name: str = Field(default="Open-Meteo", description="Operational weather data provider")
    api_base_url: str = Field(default="https://api.open-meteo.com/v1/forecast", description="Open-Meteo API endpoint")
    
    # Location defaults (e.g. Mumbai / South Asia representative node)
    default_latitude: float = Field(default=19.0760, description="Default latitude (Mumbai)")
    default_longitude: float = Field(default=72.8777, description="Default longitude (Mumbai)")
    default_location_name: str = Field(default="Mumbai, India", description="Default location name")
    
    # Network & Timeout settings
    request_timeout_seconds: float = Field(default=10.0, description="HTTP timeout for live provider calls")
    max_retries: int = Field(default=3, description="Maximum HTTP retries on transient network failures")
    retry_backoff_factor: float = Field(default=0.5, description="Backoff factor for HTTP retries")
    
    # Refresh & Cache settings
    refresh_interval_minutes: int = Field(default=15, description="Background live data refresh interval")
    fresh_threshold_minutes: int = Field(default=60, description="Age threshold for 'fresh' status (< 1 hour)")
    stale_threshold_minutes: int = Field(default=180, description="Age threshold for 'stale' status (1-3 hours)")
    cache_max_age_minutes: int = Field(default=360, description="Max cache retention (6 hours)")
    cache_dir: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parent.parent / "data" / "live_cache"
    )
    max_cache_files: int = Field(default=50, description="Maximum stored live cache files before cleanup")
    
    # Target variables & supported lead times
    supported_variables: List[str] = Field(
        default_factory=lambda: ["temperature", "wind_speed", "rainfall"]
    )
    supported_lead_times_hours: List[int] = Field(
        default_factory=lambda: [6, 12, 24, 48, 72]
    )
    
    # Alert thresholds
    heavy_rain_threshold_6h_mm: float = 15.0
    extreme_rain_threshold_6h_mm: float = 35.0
    high_wind_threshold_ms: float = 17.0  # ~61 km/h (Gale force)
    high_temp_threshold_c: float = 40.0   # Heatwave threshold
    cold_temp_threshold_c: float = 5.0    # Cold wave threshold


def get_live_config() -> LiveConfig:
    """Load LiveConfig with environment variable overrides."""
    cache_path_env = os.environ.get("LIVE_CACHE_DIR")
    cache_dir = Path(cache_path_env) if cache_path_env else None
    
    return LiveConfig(
        live_mode_enabled=os.environ.get("LIVE_MODE", "true").lower() in ("true", "1", "yes"),
        provider_name=os.environ.get("LIVE_PROVIDER", "Open-Meteo"),
        api_base_url=os.environ.get("LIVE_API_URL", "https://api.open-meteo.com/v1/forecast"),
        default_latitude=float(os.environ.get("LIVE_LATITUDE", "19.0760")),
        default_longitude=float(os.environ.get("LIVE_LONGITUDE", "72.8777")),
        default_location_name=os.environ.get("LIVE_LOCATION_NAME", "Mumbai, India"),
        refresh_interval_minutes=int(os.environ.get("LIVE_REFRESH_MINUTES", "15")),
        fresh_threshold_minutes=int(os.environ.get("LIVE_FRESH_THRESHOLD_MINUTES", "60")),
        stale_threshold_minutes=int(os.environ.get("LIVE_STALE_THRESHOLD_MINUTES", "180")),
        cache_max_age_minutes=int(os.environ.get("LIVE_CACHE_MAX_AGE_MINUTES", "360")),
        cache_dir=cache_dir or (Path(__file__).resolve().parent.parent / "data" / "live_cache"),
    )
