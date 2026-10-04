"""
Pydantic data schemas for Live Operational Data & Forecast Results.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class DataFreshnessStatus(str, Enum):
    FRESH = "fresh"
    STALE = "stale"
    EXPIRED = "expired"
    UNAVAILABLE = "unavailable"


class LiveVariableRecord(BaseModel):
    """Normalized live weather record (observation or forecast step)."""
    timestamp: str = Field(description="ISO-8601 UTC timestamp of forecast valid time or observation")
    latitude: float
    longitude: float
    variable: str = Field(description="temperature, wind_speed, rainfall")
    value: float
    unit: str = Field(description="°C, m/s, mm/6h, mm/h")
    provider: str = Field(default="Open-Meteo")
    model: str = Field(default="operational_forecast")
    lead_time_hours: Optional[int] = None
    forecast_initialization_time: Optional[str] = None
    retrieved_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class CurrentConditions(BaseModel):
    """Real-time surface observation / latest analysis."""
    location_name: str
    latitude: float
    longitude: float
    timestamp_utc: str
    temperature_c: float
    wind_speed_ms: float
    wind_direction_deg: Optional[float] = None
    precipitation_mm: float
    relative_humidity_pct: Optional[float] = None
    surface_pressure_hpa: Optional[float] = None
    provider: str = "Open-Meteo"
    model: str = "operational_analysis"
    retrieved_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    freshness: DataFreshnessStatus = DataFreshnessStatus.FRESH
    data_age_minutes: float = 0.0


class LiveForecastStep(BaseModel):
    """Single forecast lead time step in operational output."""
    lead_time_hours: int
    valid_time_utc: str
    temperature_c: float
    temperature_ci90: Optional[List[float]] = None  # [lower, upper]
    wind_speed_ms: float
    wind_speed_ci90: Optional[List[float]] = None
    rainfall_mm: float
    rainfall_exceedance_p5mm: Optional[float] = None
    dominant_model: str = "operational_forecast"
    model_weights: Dict[str, float] = Field(default_factory=dict)
    regime: str = "stable"


class LiveExtremeAlert(BaseModel):
    """Extreme weather threshold indicator."""
    hazard_type: str  # 'heavy_rain', 'extreme_rain', 'high_wind', 'heatwave', 'cold_wave'
    severity: str     # 'advisory', 'watch', 'warning'
    headline: str
    description: str
    threshold: str
    observed_or_forecast_value: str
    lead_time_hours: Optional[int] = None
    timestamp_utc: str


class LiveDisagreementMetric(BaseModel):
    """Model disagreement diagnostic across operational forecast members."""
    variable: str
    lead_time_hours: int
    mean_disagreement: float
    disagreement_level: str  # 'low', 'moderate', 'high', 'unavailable'
    models_compared: List[str]
    interpretation: str


class LiveForecastResult(BaseModel):
    """Complete response returned by the live forecast service & API."""
    status: str = "success"
    mode: str = "live"
    location: Dict[str, Any]
    retrieved_at_utc: str
    initialization_time_utc: str
    freshness: DataFreshnessStatus
    data_age_minutes: float
    provider: str
    models_available: List[str]
    blender_method: str
    current_conditions: CurrentConditions
    forecast_steps: List[LiveForecastStep]
    disagreement: List[LiveDisagreementMetric] = Field(default_factory=list)
    active_alerts: List[LiveExtremeAlert] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    scientific_disclaimer: str = (
        "Operational live forecasts are ingested from open meteorological providers and blended "
        "using SkyCast adaptive algorithms. Live operational data is independent of the historical "
        "WeatherBench 2 research benchmark and has not undergone multi-year scientific reanalysis audit."
    )
