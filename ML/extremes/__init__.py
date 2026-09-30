"""
Extreme weather guidance and threshold detection module.
"""

from extremes.rainfall import classify_rainfall_intensity, detect_extreme_rainfall
from extremes.heatwave import classify_heatwave, detect_heatwave_events
from extremes.wind import classify_wind_speed, detect_wind_extremes

__all__ = [
    "classify_rainfall_intensity",
    "detect_extreme_rainfall",
    "classify_heatwave",
    "detect_heatwave_events",
    "classify_wind_speed",
    "detect_wind_extremes",
]
