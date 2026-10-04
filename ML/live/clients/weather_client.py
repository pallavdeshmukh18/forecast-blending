"""
Open-Meteo operational weather data client.
Fetches real-time surface conditions and multi-model forecast feeds without requiring API keys.
"""

from datetime import datetime, timezone
import json
import logging
import time
from typing import Any, Dict, List, Optional
import urllib.parse
import urllib.request
import urllib.error

from ..config import LiveConfig, get_live_config

logger = logging.getLogger(__name__)


class OpenMeteoClient:
    """
    HTTP client for the Open-Meteo operational weather forecast API.
    """

    def __init__(self, config: Optional[LiveConfig] = None):
        self.config = config or get_live_config()
        self.base_url = self.config.api_base_url
        self.provider_name = "Open-Meteo"

    def fetch_live_data(
        self,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        models: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Fetch current conditions and hourly forecast steps from Open-Meteo.
        
        Args:
            latitude: Latitude of location (defaults to config).
            longitude: Longitude of location (defaults to config).
            models: Optional list of specific NWP models to request (e.g. ['gfs_seamless', 'ecmwf_ifs025']).
            
        Returns:
            Raw parsed JSON dictionary from provider with attached retrieval timestamp.
        """
        lat = latitude if latitude is not None else self.config.default_latitude
        lon = longitude if longitude is not None else self.config.default_longitude
        
        params: Dict[str, Any] = {
            "latitude": f"{lat:.4f}",
            "longitude": f"{lon:.4f}",
            "current": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m,wind_direction_10m,surface_pressure",
            "hourly": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m,wind_direction_10m,surface_pressure,cape",
            "wind_speed_unit": "ms",
            "precipitation_unit": "mm",
            "timeformat": "iso8601",
            "timezone": "UTC",
            "forecast_days": 4,
        }
        
        if models and len(models) > 0:
            params["models"] = ",".join(models)
            
        query_string = urllib.parse.urlencode(params)
        request_url = f"{self.base_url}?{query_string}"
        
        headers = {
            "User-Agent": "SkyCast-Weather-Intelligence/0.6.0 (Research-Operational Gateway)",
            "Accept": "application/json",
        }
        
        req = urllib.request.Request(request_url, headers=headers)
        
        last_exception = None
        retrieved_at = datetime.now(timezone.utc).isoformat()
        
        for attempt in range(1, self.config.max_retries + 1):
            try:
                logger.info(f"LIVE_FETCH_STARTED: Attempt {attempt}/{self.config.max_retries} -> {request_url}")
                t0 = time.time()
                with urllib.request.urlopen(req, timeout=self.config.request_timeout_seconds) as response:
                    status_code = response.getcode()
                    if status_code != 200:
                        raise ValueError(f"HTTP {status_code} returned by provider")
                    raw_body = response.read().decode("utf-8")
                    data = json.loads(raw_body)
                    latency_ms = round((time.time() - t0) * 1000, 2)
                    
                    data["_retrieved_at"] = retrieved_at
                    data["_provider"] = self.provider_name
                    data["_request_url"] = request_url
                    data["_latency_ms"] = latency_ms
                    data["_latitude"] = lat
                    data["_longitude"] = lon
                    
                    logger.info(f"LIVE_FETCH_SUCCESS: Received in {latency_ms}ms (coords: {lat}, {lon})")
                    return data
                    
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError) as e:
                last_exception = e
                logger.warning(f"LIVE_FETCH_RETRY: Attempt {attempt} failed: {e}")
                if attempt < self.config.max_retries:
                    time.sleep(self.config.retry_backoff_factor * (2 ** (attempt - 1)))
                    
        logger.error(f"LIVE_FETCH_FAILED: All {self.config.max_retries} attempts failed: {last_exception}")
        raise ConnectionError(f"Failed to fetch live weather data from {self.provider_name}: {last_exception}")
