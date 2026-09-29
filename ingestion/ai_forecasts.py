"""
AI / Machine Learning Weather Model ingestion interfaces and providers.

Defines the specialized forecast source class and adapter for data-driven neural weather models
(e.g., Google DeepMind GraphCast, Microsoft ClimaX, Huawei Pangu-Weather, NVIDIA FourCastNet).
"""

from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd

from ingestion.providers import AIProvider, BaseForecastProvider


class AIWeatherForecastSource(AIProvider):
    """
    Source adapter for neural and machine-learned meteorological forecasting models.
    Maintains compatibility with Phase 1 AIWeatherForecastSource.
    """

    def __init__(
        self,
        source_name: str = "ai_model_c",
        architecture: str = "graph_neural_network",
        resolution_km: float = 25.0,
        config: Optional[Dict] = None,
    ):
        super().__init__(
            provider_name=source_name,
            default_model_name=source_name,
            architecture=architecture,
        )
        self.source_name = source_name
        self.resolution_km = resolution_km

    def fetch(
        self,
        run_times: List[datetime],
        lead_times: List[int],
        variables: Optional[List[str]] = None,
        region: Optional[str] = None,
    ) -> pd.DataFrame:
        return self.fetch_forecast(run_times=run_times, lead_times=lead_times, variables=variables, region=region)


__all__ = [
    "AIProvider",
    "AIWeatherForecastSource",
]
