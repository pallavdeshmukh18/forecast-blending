"""
Abstract Base Class for all multi-model forecast blending engines.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
import pandas as pd

from ingestion.schema import validate_forecast_dataframe


class BaseBlender(ABC):
    """
    Abstract interface for forecast blending models.
    """

    def __init__(self, name: str = "base_blender", config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.config = config or {}
        self.is_fitted = False

    def fit(self, historical_forecasts: pd.DataFrame, historical_observations: pd.DataFrame) -> "BaseBlender":
        """
        Fit blending weights or model parameters using historical forecast and observation pairs.
        """
        self.is_fitted = True
        return self

    def fit_verification_data(self, verified_df: pd.DataFrame) -> "BaseBlender":
        """
        Fit blending weights directly from pre-aligned verified forecast-observation records.
        """
        self.is_fitted = True
        return self

    @abstractmethod
    def predict(self, forecasts_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate blended forecasts from multi-model input predictions.

        Parameters
        ----------
        forecasts_df : pd.DataFrame
            Input multi-model forecast records conforming to standard schema.

        Returns
        -------
        pd.DataFrame
            Blended forecast records conforming to standard schema.
        """
        pass

    def get_weights(
        self,
        variable: str,
        lead_time: Optional[int] = None,
        region: Optional[str] = None,
        available_models: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        Retrieve calculated model weights for a specific meteorological context.
        """
        if available_models:
            equal_w = 1.0 / len(available_models)
            return {m: equal_w for m in available_models}
        return {}

    def get_weight_metadata(
        self,
        variable: str,
        lead_time: Optional[int] = None,
        region: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Retrieve diagnostic metadata explaining how weights were derived.
        """
        return {
            "blender": self.name,
            "variable": variable,
            "lead_time": lead_time,
            "region": region,
            "is_fitted": self.is_fitted,
        }

    def get_weight_map(self, forecasts_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate a structured spatial/contextual weight table for visualization:
        columns: ['latitude', 'longitude', 'lead_time', 'variable', 'model_name', 'weight']
        """
        df = validate_forecast_dataframe(forecasts_df, allow_nans=True)
        records = []

        group_cols = ["latitude", "longitude", "lead_time", "variable"]
        if "region" in df.columns:
            group_cols.append("region")

        for keys, grp in df.groupby(group_cols, observed=True):
            if not isinstance(keys, tuple):
                keys = (keys,)
            key_dict = dict(zip(group_cols, keys))
            avail_models = list(grp["model_name"].unique())

            weights = self.get_weights(
                variable=key_dict["variable"],
                lead_time=key_dict["lead_time"],
                region=key_dict.get("region"),
                available_models=avail_models,
            )

            for m_name, w in weights.items():
                rec = {
                    "latitude": key_dict["latitude"],
                    "longitude": key_dict["longitude"],
                    "lead_time": key_dict["lead_time"],
                    "variable": key_dict["variable"],
                    "model_name": m_name,
                    "weight": round(w, 4),
                }
                if "region" in key_dict:
                    rec["region"] = key_dict["region"]
                records.append(rec)

        return pd.DataFrame(records)
