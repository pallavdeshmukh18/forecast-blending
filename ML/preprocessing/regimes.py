"""
Weather Regime Classification interface and deterministic placeholder implementation.
Establishes the contract for future synoptic regime classification (e.g. convective,
synoptic gale, calm/heat, or standard) without changing the data schema.
"""

from typing import Optional
import pandas as pd


class WeatherRegimeClassifier:
    """
    Interface and placeholder classifier for synoptic meteorological weather regimes.
    Future phases will implement clustering / neural classification of atmospheric patterns.
    """

    def __init__(self, default_regime: str = "standard"):
        self.default_regime = default_regime

    def classify(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Assign a synoptic weather regime tag to each record in a forecast/observation DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame containing meteorological variables.

        Returns
        -------
        pd.DataFrame
            DataFrame with populated 'weather_regime' column.
        """
        out = df.copy()

        # If already populated with non-null values, keep them
        if "weather_regime" in out.columns and out["weather_regime"].notna().any():
            out["weather_regime"] = out["weather_regime"].fillna(self.default_regime)
            return out

        # Simple deterministic rule-based placeholder based on variable values if present
        if "variable" in out.columns and "forecast_value" in out.columns:
            regimes = []
            for _, row in out.iterrows():
                var = row.get("variable", "")
                val = row.get("forecast_value", 0.0)
                if var == "rainfall" and val >= 35.5:
                    regimes.append("convective_storm")
                elif var == "wind_speed" and val >= 17.2:
                    regimes.append("synoptic_gale")
                elif var == "temperature" and val >= 40.0:
                    regimes.append("extreme_heat")
                else:
                    regimes.append(self.default_regime)
            out["weather_regime"] = regimes
        else:
            out["weather_regime"] = self.default_regime

        return out
