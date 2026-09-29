"""
Weather Regime Feature Extraction and Rule-Based Atmospheric State Classifier.
Extracts physically consistent meteorological features and classifies forecast-relevant
atmospheric states (stable, convective, high_wind, heavy_rain, hot_extreme, cold_extreme, transition).
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from configs import AppConfig, get_config


REGIME_TO_ID: Dict[str, int] = {
    "stable": 0,
    "convective": 1,
    "high_wind": 2,
    "heavy_rain": 3,
    "hot_extreme": 4,
    "cold_extreme": 5,
    "transition": 6,
    "unknown": -1,
}

ID_TO_REGIME: Dict[int, str] = {v: k for k, v in REGIME_TO_ID.items()}


def extract_atmospheric_features(
    df: pd.DataFrame,
    config: Optional[AppConfig] = None,
) -> pd.DataFrame:
    """
    Extract atmospheric and temporal features from input meteorological DataFrame.
    Gracefully handles missing columns and computes cyclical temporal signals.
    """
    if df.empty:
        return df.copy()

    cfg = config or get_config()
    out = df.copy()

    # 1. Ensure UTC datetime and extract temporal cycles
    if "timestamp" in out.columns:
        ts = pd.to_datetime(out["timestamp"], utc=True)
        out["hour"] = ts.dt.hour
        out["month"] = ts.dt.month
        out["day_of_year"] = ts.dt.dayofyear

        # Cyclical diurnal features (24-hour cycle)
        out["diurnal_sin"] = np.sin(2.0 * np.pi * out["hour"] / 24.0)
        out["diurnal_cos"] = np.cos(2.0 * np.pi * out["hour"] / 24.0)

        # Cyclical annual/seasonal features (365.25-day cycle)
        out["seasonal_sin"] = np.sin(2.0 * np.pi * out["day_of_year"] / 365.25)
        out["seasonal_cos"] = np.cos(2.0 * np.pi * out["day_of_year"] / 365.25)

    # 2. Extract wide variables if long format is provided
    # If the dataframe has 'variable' and 'forecast_value' or 'observed_value'
    val_col = "observed_value" if "observed_value" in out.columns else ("forecast_value" if "forecast_value" in out.columns else None)

    if "variable" in out.columns and val_col and "temperature" not in out.columns:
        # Check if we have multiple variables in long format
        id_cols = [c for c in ["timestamp", "latitude", "longitude", "lead_time", "region", "station_id"] if c in out.columns]
        if id_cols:
            wide_piv = out.pivot_table(index=id_cols, columns="variable", values=val_col, aggfunc="first").reset_index()
            wide_piv.columns.name = None
            out = pd.merge(out, wide_piv, on=id_cols, how="left", suffixes=("", "_wide"))

    # 3. Dynamic atmospheric gradients and rolling metrics (if temporal sequence available)
    if "timestamp" in out.columns and "temperature" in out.columns:
        sort_keys = [c for c in ["latitude", "longitude", "timestamp"] if c in out.columns]
        out = out.sort_values(sort_keys).reset_index(drop=True)

        # Temperature step delta
        if "station_id" in out.columns:
            out["temperature_gradient"] = out.groupby("station_id")["temperature"].diff().fillna(0.0)
        else:
            out["temperature_gradient"] = out["temperature"].diff().fillna(0.0)

    if "rainfall" in out.columns:
        out["rainfall"] = out["rainfall"].clip(lower=0.0)
        if "station_id" in out.columns:
            out["rolling_precipitation"] = (
                out.groupby("station_id")["rainfall"].rolling(window=4, min_periods=1).sum().reset_index(drop=True)
            )
        else:
            out["rolling_precipitation"] = out["rainfall"].rolling(window=4, min_periods=1).sum()

    if "wind_speed" in out.columns:
        out["wind_speed"] = out["wind_speed"].clip(lower=0.0)

    return out


def classify_rule_based_regime(
    data: Union[pd.DataFrame, pd.Series, dict],
    config: Optional[AppConfig] = None,
) -> Union[pd.DataFrame, str]:
    """
    Classify forecast-relevant weather regimes using deterministic threshold priority rules.

    Categories:
    - 'heavy_rain': Extreme precipitation (rainfall >= heavy_rain_mm)
    - 'convective': High precipitation / convective activity (rainfall >= convective_rain_mm)
    - 'high_wind': Gale / strong breeze (wind_speed >= high_wind_mps)
    - 'hot_extreme': Extreme temperature / heatwave (temperature >= hot_temperature_c)
    - 'cold_extreme': Low temperature / cold snap (temperature <= cold_temperature_c)
    - 'transition': Rapid temperature gradient (|temp_gradient| >= 4.0 °C/step)
    - 'stable': Calm / standard baseline atmospheric conditions
    """
    cfg = config or get_config()
    thresh = cfg.weather_regime.thresholds

    heavy_rain_th = thresh.heavy_rain_mm
    conv_rain_th = thresh.convective_rain_mm
    high_wind_th = thresh.high_wind_mps
    hot_temp_th = thresh.hot_temperature_c
    cold_temp_th = thresh.cold_temperature_c

    def _eval_row(row: Union[pd.Series, dict]) -> str:
        temp = row.get("temperature", np.nan)
        rain = row.get("rainfall", np.nan)
        wind = row.get("wind_speed", np.nan)
        temp_grad = abs(row.get("temperature_gradient", 0.0))

        # Fallback to single row 'variable'/'forecast_value'/'observed_value' if wide cols missing
        if pd.isna(temp) and pd.isna(rain) and pd.isna(wind):
            var = row.get("variable", "")
            val = row.get("observed_value", row.get("forecast_value", np.nan))
            if var == "temperature":
                temp = val
            elif var == "rainfall":
                rain = val
            elif var == "wind_speed":
                wind = val

        if pd.notna(rain) and rain >= heavy_rain_th:
            return "heavy_rain"
        elif pd.notna(rain) and rain >= conv_rain_th:
            return "convective"
        elif pd.notna(wind) and wind >= high_wind_th:
            return "high_wind"
        elif pd.notna(temp) and temp >= hot_temp_th:
            return "hot_extreme"
        elif pd.notna(temp) and temp <= cold_temp_th:
            return "cold_extreme"
        elif temp_grad >= 4.0:
            return "transition"
        elif pd.isna(temp) and pd.isna(rain) and pd.isna(wind):
            return "unknown"
        else:
            return "stable"

    # If single Series or dict passed, return single regime string
    if isinstance(data, (pd.Series, dict)):
        return _eval_row(data)

    df = data
    if df.empty:
        return df.copy()

    feat_df = extract_atmospheric_features(df, config=cfg)
    regimes = [_eval_row(row) for _, row in feat_df.iterrows()]
    regime_ids = [REGIME_TO_ID.get(r, 0) for r in regimes]

    out_df = df.copy()
    out_df["weather_regime"] = regimes
    out_df["regime_id"] = regime_ids
    out_df["regime_confidence"] = None  # None for rule-based as required

    # Optional temporal smoothing for regime stability if configured
    if cfg.weather_regime.smoothing_window > 1 and len(out_df) >= cfg.weather_regime.smoothing_window:
        smoothed = (
            out_df["regime_id"]
            .rolling(window=cfg.weather_regime.smoothing_window, min_periods=1, center=True)
            .median()
            .round()
            .astype(int)
        )
        out_df["regime_id"] = smoothed
        out_df["weather_regime"] = [ID_TO_REGIME.get(int(i), "stable") for i in smoothed]

    return out_df



class RuleBasedRegimeClassifier:
    """
    Deterministic rule-based weather regime classifier.
    """

    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config or get_config()

    def classify(self, df: pd.DataFrame) -> pd.DataFrame:
        return classify_rule_based_regime(df, config=self.config)

    @classmethod
    def get_regime_names(cls) -> List[str]:
        return ["stable", "convective", "high_wind", "heavy_rain", "hot_extreme", "cold_extreme", "transition"]


__all__ = [
    "REGIME_TO_ID",
    "ID_TO_REGIME",
    "extract_atmospheric_features",
    "classify_rule_based_regime",
    "RuleBasedRegimeClassifier",
]
