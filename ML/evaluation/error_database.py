"""
Historical Verification Error Database and Model Skill Store.

Provides persistent storage, querying, and analysis of verified forecast-observation
error pairs across models, lead times, regions, seasons, and weather regimes.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd

from configs import AppConfig, get_config
from evaluation.metrics import compute_all_metrics, mean_absolute_error, root_mean_squared_error, mean_bias_error, pearson_correlation


class HistoricalErrorDatabase:
    """
    Persistent repository and querying engine for verified forecast-observation error pairs.
    """

    STANDARD_COLUMNS = [
        "timestamp",
        "latitude",
        "longitude",
        "variable",
        "model_name",
        "lead_time",
        "lead_time_bin",
        "forecast_value",
        "observed_value",
        "error",
        "absolute_error",
        "squared_error",
        "spatial_distance_km",
        "season",
        "weather_regime",
        "region",
    ]

    def __init__(self, data: Optional[pd.DataFrame] = None, config: Optional[AppConfig] = None):
        self.config = config or get_config()
        if data is not None and not data.empty:
            self._df = self._standardize_columns(data)
        else:
            self._df = pd.DataFrame(columns=self.STANDARD_COLUMNS)

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure standard column names and error metrics are computed."""
        out = df.copy()

        # Handle forecast_error vs error alias
        if "forecast_error" in out.columns and "error" not in out.columns:
            out["error"] = out["forecast_error"]
        elif "error" not in out.columns and "forecast_value" in out.columns and "observed_value" in out.columns:
            out["error"] = out["forecast_value"] - out["observed_value"]

        if "absolute_error" not in out.columns and "error" in out.columns:
            out["absolute_error"] = out["error"].abs()

        if "squared_error" not in out.columns and "error" in out.columns:
            out["squared_error"] = out["error"] ** 2

        if "timestamp" in out.columns:
            out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)

        # Ensure optional columns exist if missing
        for col in ["region", "season", "weather_regime", "lead_time_bin", "spatial_distance_km"]:
            if col not in out.columns:
                out[col] = None

        return out

    def add_records(self, verified_df: pd.DataFrame) -> None:
        """
        Add verified forecast-observation pairs to the error database.
        """
        if verified_df.empty:
            return

        std_new = self._standardize_columns(verified_df)
        if self._df.empty:
            self._df = std_new
        else:
            self._df = pd.concat([self._df, std_new], ignore_index=True)

        # Deduplicate and sort chronologically
        dedup_keys = [c for c in ["timestamp", "latitude", "longitude", "variable", "model_name", "lead_time"] if c in self._df.columns]
        self._df = self._df.drop_duplicates(subset=dedup_keys, keep="last")
        if "timestamp" in self._df.columns:
            self._df = self._df.sort_values(by=["timestamp", "lead_time", "variable"]).reset_index(drop=True)

    @property
    def dataframe(self) -> pd.DataFrame:
        """Return raw error records DataFrame."""
        return self._df

    def __len__(self) -> int:
        return len(self._df)

    def save(self, filepath: Union[str, Path], format: Optional[str] = None) -> Path:
        """
        Persist the error database to disk (Parquet or CSV).
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        fmt = (format or path.suffix.lstrip(".")).lower()

        if fmt in ["parquet", "pq"]:
            self._df.to_parquet(path, index=False)
        elif fmt in ["csv", "txt"]:
            self._df.to_csv(path, index=False)
        else:
            # Default to parquet
            self._df.to_parquet(path, index=False)

        return path

    @classmethod
    def load(cls, filepath: Union[str, Path], config: Optional[AppConfig] = None) -> "HistoricalErrorDatabase":
        """
        Load historical error database from disk.
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Historical error database file not found: {path}")

        fmt = path.suffix.lower().lstrip(".")
        if fmt in ["parquet", "pq"]:
            df = pd.read_parquet(path)
        else:
            df = pd.read_csv(path)

        return cls(data=df, config=config)

    def query_records(
        self,
        model_name: Optional[Union[str, List[str]]] = None,
        variable: Optional[str] = None,
        lead_time: Optional[int] = None,
        lead_time_bin: Optional[str] = None,
        region: Optional[str] = None,
        season: Optional[str] = None,
        weather_regime: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Filter error records by arbitrary dimensional attributes.
        """
        sub = self._df.copy()

        if model_name:
            if isinstance(model_name, list):
                sub = sub[sub["model_name"].isin(model_name)]
            else:
                sub = sub[sub["model_name"] == model_name]

        if variable:
            sub = sub[sub["variable"] == variable]

        if lead_time is not None:
            sub = sub[sub["lead_time"] == lead_time]

        if lead_time_bin:
            sub = sub[sub["lead_time_bin"] == lead_time_bin]

        if region and "region" in sub.columns:
            sub = sub[sub["region"] == region]

        if season and "season" in sub.columns:
            sub = sub[sub["season"] == season]

        if weather_regime and "weather_regime" in sub.columns:
            sub = sub[sub["weather_regime"] == weather_regime]

        return sub.reset_index(drop=True)

    def query_metric(
        self,
        model_name: str,
        variable: str,
        metric: str = "rmse",
        lead_time: Optional[int] = None,
        region: Optional[str] = None,
        season: Optional[str] = None,
        weather_regime: Optional[str] = None,
    ) -> float:
        """
        Calculate scalar verification metric for a specified slice.
        """
        records = self.query_records(
            model_name=model_name,
            variable=variable,
            lead_time=lead_time,
            region=region,
            season=season,
            weather_regime=weather_regime,
        )
        if records.empty:
            raise ValueError(
                f"No records found for model='{model_name}', variable='{variable}', lead_time={lead_time}"
            )

        obs = records["observed_value"].values
        pred = records["forecast_value"].values
        m = metric.lower()

        if m == "rmse":
            return root_mean_squared_error(obs, pred)
        elif m == "mae":
            return mean_absolute_error(obs, pred)
        elif m == "bias":
            return mean_bias_error(obs, pred)
        elif m == "correlation":
            return pearson_correlation(obs, pred)
        else:
            raise ValueError(f"Unknown metric: {metric}. Expected 'rmse', 'mae', 'bias', or 'correlation'.")

    def query_lead_time_profile(
        self,
        variable: str,
        metric: str = "rmse",
        models: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Analyze how error grows across forecast lead time horizons for each model.
        """
        sub = self.query_records(model_name=models, variable=variable)
        if sub.empty:
            return pd.DataFrame()

        rows = []
        for (m_name, lt), grp in sub.groupby(["model_name", "lead_time"], observed=True):
            obs = grp["observed_value"].values
            pred = grp["forecast_value"].values
            try:
                metrics = compute_all_metrics(obs, pred)
                rows.append({
                    "model_name": m_name,
                    "variable": variable,
                    "lead_time": lt,
                    "metric_name": metric.upper(),
                    "score": metrics.get(metric.lower(), np.nan),
                    "mae": metrics["mae"],
                    "rmse": metrics["rmse"],
                    "bias": metrics["bias"],
                    "correlation": metrics["correlation"],
                    "sample_count": metrics["sample_count"],
                })
            except ValueError:
                continue

        res = pd.DataFrame(rows)
        if not res.empty:
            res = res.sort_values(by=["lead_time", "score"]).reset_index(drop=True)
        return res

    def query_regime_skill(
        self,
        variable: str,
        weather_regime: str,
        metric: str = "mae",
    ) -> pd.DataFrame:
        """
        Compare model performance specifically under a given weather regime (e.g. convective_storm).
        """
        sub = self.query_records(variable=variable, weather_regime=weather_regime)
        if sub.empty:
            return pd.DataFrame()

        rows = []
        for m_name, grp in sub.groupby("model_name", observed=True):
            obs = grp["observed_value"].values
            pred = grp["forecast_value"].values
            try:
                metrics = compute_all_metrics(obs, pred)
                rows.append({
                    "model_name": m_name,
                    "variable": variable,
                    "weather_regime": weather_regime,
                    "target_metric": metric.upper(),
                    "score": metrics.get(metric.lower(), np.nan),
                    "mae": metrics["mae"],
                    "rmse": metrics["rmse"],
                    "bias": metrics["bias"],
                    "correlation": metrics["correlation"],
                    "sample_count": metrics["sample_count"],
                })
            except ValueError:
                continue

        res = pd.DataFrame(rows)
        if not res.empty:
            res = res.sort_values(by=["score"]).reset_index(drop=True)
        return res

    def query_extreme_threshold_skill(
        self,
        variable: str,
        threshold: float,
        metric: str = "rmse",
    ) -> pd.DataFrame:
        """
        Evaluate which model performs better during high-impact / extreme conditions (e.g. rainfall >= 35.5mm).
        """
        sub = self.query_records(variable=variable)
        # Filter for observations or forecasts exceeding threshold
        extreme_mask = (sub["observed_value"] >= threshold) | (sub["forecast_value"] >= threshold)
        ext_sub = sub[extreme_mask]

        if ext_sub.empty:
            return pd.DataFrame()

        rows = []
        for m_name, grp in ext_sub.groupby("model_name", observed=True):
            obs = grp["observed_value"].values
            pred = grp["forecast_value"].values
            try:
                metrics = compute_all_metrics(obs, pred)
                rows.append({
                    "model_name": m_name,
                    "variable": variable,
                    "threshold": threshold,
                    "score": metrics.get(metric.lower(), np.nan),
                    "mae": metrics["mae"],
                    "rmse": metrics["rmse"],
                    "bias": metrics["bias"],
                    "correlation": metrics["correlation"],
                    "extreme_events_count": len(grp),
                })
            except ValueError:
                continue

        res = pd.DataFrame(rows)
        if not res.empty:
            res = res.sort_values(by=["score"]).reset_index(drop=True)
        return res
