"""
Deterministic Synthetic Weather Dataset Generator.

Generates physically coherent synthetic observations and multi-model forecast datasets
with realistic error characteristics for baseline benchmarking and blending development.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from configs import AppConfig, get_config
from ingestion.nwp import ForecastSource
from ingestion.observations import ObservationSource
from ingestion.schema import validate_forecast_dataframe, validate_observation_dataframe


class SyntheticDataGenerator:
    """
    Deterministic meteorological synthetic dataset generator.
    """

    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config or get_config()
        self.seed = self.config.synthetic_data.seed

    def _get_station_metadata(self) -> List[Dict]:
        """Generate station coordinates across defined geographical regions."""
        stations = []
        for region_name, region_cfg in self.config.regions.items():
            lats = np.linspace(region_cfg.lat_min + 1.0, region_cfg.lat_max - 1.0, self.config.synthetic_data.stations_per_region)
            lons = np.linspace(region_cfg.lon_min + 1.0, region_cfg.lon_max - 1.0, self.config.synthetic_data.stations_per_region)
            for idx, (lat, lon) in enumerate(zip(lats, lons)):
                station_id = f"STN_{region_name.upper()}_{idx + 1:02d}"
                stations.append({
                    "station_id": station_id,
                    "region": region_name,
                    "latitude": round(float(lat), 4),
                    "longitude": round(float(lon), 4),
                })
        return stations

    def generate(
        self,
        seed: Optional[int] = None,
        save_to_disk: bool = False,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate synthetic ground-truth observations and aligned multi-model forecasts.

        Parameters
        ----------
        seed : Optional[int]
            Random seed for reproducibility. Defaults to config seed (42).
        save_to_disk : bool, default False
            Whether to persist generated datasets to parquet files in data paths.

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            (observations_df, forecasts_df) conforming to standard schemas.
        """
        rng_seed = seed if seed is not None else self.seed
        rng = np.random.default_rng(rng_seed)

        start_dt = datetime.fromisoformat(self.config.synthetic_data.start_date)
        end_dt = datetime.fromisoformat(self.config.synthetic_data.end_date)
        step_hours = self.config.synthetic_data.time_step_hours
        lead_times = self.config.lead_times_hours

        # Timestamps
        num_steps = int((end_dt - start_dt).total_seconds() // (step_hours * 3600)) + 1
        timestamps = [start_dt + timedelta(hours=i * step_hours) for i in range(num_steps)]

        stations = self._get_station_metadata()

        obs_records: List[Dict] = []
        forecast_records: List[Dict] = []

        # Generate continuous base weather fields for each station
        for stn in stations:
            lat = stn["latitude"]
            lon = stn["longitude"]
            region = stn["region"]
            stn_id = stn["station_id"]

            # Regional base climatology offsets
            if region == "coastal":
                base_temp = 28.0
                diurnal_amp = 4.0
                base_wind = 7.0
                rain_prob = 0.35
            elif region == "plains":
                base_temp = 24.0
                diurnal_amp = 10.0
                base_wind = 4.5
                rain_prob = 0.20
            else:  # mountain
                base_temp = 12.0
                diurnal_amp = 6.0
                base_wind = 8.5
                rain_prob = 0.40

            for t_idx, ts in enumerate(timestamps):
                hour = ts.hour
                day_of_year = ts.timetuple().tm_yday

                # Diurnal temperature cycle + synoptic wave
                diurnal_factor = np.sin((hour - 6) / 24.0 * 2 * np.pi)
                synoptic_wave = 3.0 * np.sin(t_idx / 8.0)
                temp_true = base_temp + diurnal_amp * diurnal_factor + synoptic_wave + rng.normal(0, 0.8)
                temp_true = float(np.clip(temp_true, -20.0, 50.0))

                # Synoptic regime & Rain generation (gamma distribution + convective burst)
                is_convective_day = (t_idx % 16) in [6, 7, 13]  # Periodic convective instability
                weather_regime = "convective" if is_convective_day else ("synoptic_active" if synoptic_wave > 1.5 else "quiescent")

                effective_rain_prob = min(0.85, rain_prob * 2.2 if is_convective_day else rain_prob)
                if rng.uniform(0, 1) < effective_rain_prob:
                    scale = 18.0 if is_convective_day else 6.0
                    rain_true = float(rng.gamma(shape=1.5, scale=scale))
                else:
                    rain_true = 0.0

                # Wind generation (Weibull-like, gusty during convection)
                wind_base = base_wind + (4.0 if is_convective_day else 0.0) + abs(synoptic_wave) * 0.8
                wind_true = float(max(0.5, rng.weibull(2.0) * wind_base + rng.normal(0, 0.5)))

                # Append ground truth observations
                obs_records.append({
                    "timestamp": ts,
                    "latitude": lat,
                    "longitude": lon,
                    "variable": "temperature",
                    "observed_value": round(temp_true, 2),
                    "station_id": stn_id,
                    "region": region,
                })
                obs_records.append({
                    "timestamp": ts,
                    "latitude": lat,
                    "longitude": lon,
                    "variable": "rainfall",
                    "observed_value": round(rain_true, 2),
                    "station_id": stn_id,
                    "region": region,
                })
                obs_records.append({
                    "timestamp": ts,
                    "latitude": lat,
                    "longitude": lon,
                    "variable": "wind_speed",
                    "observed_value": round(wind_true, 2),
                    "station_id": stn_id,
                    "region": region,
                })

                # Generate multi-model forecasts for each lead time
                for lt in lead_times:
                    run_time = ts - timedelta(hours=lt)

                    # Model A: High-res NWP - Excellent short range, fast error growth, warm bias
                    # Temperature
                    err_a_temp = 0.4 + (0.04 * lt) * rng.normal(0, 1) + rng.normal(0.3, 0.4)
                    val_a_temp = float(np.clip(temp_true + err_a_temp, -40.0, 55.0))
                    # Rainfall (overpredicts light rain, good on organized events)
                    err_a_rain = (0.03 * lt) * rng.normal(0, 2.0)
                    val_a_rain = float(max(0.0, rain_true + (0.5 if rain_true == 0 and rng.uniform(0, 1) < 0.2 else 0.0) + err_a_rain))
                    # Wind speed (accurate physics peaks)
                    err_a_wind = (0.02 * lt) * rng.normal(0, 0.8) + rng.normal(0.2, 0.5)
                    val_a_wind = float(max(0.0, wind_true + err_a_wind))

                    # Model B: Global NWP - Medium baseline skill, slow error growth, cold bias
                    # Temperature
                    err_b_temp = -0.3 + (0.018 * lt) * rng.normal(0, 1) + rng.normal(-0.3, 0.8)
                    val_b_temp = float(np.clip(temp_true + err_b_temp, -40.0, 55.0))
                    # Rainfall (diffusive, underestimates extreme peaks at short range, stable at long range)
                    err_b_rain = -0.15 * rain_true + (0.015 * lt) * rng.normal(0, 1.8)
                    val_b_rain = float(max(0.0, rain_true + err_b_rain))
                    # Wind speed
                    err_b_wind = (0.015 * lt) * rng.normal(0, 0.9) - 0.2
                    val_b_wind = float(max(0.0, wind_true + err_b_wind))

                    # Model C: Neural AI Model - Non-linear skill, strong on extreme rainfall, smooth wind
                    # Temperature
                    err_c_temp = (0.025 * lt) * rng.normal(0, 0.9)
                    val_c_temp = float(np.clip(temp_true + err_c_temp, -40.0, 55.0))
                    # Rainfall (superior skill during heavy convective events)
                    err_c_rain = (0.05 * np.sqrt(lt)) * rng.normal(0, 1.2)
                    val_c_rain = float(max(0.0, rain_true + err_c_rain))
                    # Wind speed (mildly smoothed peak winds)
                    val_c_wind = float(max(0.0, wind_true * 0.92 + (0.02 * lt) * rng.normal(0, 0.7)))

                    season_tag = "winter" if ts.month in [12, 1, 2] else "summer"

                    model_outputs = [
                        ("nwp_model_a", val_a_temp, val_a_rain, val_a_wind),
                        ("nwp_model_b", val_b_temp, val_b_rain, val_b_wind),
                        ("ai_model_c", val_c_temp, val_c_rain, val_c_wind),
                    ]

                    for m_name, m_temp, m_rain, m_wind in model_outputs:
                        forecast_records.append({
                            "timestamp": ts,
                            "latitude": lat,
                            "longitude": lon,
                            "lead_time": lt,
                            "model_name": m_name,
                            "variable": "temperature",
                            "forecast_value": round(m_temp, 2),
                            "run_time": run_time,
                            "region": region,
                            "season": season_tag,
                            "weather_regime": weather_regime,
                        })
                        forecast_records.append({
                            "timestamp": ts,
                            "latitude": lat,
                            "longitude": lon,
                            "lead_time": lt,
                            "model_name": m_name,
                            "variable": "rainfall",
                            "forecast_value": round(m_rain, 2),
                            "run_time": run_time,
                            "region": region,
                            "season": season_tag,
                            "weather_regime": weather_regime,
                        })
                        forecast_records.append({
                            "timestamp": ts,
                            "latitude": lat,
                            "longitude": lon,
                            "lead_time": lt,
                            "model_name": m_name,
                            "variable": "wind_speed",
                            "forecast_value": round(m_wind, 2),
                            "run_time": run_time,
                            "region": region,
                            "season": season_tag,
                            "weather_regime": weather_regime,
                        })

        obs_df = pd.DataFrame(obs_records)
        forecasts_df = pd.DataFrame(forecast_records)

        obs_df = validate_observation_dataframe(obs_df)
        forecasts_df = validate_forecast_dataframe(forecasts_df)

        if save_to_disk:
            import os
            os.makedirs(self.config.paths.processed_data_dir, exist_ok=True)
            os.makedirs(self.config.paths.forecasts_data_dir, exist_ok=True)
            obs_df.to_parquet(self.config.paths.synthetic_observations_file, index=False)
            forecasts_df.to_parquet(self.config.paths.synthetic_forecasts_file, index=False)

        return obs_df, forecasts_df


class SyntheticObservationSource(ObservationSource):
    """
    Observation source adapter backed by the deterministic synthetic generator.
    """

    def __init__(self, source_name: str = "synthetic_observations", config: Optional[AppConfig] = None):
        super().__init__(source_name=source_name)
        self.app_config = config or get_config()
        self.generator = SyntheticDataGenerator(self.app_config)

    def fetch(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        variables: Optional[List[str]] = None,
        region: Optional[str] = None,
    ) -> pd.DataFrame:
        obs_df, _ = self.generator.generate()
        if start_time is not None:
            obs_df = obs_df[obs_df["timestamp"] >= pd.to_datetime(start_time)]
        if end_time is not None:
            obs_df = obs_df[obs_df["timestamp"] <= pd.to_datetime(end_time)]
        if variables is not None:
            obs_df = obs_df[obs_df["variable"].isin(variables)]
        if region is not None and "region" in obs_df.columns:
            obs_df = obs_df[obs_df["region"] == region]
        return self.validate(obs_df)


class SyntheticForecastSource(ForecastSource):
    """
    Forecast source adapter backed by the deterministic synthetic generator.
    """

    def __init__(
        self,
        source_name: str = "nwp_model_a",
        model_type: str = "nwp",
        config: Optional[AppConfig] = None,
    ):
        super().__init__(source_name=source_name, model_type=model_type)
        self.app_config = config or get_config()
        self.generator = SyntheticDataGenerator(self.app_config)

    def fetch(
        self,
        run_times: Optional[List[datetime]] = None,
        lead_times: Optional[List[int]] = None,
        variables: Optional[List[str]] = None,
        region: Optional[str] = None,
    ) -> pd.DataFrame:
        _, forecasts_df = self.generator.generate()
        forecasts_df = forecasts_df[forecasts_df["model_name"] == self.source_name]
        if lead_times is not None:
            forecasts_df = forecasts_df[forecasts_df["lead_time"].isin(lead_times)]
        if variables is not None:
            forecasts_df = forecasts_df[forecasts_df["variable"].isin(variables)]
        if region is not None and "region" in forecasts_df.columns:
            forecasts_df = forecasts_df[forecasts_df["region"] == region]
        return self.validate(forecasts_df)
