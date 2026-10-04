"""
WeatherBench 2 Real Meteorological Data Ingestion Script.

Downloads/streams a configured, reproducible subset of WeatherBench 2 datasets from
Google Cloud Storage (anonymous public access):
- ERA5 Atmospheric Reanalysis (Ground-Truth Observation Reference)
- ECMWF HRES (Deterministic High-Resolution NWP)
- GraphCast v2 (Google DeepMind Neural Weather Prediction)
- Pangu-Weather (Huawei Cloud Neural Weather Prediction)

Standardizes coordinates, variable names, and physical units, performs quality control (QC),
matches forecast-observation verification pairs, generates machine-readable manifests & provenance,
and persists independent real-data Parquet files.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr
import gcsfs

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "ML"))
sys.path.insert(0, str(PROJECT_ROOT / "Backend"))
sys.path.insert(0, str(PROJECT_ROOT))

from configs import AppConfig, RegionConfig, get_config
from ingestion.schema import (
    ForecastRecord,
    ObservationRecord,
    validate_forecast_dataframe,
    validate_observation_dataframe,
)
from ingestion.variable_mapping import standardize_variable_name
from ingestion.units import convert_units, normalize_dataframe_units
from ingestion.quality_control import apply_quality_control, DataQualityReport
from preprocessing.matching import match_forecasts_to_observations


def compute_sha256(file_path: Path) -> str:
    """Compute SHA-256 cryptographic hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(65536), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


class WeatherBench2Ingestor:
    """
    Ingestor and processor for WeatherBench 2 Zarr datasets from Google Cloud Storage.
    """

    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config or get_config()
        self.wb2_cfg = self.config.weatherbench2
        self.fs = gcsfs.GCSFileSystem(token="anon")

    def _open_zarr_store(self, gcs_path: str) -> xr.Dataset:
        """Open consolidated Zarr store from GCS anonymously."""
        clean_path = gcs_path.replace("gs://", "")
        store = gcsfs.GCSMap(clean_path, gcs=self.fs, check=False)
        return xr.open_zarr(store, consolidated=True)

    def ingest_observations(
        self,
        start_date: str,
        end_date: str,
        region: RegionConfig,
        variables: List[str],
    ) -> pd.DataFrame:
        """
        Ingest ERA5 ground-truth reference data from WeatherBench 2.
        """
        print(f"Connecting to ERA5 store: {self.wb2_cfg.era5_source}...")
        ds = self._open_zarr_store(self.wb2_cfg.era5_source)

        # Coordinate slicing
        lat_vals = ds.latitude.values
        lat_slice = (
            slice(region.lat_min, region.lat_max)
            if lat_vals[0] < lat_vals[-1]
            else slice(region.lat_max, region.lat_min)
        )
        lon_slice = slice(region.lon_min, region.lon_max)
        time_slice = slice(f"{start_date}T00:00:00", f"{end_date}T23:59:59")

        # Determine target data variables
        var_map = {
            "temperature": "2m_temperature",
            "wind_speed": "10m_wind_speed",
            "rainfall": "total_precipitation_6hr",
        }
        wb_vars = [var_map[v] for v in variables if v in var_map and var_map[v] in ds.data_vars]

        print(f"Slicing ERA5: Lat[{region.lat_min}, {region.lat_max}], Lon[{region.lon_min}, {region.lon_max}], Time[{start_date}, {end_date}], Vars={wb_vars}...")
        sub_ds = ds[wb_vars].sel(latitude=lat_slice, longitude=lon_slice, time=time_slice).load()

        df_raw = sub_ds.to_dataframe().reset_index()
        df_raw["timestamp"] = pd.to_datetime(df_raw["time"], utc=True)
        df_raw["longitude"] = ((df_raw["longitude"] + 180.0) % 360.0) - 180.0
        df_raw["latitude"] = df_raw["latitude"].astype(float)
        df_raw["longitude"] = df_raw["longitude"].astype(float)

        if "2m_temperature" in df_raw.columns:
            df_raw["temperature"] = df_raw["2m_temperature"] - 273.15
        if "10m_wind_speed" in df_raw.columns:
            df_raw["wind_speed"] = df_raw["10m_wind_speed"]
        if "total_precipitation_6hr" in df_raw.columns:
            df_raw["rainfall"] = np.maximum(0.0, df_raw["total_precipitation_6hr"] * 1000.0)

        value_vars = [v for v in variables if v in df_raw.columns]
        obs_long = pd.melt(
            df_raw,
            id_vars=["timestamp", "latitude", "longitude"],
            value_vars=value_vars,
            var_name="variable",
            value_name="observed_value",
        )
        obs_long["observed_value"] = obs_long["observed_value"].round(4)
        obs_long["station_id"] = "era5_grid_" + obs_long["latitude"].round(1).astype(str) + "_" + obs_long["longitude"].round(1).astype(str)
        obs_long["region"] = "south_asia"
        obs_long["elevation"] = 0.0
        obs_long["quality_flag"] = 1
        obs_long = obs_long.dropna(subset=["observed_value"]).reset_index(drop=True)

        if obs_long.empty:
            raise ValueError(f"No ERA5 observation records extracted for range {start_date} to {end_date}.")

        print(f"Extracted {len(obs_long):,} raw ERA5 observation records.")
        return obs_long

    def ingest_model_forecasts(
        self,
        model_name: str,
        start_date: str,
        end_date: str,
        region: RegionConfig,
        variables: List[str],
        lead_times: List[int],
    ) -> pd.DataFrame:
        """
        Ingest forecasts for a specific model (e.g. 'ecmwf_hres', 'graphcast', 'pangu').
        """
        source_map = {
            "ecmwf_hres": self.wb2_cfg.hres_source,
            "hres": self.wb2_cfg.hres_source,
            "graphcast": self.wb2_cfg.graphcast_source,
            "graphcast_v2": self.wb2_cfg.graphcast_source,
            "pangu": self.wb2_cfg.pangu_source,
        }

        gcs_source = source_map.get(model_name.lower())
        if not gcs_source:
            raise ValueError(f"Unknown WeatherBench 2 model: {model_name}. Supported: {list(source_map.keys())}")

        print(f"Connecting to {model_name} store: {gcs_source}...")
        ds = self._open_zarr_store(gcs_source)

        lat_vals = ds.latitude.values
        lat_slice = (
            slice(region.lat_min, region.lat_max)
            if lat_vals[0] < lat_vals[-1]
            else slice(region.lat_max, region.lat_min)
        )
        lon_slice = slice(region.lon_min, region.lon_max)
        time_slice = slice(f"{start_date}T00:00:00", f"{end_date}T23:59:59")

        # Map canonical variables to WeatherBench variable names
        var_map = {
            "temperature": "2m_temperature",
            "wind_speed": "10m_wind_speed",
            "rainfall": "total_precipitation_6hr",
        }

        # Available lead times in store
        avail_leads = ds.prediction_timedelta.values
        target_leads = [int(lt) for lt in lead_times if lt in avail_leads]
        if not target_leads:
            target_leads = [int(lt) for lt in avail_leads[:4]]

        target_vars = []
        has_uv_wind = False
        for v in variables:
            wb_v = var_map.get(v)
            if wb_v and wb_v in ds.data_vars:
                target_vars.append(wb_v)
            elif v == "wind_speed" and "10m_u_component_of_wind" in ds.data_vars and "10m_v_component_of_wind" in ds.data_vars:
                target_vars.extend(["10m_u_component_of_wind", "10m_v_component_of_wind"])
                has_uv_wind = True

        target_vars = list(set(target_vars))
        print(f"Slicing {model_name}: Leads={target_leads}h, Vars={target_vars}...")

        sub_ds = ds[target_vars].sel(
            latitude=lat_slice,
            longitude=lon_slice,
            time=time_slice,
            prediction_timedelta=target_leads,
        ).load()

        df_raw = sub_ds.to_dataframe().reset_index()
        df_raw["run_time"] = pd.to_datetime(df_raw["time"], utc=True)
        df_raw["lead_time"] = df_raw["prediction_timedelta"].astype(int)
        df_raw["timestamp"] = df_raw["run_time"] + pd.to_timedelta(df_raw["lead_time"], unit="h")
        df_raw["longitude"] = ((df_raw["longitude"] + 180.0) % 360.0) - 180.0
        df_raw["latitude"] = df_raw["latitude"].astype(float)
        df_raw["longitude"] = df_raw["longitude"].astype(float)

        if "2m_temperature" in df_raw.columns:
            df_raw["temperature"] = df_raw["2m_temperature"] - 273.15
        if "10m_wind_speed" in df_raw.columns:
            df_raw["wind_speed"] = df_raw["10m_wind_speed"]
        elif has_uv_wind and "10m_u_component_of_wind" in df_raw.columns:
            df_raw["wind_speed"] = np.sqrt(df_raw["10m_u_component_of_wind"]**2 + df_raw["10m_v_component_of_wind"]**2)
        if "total_precipitation_6hr" in df_raw.columns:
            df_raw["rainfall"] = np.maximum(0.0, df_raw["total_precipitation_6hr"] * 1000.0)

        value_vars = [v for v in variables if v in df_raw.columns]
        f_long = pd.melt(
            df_raw,
            id_vars=["timestamp", "run_time", "lead_time", "latitude", "longitude"],
            value_vars=value_vars,
            var_name="variable",
            value_name="forecast_value",
        )
        f_long["forecast_value"] = f_long["forecast_value"].round(4)
        f_long["model_name"] = model_name
        f_long["region"] = "south_asia"
        f_long = f_long.dropna(subset=["forecast_value"]).reset_index(drop=True)

        print(f"Extracted {len(f_long):,} raw forecast records for {model_name}.")
        return f_long

    def run_ingestion(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        models: Optional[List[str]] = None,
        variables: Optional[List[str]] = None,
        lead_times: Optional[List[int]] = None,
        save_datasets: bool = True,
    ) -> Dict[str, Any]:
        """
        Execute full WeatherBench 2 ingestion pipeline.
        """
        t0 = time.time()
        s_date = start_date or self.wb2_cfg.start_date
        e_date = end_date or self.wb2_cfg.end_date
        target_models = models or self.wb2_cfg.forecast_models
        target_vars = variables or self.wb2_cfg.variables
        target_leads = lead_times or self.wb2_cfg.lead_times_hours
        region = self.wb2_cfg.region

        print("=" * 60)
        print("WEATHERBENCH 2 REAL-DATA INGESTION PIPELINE")
        print("=" * 60)
        print(f"Start Date:     {s_date}")
        print(f"End Date:       {e_date}")
        print(f"Models:         {target_models}")
        print(f"Variables:      {target_vars}")
        print(f"Lead Times (h): {lead_times}")
        print(f"Region:         Lat [{region.lat_min}, {region.lat_max}], Lon [{region.lon_min}, {region.lon_max}] ({region.description})")
        print(f"Resolution:     {self.wb2_cfg.resolution}")
        print("-" * 60)

        # 1. Ingest ERA5 Observations
        raw_obs_df = self.ingest_observations(
            start_date=s_date,
            end_date=e_date,
            region=region,
            variables=target_vars,
        )

        # 2. Ingest Forecasts for all target models
        forecast_dfs = []
        for m in target_models:
            m_df = self.ingest_model_forecasts(
                model_name=m,
                start_date=s_date,
                end_date=e_date,
                region=region,
                variables=target_vars,
                lead_times=target_leads,
            )
            forecast_dfs.append(m_df)

        raw_forecasts_df = pd.concat(forecast_dfs, ignore_index=True)

        # 3. Apply Quality Control
        print("\nApplying Quality Control Checks...")
        valid_f_df, rej_f_df, qc_f_report = apply_quality_control(
            raw_forecasts_df,
            value_col="forecast_value",
            variable_col="variable",
            config=self.config,
        )
        valid_o_df, rej_o_df, qc_o_report = apply_quality_control(
            raw_obs_df,
            value_col="observed_value",
            variable_col="variable",
            config=self.config,
        )

        print(f"Forecast QC:    {qc_f_report.valid_records:,} valid / {len(rej_f_df):,} flagged")
        print(f"Observation QC: {qc_o_report.valid_records:,} valid / {len(rej_o_df):,} flagged")

        # 4. Validate standard schemas
        clean_f_df = validate_forecast_dataframe(valid_f_df)
        clean_o_df = validate_observation_dataframe(valid_o_df)

        # 5. Temporal & Spatial Verification Pairing
        print("\nMatching Forecast-Observation Verification Pairs...")
        verified_df = match_forecasts_to_observations(
            forecasts_df=clean_f_df,
            observations_df=clean_o_df,
            config=self.config,
        )
        print(f"Matched {len(verified_df):,} verified forecast-observation pairs.")

        # Summary statistics
        raw_f_count = len(raw_forecasts_df)
        raw_o_count = len(raw_obs_df)
        matched_count = len(verified_df)
        unmatched_f = raw_f_count - matched_count
        unmatched_o = raw_o_count - (matched_count // max(1, len(target_models)))

        output_paths: Dict[str, Path] = {}
        file_hashes: Dict[str, str] = {}

        if save_datasets:
            raw_dir = PROJECT_ROOT / "ML" / self.wb2_cfg.raw_dir
            proc_dir = PROJECT_ROOT / "ML" / self.wb2_cfg.processed_dir
            prov_dir = PROJECT_ROOT / "ML" / self.wb2_cfg.provenance_dir

            raw_dir.mkdir(parents=True, exist_ok=True)
            proc_dir.mkdir(parents=True, exist_ok=True)
            prov_dir.mkdir(parents=True, exist_ok=True)

            # File locations
            raw_f_path = raw_dir / "weatherbench2_forecasts_raw.parquet"
            raw_o_path = raw_dir / "weatherbench2_observations_raw.parquet"
            proc_f_path = proc_dir / "real_forecasts.parquet"
            proc_o_path = proc_dir / "real_observations.parquet"
            proc_v_path = proc_dir / "real_verification_dataset.parquet"

            # Save Parquet files
            raw_forecasts_df.to_parquet(raw_f_path, index=False)
            raw_obs_df.to_parquet(raw_o_path, index=False)
            clean_f_df.to_parquet(proc_f_path, index=False)
            clean_o_df.to_parquet(proc_o_path, index=False)
            verified_df.to_parquet(proc_v_path, index=False)

            output_paths = {
                "raw_forecasts": raw_f_path,
                "raw_observations": raw_o_path,
                "processed_forecasts": proc_f_path,
                "processed_observations": proc_o_path,
                "verification_dataset": proc_v_path,
            }

            # Calculate SHA-256 hashes
            for name, path in output_paths.items():
                file_hashes[name] = compute_sha256(path)

            # 6. Generate Manifest
            manifest = {
                "dataset": "WeatherBench 2",
                "source": "Google Cloud Storage (gs://weatherbench2/datasets/)",
                "dataset_version": "1.5_deg_equiangular_with_poles_conservative",
                "download_timestamp": datetime.now(timezone.utc).isoformat(),
                "variables": target_vars,
                "forecast_models": target_models,
                "start_date": s_date,
                "end_date": e_date,
                "region": {
                    "lat_min": region.lat_min,
                    "lat_max": region.lat_max,
                    "lon_min": region.lon_min,
                    "lon_max": region.lon_max,
                    "description": region.description,
                },
                "resolution": self.wb2_cfg.resolution,
                "lead_times_hours": lead_times,
                "records_summary": {
                    "raw_forecast_records": raw_f_count,
                    "raw_observation_records": raw_o_count,
                    "valid_forecast_records": len(clean_f_df),
                    "valid_observation_records": len(clean_o_df),
                    "matched_pairs": matched_count,
                    "unmatched_forecasts": unmatched_f,
                },
                "files": {name: str(p.relative_to(PROJECT_ROOT)) for name, p in output_paths.items()},
                "sha256": file_hashes,
                "software_environment": {
                    "python": sys.version.split()[0],
                    "xarray": xr.__version__,
                    "pandas": pd.__version__,
                    "numpy": np.__version__,
                    "gcsfs": gcsfs.__version__,
                },
                "transformations": [
                    {"variable": "temperature", "from": "Kelvin (K)", "to": "Celsius (°C)", "formula": "T_C = T_K - 273.15"},
                    {"variable": "rainfall", "from": "meters (m)", "to": "millimeters (mm)", "formula": "P_mm = P_m * 1000.0"},
                    {"variable": "wind_speed", "from": "m/s or u10/v10", "to": "m/s", "formula": "ws = sqrt(u10^2 + v10^2) if u/v else si10"},
                    {"coordinate": "longitude", "from": "[0, 360)", "to": "[-180, 180]", "formula": "((lon + 180) % 360) - 180"},
                    {"temporal": "valid_time", "formula": "valid_time = run_time + lead_time"},
                ],
            }

            manifest_path = prov_dir / "manifest.json"
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)

            # 7. Generate Provenance
            provenance = {
                "source": "WeatherBench 2 Benchmark (GCP gs://weatherbench2/)",
                "era5_source": self.wb2_cfg.era5_source,
                "hres_source": self.wb2_cfg.hres_source,
                "graphcast_source": self.wb2_cfg.graphcast_source,
                "pangu_source": self.wb2_cfg.pangu_source,
                "retrieval_date": datetime.now(timezone.utc).isoformat(),
                "dataset_version": "1.5_deg_equiangular_with_poles_conservative",
                "variables": target_vars,
                "forecast_models": target_models,
                "temporal_coverage": {"start": s_date, "end": e_date, "time_step_hours": 6},
                "spatial_coverage": {
                    "lat_min": region.lat_min,
                    "lat_max": region.lat_max,
                    "lon_min": region.lon_min,
                    "lon_max": region.lon_max,
                    "grid_step": 1.5,
                },
                "resolution": self.wb2_cfg.resolution,
                "units": {
                    "temperature": "Celsius (°C)",
                    "rainfall": "millimeters (mm)",
                    "wind_speed": "m/s",
                },
                "transformations": [
                    "Kelvin to Celsius temperature conversion",
                    "Meters to mm 6-hour precipitation accumulation scaling",
                    "Wind speed component vector magnitude derivation when applicable",
                    "Longitude normalization to standard [-180, 180] degrees",
                    "Canonical valid verification timestamp calculation (run_time + lead_time)",
                ],
                "interpolation": "Direct native 1.5° equiangular grid alignment (identical spatial mesh across ERA5, HRES, and GraphCast)",
                "filtering": "Physical bound validation and NaN elimination via apply_quality_control",
                "missing_value_handling": "Records containing NaNs/Infs flagged and filtered out; no synthetic imputation applied",
                "software_environment": {
                    "python": sys.version.split()[0],
                    "xarray": xr.__version__,
                    "pandas": pd.__version__,
                    "numpy": np.__version__,
                },
                "qc_summary": {
                    "forecast_valid": qc_f_report.valid_records,
                    "forecast_flagged": len(rej_f_df),
                    "observation_valid": qc_o_report.valid_records,
                    "observation_flagged": len(rej_o_df),
                },
            }

            provenance_path = prov_dir / "provenance.json"
            with open(provenance_path, "w", encoding="utf-8") as f:
                json.dump(provenance, f, indent=2)

            print(f"\nSaved Manifest:   {manifest_path}")
            print(f"Saved Provenance: {provenance_path}")

        elapsed = round(time.time() - t0, 2)
        print("\n" + "=" * 60)
        print(f"INGESTION COMPLETED SUCCESSFULLY in {elapsed}s")
        print("=" * 60)

        return {
            "status": "SUCCESS",
            "start_date": s_date,
            "end_date": e_date,
            "models": target_models,
            "variables": target_vars,
            "raw_forecast_records": raw_f_count,
            "raw_observation_records": raw_o_count,
            "valid_forecast_records": len(clean_f_df),
            "valid_observation_records": len(clean_o_df),
            "matched_pairs": matched_count,
            "unmatched_forecasts": unmatched_f,
            "unmatched_observations": unmatched_o,
            "missing_values": qc_f_report.missing_records + qc_o_report.missing_records,
            "output_paths": {k: str(v) for k, v in output_paths.items()},
            "file_hashes": file_hashes,
            "elapsed_seconds": elapsed,
        }


def main():
    parser = argparse.ArgumentParser(description="Ingest WeatherBench 2 Real Meteorological Dataset")
    parser.add_argument("--start-date", type=str, default=None, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, default=None, help="End date (YYYY-MM-DD)")
    parser.add_argument("--models", nargs="+", default=None, help="Forecast models to ingest")
    parser.add_argument("--variables", nargs="+", default=None, help="Variables to ingest")
    parser.add_argument("--lead-times", nargs="+", type=int, default=None, help="Lead times in hours")
    parser.add_argument("--no-save", action="store_true", help="Do not write Parquet files to disk")

    args = parser.parse_args()

    ingestor = WeatherBench2Ingestor()
    result = ingestor.run_ingestion(
        start_date=args.start_date,
        end_date=args.end_date,
        models=args.models,
        variables=args.variables,
        lead_times=args.lead_times,
        save_datasets=not args.no_save,
    )

    print("\nWeatherBench 2 Smoke Test Summary:")
    print("-----------------------------------")
    print(f"Forecast models:     {', '.join(result['models'])}")
    print(f"Variables:           {', '.join(result['variables'])}")
    print(f"Date range:          {result['start_date']} to {result['end_date']}")
    print(f"Forecast records:    {result['raw_forecast_records']:,}")
    print(f"Observation records: {result['raw_observation_records']:,}")
    print(f"Matched pairs:       {result['matched_pairs']:,}")
    print(f"Missing values:      {result['missing_values']:,}")
    print(f"Unmatched records:   {result['unmatched_forecasts']:,}")


if __name__ == "__main__":
    main()
