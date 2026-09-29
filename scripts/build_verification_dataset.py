"""
Command-line utility to ingest real forecasts and observations, execute spatiotemporal matching,
compute error metrics, and persist the standardized verification dataset.

Usage:
    python scripts/build_verification_dataset.py \
        --forecast path/to/forecast.csv \
        --observations path/to/observations.csv \
        --output data/processed/verification_dataset.parquet
"""

import argparse
import sys
from pathlib import Path

# Add repo root to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
from configs import get_config
from ingestion.readers import load_weather_file
from ingestion.variable_mapping import standardize_dataframe_variables
from ingestion.units import normalize_dataframe_units
from ingestion.quality_control import apply_quality_control, format_quality_report_text
from preprocessing.matching import match_forecasts_to_observations
from evaluation.error_database import HistoricalErrorDatabase
from evaluation.comparison import compare_models


def main():
    parser = argparse.ArgumentParser(description="Build standardized verification dataset from real forecasts & observations.")
    parser.add_argument("--forecast", "-f", type=str, required=True, help="Path to forecast dataset file (CSV, Parquet, NetCDF, GRIB2)")
    parser.add_argument("--observations", "-o", type=str, required=True, help="Path to observation dataset file (CSV, Parquet, NetCDF)")
    parser.add_argument("--output", type=str, default="data/processed/verification_dataset.parquet", help="Path to output verification dataset")
    parser.add_argument("--format", type=str, default=None, choices=["parquet", "csv"], help="Output file format (inferred from output if None)")
    parser.add_argument("--max-distance", type=float, default=50.0, help="Maximum allowed spatial matching distance in km")
    parser.add_argument("--tolerance", type=int, default=30, help="Maximum temporal tolerance window in minutes")
    parser.add_argument("--save-error-db", action="store_true", help="Also update the persistent HistoricalErrorDatabase")
    args = parser.parse_args()

    cfg = get_config()
    print("=" * 75)
    print(" Building Meteorological Forecast Verification Dataset")
    print("=" * 75)

    # 1. Ingest Raw Forecasts
    print(f"\n[1/5] Ingesting Forecasts: {args.forecast}")
    f_raw = load_weather_file(args.forecast)
    f_raw = standardize_dataframe_variables(f_raw, config=cfg)
    f_valid, f_rej, f_report = apply_quality_control(f_raw, value_col="forecast_value", config=cfg)
    print(f"      Loaded {len(f_raw):,} forecast records -> {len(f_valid):,} valid after QC.")

    # 2. Ingest Raw Observations
    print(f"\n[2/5] Ingesting Observations: {args.observations}")
    o_raw = load_weather_file(args.observations)
    o_raw = standardize_dataframe_variables(o_raw, config=cfg)
    o_valid, o_rej, o_report = apply_quality_control(o_raw, value_col="observed_value", config=cfg)
    print(f"      Loaded {len(o_raw):,} observation records -> {len(o_valid):,} valid after QC.")

    # 3. Spatiotemporal Matching
    print(f"\n[3/5] Executing Spatiotemporal Alignment & Error Computation...")
    print(f"      Spatial Max Distance: {args.max_distance} km | Temporal Tolerance: {args.tolerance} mins")
    verified_df = match_forecasts_to_observations(
        forecasts_df=f_valid,
        observations_df=o_valid,
        spatial_config={"max_distance_km": args.max_distance},
        temporal_config={"tolerance_minutes": args.tolerance},
        config=cfg,
    )

    print(f"      Produced {len(verified_df):,} verified forecast-observation pairs.")
    if verified_df.empty:
        print("      [WARN] No valid forecast-observation pairs could be matched within the spatiotemporal thresholds.")
        sys.exit(0)

    # 4. Save Output Verification Dataset
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fmt = args.format or out_path.suffix.lstrip(".") or "parquet"

    print(f"\n[4/5] Saving Verification Dataset to: {out_path} (format={fmt})")
    if fmt == "parquet":
        verified_df.to_parquet(out_path, index=False)
    else:
        verified_df.to_csv(out_path, index=False)

    if args.save_error_db:
        db_path = Path(cfg.paths.error_db_file)
        print(f"      Updating Error Database at: {db_path}")
        error_db = HistoricalErrorDatabase(config=cfg)
        if db_path.exists():
            error_db = HistoricalErrorDatabase.load(db_path, config=cfg)
        error_db.add_records(verified_df)
        error_db.save(db_path)

    # 5. Summary Verification Metrics
    print(f"\n[5/5] Multi-Model Skill Benchmark Summary on Verified Real Data:")
    print("=" * 75)
    summary_df = compare_models(verified_df, group_by=["variable", "lead_time"])
    if not summary_df.empty:
        cols = ["variable", "lead_time", "model_name", "mae", "rmse", "bias", "correlation", "sample_count"]
        print(summary_df[[c for c in cols if c in summary_df.columns]].to_string(index=False))

    print("\n" + "=" * 75)
    print(" Verification dataset build complete.")
    print("=" * 75)


if __name__ == "__main__":
    main()
