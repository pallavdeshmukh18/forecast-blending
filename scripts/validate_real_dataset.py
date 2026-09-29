"""
Command-line utility to validate, inspect, and generate Data Quality Reports for real weather datasets.

Usage:
    python scripts/validate_real_dataset.py --input path/to/file.csv [--type forecast|observation]
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
from ingestion.schema import validate_forecast_dataframe, validate_observation_dataframe


def main():
    parser = argparse.ArgumentParser(description="Validate and assess quality of meteorological datasets.")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to input dataset file (CSV, Parquet, NetCDF, GRIB2)")
    parser.add_argument("--type", "-t", type=str, default="auto", choices=["auto", "forecast", "observation"], help="Dataset type")
    parser.add_argument("--format", "-f", type=str, default=None, help="Explicit file format (csv, parquet, netcdf, grib2)")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: File not found at '{input_path}'", file=sys.stderr)
        sys.exit(1)

    cfg = get_config()
    print("=" * 70)
    print(f" Validating Meteorological Dataset: {input_path.name}")
    print("=" * 70)

    # 1. Load File
    try:
        df = load_weather_file(input_path, file_format=args.format)
        print(f"Successfully read {len(df):,} records from {input_path}")
    except Exception as e:
        print(f"Error reading file: {e}", file=sys.stderr)
        sys.exit(1)

    # Auto-detect type if not specified
    ds_type = args.type
    if ds_type == "auto":
        if "lead_time" in df.columns or "lead" in df.columns or "model_name" in df.columns or "forecast_value" in df.columns:
            ds_type = "forecast"
        elif "station_id" in df.columns or "observed_value" in df.columns:
            ds_type = "observation"
        else:
            ds_type = "forecast" if "forecast_value" in df.columns else "observation"
        print(f"Auto-detected dataset type: {ds_type.upper()}")

    # 2. Standardize column names
    rename_map = {
        "valid_time": "timestamp",
        "time": "timestamp",
        "forecast_time": "timestamp",
        "lat": "latitude",
        "lon": "longitude",
        "lead": "lead_time",
        "leadtime": "lead_time",
        "val": "forecast_value" if ds_type == "forecast" else "observed_value",
        "value": "forecast_value" if ds_type == "forecast" else "observed_value",
        "var": "variable",
        "param": "variable",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns and v not in df.columns})

    # 3. Standardize variables
    df = standardize_dataframe_variables(df, config=cfg)

    # 4. Quality Control
    val_col = "forecast_value" if ds_type == "forecast" else "observed_value"
    if val_col not in df.columns:
        print(f"Warning: Expected numerical value column '{val_col}' not found in dataset.", file=sys.stderr)

    valid_df, rejected_df, report = apply_quality_control(df, value_col=val_col, config=cfg)

    # Print Report
    print("\n" + format_quality_report_text(report))

    if report.issues_summary:
        print("\nQuality Issues Identified:")
        for issue in report.issues_summary:
            print(f"  [WARN] {issue}")

    # 5. Schema Validation Check
    print("\n" + "-" * 70)
    print(" Schema Validation Status:")
    try:
        if ds_type == "forecast":
            validate_forecast_dataframe(valid_df, allow_nans=False)
        else:
            validate_observation_dataframe(valid_df)
        print("  [PASS] All valid records strictly conform to the project's standardized schema.")
    except Exception as e:
        print(f"  [FAIL] Schema validation error on valid records: {e}")

    print("=" * 70)


if __name__ == "__main__":
    main()
