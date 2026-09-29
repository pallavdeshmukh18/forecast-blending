"""
Meteorological Data Quality Control (QC) and Diagnostics Reporting.
"""

from typing import Dict, List, Optional, Tuple, Union
from pydantic import BaseModel, Field
import numpy as np
import pandas as pd

from configs import AppConfig, get_config


import json
from pathlib import Path

class DataQualityReport(BaseModel):
    """
    Summary metrics and diagnostic counts from data quality assessment.
    """
    total_records: int = 0
    valid_records: int = 0
    missing_records: int = 0
    out_of_range_records: int = 0
    invalid_coord_records: int = 0
    duplicate_records: int = 0
    spatial_mismatch_records: int = 0
    temporal_mismatch_records: int = 0
    temporal_gaps_detected: int = 0
    spatial_gaps_detected: int = 0
    records_by_model: Dict[str, int] = Field(default_factory=dict)
    records_by_variable: Dict[str, int] = Field(default_factory=dict)
    records_by_lead_time: Dict[str, int] = Field(default_factory=dict)
    issues_summary: List[str] = Field(default_factory=list)
    rejection_reasons: Dict[str, int] = Field(default_factory=dict)


def apply_quality_control(
    df: pd.DataFrame,
    value_col: str = "forecast_value",
    variable_col: str = "variable",
    config: Optional[AppConfig] = None,
    save_report_path: Optional[str | Path] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, DataQualityReport]:
    """
    Execute comprehensive meteorological quality control checks on a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame (forecasts or observations).
    value_col : str
        Column name containing numerical weather values.
    variable_col : str
        Column name indicating meteorological variable ('temperature', 'rainfall', 'wind_speed').
    config : Optional[AppConfig]
        Application configuration containing physical bound rules.
    save_report_path : Optional[str | Path]
        Optional path to persist JSON quality report (e.g. 'reports/data_quality.json').

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, DataQualityReport]
        - valid_df: DataFrame containing only quality-verified records (quality_flag = 1)
        - rejected_df: DataFrame containing flagged/rejected records with 'qc_reason' column
        - report: Structured DataQualityReport instance
    """
    cfg = config or get_config()
    total_records = len(df)
    if total_records == 0:
        empty_report = DataQualityReport()
        if save_report_path:
            save_quality_report(empty_report, save_report_path)
        return df.copy(), pd.DataFrame(), empty_report

    qc_df = df.copy()
    qc_df["quality_flag"] = 1
    qc_df["qc_reason"] = "valid"

    # 1. Check coordinates
    if "latitude" in qc_df.columns:
        invalid_lat = (qc_df["latitude"] < -90.0) | (qc_df["latitude"] > 90.0) | qc_df["latitude"].isna()
    else:
        invalid_lat = pd.Series(False, index=qc_df.index)

    if "longitude" in qc_df.columns:
        invalid_lon = (qc_df["longitude"] < -180.0) | (qc_df["longitude"] > 180.0) | qc_df["longitude"].isna()
    else:
        invalid_lon = pd.Series(False, index=qc_df.index)

    coord_mask = invalid_lat | invalid_lon
    qc_df.loc[coord_mask, "quality_flag"] = 0
    qc_df.loc[coord_mask, "qc_reason"] = "invalid_coords"

    # 2. Check duplicates (same timestamp, coordinates, model, lead_time, variable)
    dup_cols = [c for c in ["timestamp", "latitude", "longitude", "model_name", "lead_time", variable_col] if c in qc_df.columns]
    dup_mask = pd.Series(False, index=qc_df.index)
    if "timestamp" in qc_df.columns and len(dup_cols) >= 3:
        dup_mask = qc_df.duplicated(subset=dup_cols, keep="first")
        qc_df.loc[dup_mask & (qc_df["quality_flag"] == 1), "quality_flag"] = 0
        qc_df.loc[dup_mask & (qc_df["qc_reason"] == "valid"), "qc_reason"] = "duplicate_record"

    # 3. Check missing / infinite values (Do not convert NaNs to zero!)
    missing_val_mask = qc_df[value_col].isna() | np.isinf(qc_df[value_col])
    qc_df.loc[missing_val_mask & (qc_df["quality_flag"] == 1), "quality_flag"] = 0
    qc_df.loc[missing_val_mask & (qc_df["qc_reason"] == "valid"), "qc_reason"] = "missing_value"

    # 4. Check physical bounds per variable
    if variable_col in qc_df.columns:
        for var_name, var_cfg in cfg.variables.items():
            var_mask = (qc_df[variable_col] == var_name) & (qc_df["quality_flag"] == 1)
            if not var_mask.any():
                continue

            vals = qc_df.loc[var_mask, value_col]
            out_of_bounds = (vals < var_cfg.min_physical_bound) | (vals > var_cfg.max_physical_bound)

            bad_indices = qc_df.loc[var_mask][out_of_bounds].index
            qc_df.loc[bad_indices, "quality_flag"] = 0
            qc_df.loc[bad_indices, "qc_reason"] = f"out_of_range_{var_name}"

    # 5. Temporal gap detection
    temporal_gaps = 0
    if "timestamp" in qc_df.columns and len(qc_df) > 1:
        try:
            ts = pd.to_datetime(qc_df["timestamp"]).drop_duplicates().sort_values()
            if len(ts) > 2:
                diffs = ts.diff().dropna()
                median_step = diffs.median()
                temporal_gaps = int((diffs > 2.0 * median_step).sum())
        except Exception:
            pass

    # Split into valid and rejected
    valid_df = qc_df[qc_df["quality_flag"] == 1].reset_index(drop=True)
    rejected_df = qc_df[qc_df["quality_flag"] == 0].reset_index(drop=True)

    # Build report
    by_var = {}
    if variable_col in valid_df.columns:
        by_var = valid_df[variable_col].value_counts().to_dict()

    by_model = {}
    if "model_name" in valid_df.columns:
        by_model = valid_df["model_name"].value_counts().to_dict()

    by_lt = {}
    if "lead_time" in valid_df.columns:
        by_lt = {str(k): int(v) for k, v in valid_df["lead_time"].value_counts().items()}

    rejection_reasons = {}
    if not rejected_df.empty and "qc_reason" in rejected_df.columns:
        rejection_reasons = {str(k): int(v) for k, v in rejected_df["qc_reason"].value_counts().items()}

    invalid_coords_count = int(coord_mask.sum())
    dup_count = int(dup_mask.sum())
    missing_count = int(missing_val_mask.sum())
    out_of_range_count = int((qc_df["qc_reason"].str.startswith("out_of_range")).sum())

    issues = []
    if invalid_coords_count > 0:
        issues.append(f"{invalid_coords_count} records had invalid or out-of-bounds geographic coordinates.")
    if dup_count > 0:
        issues.append(f"{dup_count} duplicate timestamp/spatial records were flagged.")
    if missing_count > 0:
        issues.append(f"{missing_count} records had missing, NaN, or infinite values.")
    if out_of_range_count > 0:
        issues.append(f"{out_of_range_count} records violated physical meteorological bounds (e.g. negative rainfall/wind).")
    if temporal_gaps > 0:
        issues.append(f"{temporal_gaps} temporal sequence gaps were detected in time series.")

    report = DataQualityReport(
        total_records=total_records,
        valid_records=len(valid_df),
        missing_records=missing_count,
        out_of_range_records=out_of_range_count,
        invalid_coord_records=invalid_coords_count,
        duplicate_records=dup_count,
        spatial_mismatch_records=0,
        temporal_mismatch_records=0,
        temporal_gaps_detected=temporal_gaps,
        spatial_gaps_detected=0,
        records_by_model=by_model,
        records_by_variable=by_var,
        records_by_lead_time=by_lt,
        issues_summary=issues,
        rejection_reasons=rejection_reasons,
    )

    if save_report_path:
        save_quality_report(report, save_report_path)

    return valid_df, rejected_df, report


def save_quality_report(report: DataQualityReport, output_path: str | Path = "reports/data_quality.json") -> None:
    """
    Save the machine-readable DataQualityReport to a JSON file.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report.model_dump(), f, indent=2)


def format_quality_report_text(report: DataQualityReport) -> str:
    """
    Format DataQualityReport into a clean, human-readable terminal/log text block.
    """
    lines = [
        "DATA QUALITY REPORT",
        "-" * 38,
        f"Total Records:         {report.total_records:>12,}",
        f"Valid Records:         {report.valid_records:>12,}",
        f"Missing Values:        {report.missing_records:>12,}",
        f"Out of Physical Range: {report.out_of_range_records:>12,}",
        f"Invalid Coordinates:   {report.invalid_coord_records:>12,}",
        f"Spatial Mismatches:    {report.spatial_mismatch_records:>12,}",
        f"Temporal Mismatches:   {report.temporal_mismatch_records:>12,}",
        "",
        "Variables (Valid Records):",
    ]
    for var, count in report.records_by_variable.items():
        lines.append(f"  {var:<20} {count:>12,}")

    if report.records_by_model:
        lines.append("\nModels (Valid Records):")
        for m, count in report.records_by_model.items():
            lines.append(f"  {m:<20} {count:>12,}")

    if report.records_by_lead_time:
        lines.append("\nLead Times:")
        for lt, count in sorted(report.records_by_lead_time.items(), key=lambda x: int(x[0]) if x[0].isdigit() else 0):
            lines.append(f"  +{lt}h:{' '*(17-len(str(lt)))} {count:>12,}")

    return "\n".join(lines)
