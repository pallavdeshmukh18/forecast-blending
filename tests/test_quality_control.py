"""
Unit tests for data quality control, physical bounds, coordinate checks, and diagnostic reports.
"""

import numpy as np
import pandas as pd
from ingestion.quality_control import apply_quality_control, format_quality_report_text


def test_qc_negative_rainfall_rejection():
    df = pd.DataFrame({
        "latitude": [20.0, 20.0],
        "longitude": [75.0, 75.0],
        "variable": ["rainfall", "rainfall"],
        "forecast_value": [-5.0, 15.0],  # -5.0 is physically impossible
    })
    valid_df, rej_df, report = apply_quality_control(df, value_col="forecast_value")
    assert len(valid_df) == 1
    assert valid_df["forecast_value"].iloc[0] == 15.0
    assert len(rej_df) == 1
    assert rej_df["qc_reason"].iloc[0] == "out_of_range_rainfall"
    assert report.out_of_range_records == 1


def test_qc_negative_wind_speed_rejection():
    df = pd.DataFrame({
        "latitude": [20.0, 20.0],
        "longitude": [75.0, 75.0],
        "variable": ["wind_speed", "wind_speed"],
        "forecast_value": [-2.5, 8.0],
    })
    valid_df, rej_df, report = apply_quality_control(df, value_col="forecast_value")
    assert len(valid_df) == 1
    assert valid_df["forecast_value"].iloc[0] == 8.0
    assert len(rej_df) == 1
    assert rej_df["qc_reason"].iloc[0] == "out_of_range_wind_speed"


def test_qc_invalid_temperature_bounds():
    df = pd.DataFrame({
        "latitude": [20.0, 20.0],
        "longitude": [75.0, 75.0],
        "variable": ["temperature", "temperature"],
        "forecast_value": [85.0, 25.0],  # 85 C exceeds maximum physical bound (65 C)
    })
    valid_df, rej_df, report = apply_quality_control(df, value_col="forecast_value")
    assert len(valid_df) == 1
    assert valid_df["forecast_value"].iloc[0] == 25.0
    assert len(rej_df) == 1
    assert rej_df["qc_reason"].iloc[0] == "out_of_range_temperature"


def test_qc_invalid_coordinates():
    df = pd.DataFrame({
        "latitude": [95.0, 20.0],   # 95 > 90
        "longitude": [75.0, 200.0], # 200 > 180
        "variable": ["temperature", "temperature"],
        "forecast_value": [25.0, 25.0],
    })
    valid_df, rej_df, report = apply_quality_control(df, value_col="forecast_value")
    assert len(valid_df) == 0
    assert len(rej_df) == 2
    assert (rej_df["qc_reason"] == "invalid_coords").all()
    assert report.invalid_coord_records == 2


def test_qc_nan_and_inf_handling():
    df = pd.DataFrame({
        "latitude": [20.0, 20.0, 20.0],
        "longitude": [75.0, 75.0, 75.0],
        "variable": ["temperature", "temperature", "temperature"],
        "forecast_value": [np.nan, np.inf, 22.0],
    })
    valid_df, rej_df, report = apply_quality_control(df, value_col="forecast_value")
    assert len(valid_df) == 1
    assert valid_df["forecast_value"].iloc[0] == 22.0
    assert len(rej_df) == 2
    assert report.missing_records == 2


def test_format_quality_report_text():
    df = pd.DataFrame({
        "latitude": [20.0],
        "longitude": [75.0],
        "variable": ["temperature"],
        "forecast_value": [22.0],
        "model_name": ["nwp_model_a"],
        "lead_time": [24],
    })
    _, _, report = apply_quality_control(df, value_col="forecast_value")
    text = format_quality_report_text(report)
    assert "DATA QUALITY REPORT" in text
    assert "Total Records:" in text
    assert "Valid Records:" in text
