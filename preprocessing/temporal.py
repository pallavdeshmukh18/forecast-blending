"""
Temporal alignment and verification timestamp validation routines.
"""

from typing import Optional, Tuple
import numpy as np
import pandas as pd

from configs import AppConfig, get_config


def validate_and_compute_valid_time(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure forecast valid timestamp conforms to: valid_time = run_time + lead_time.
    If 'timestamp' is missing or null, it is calculated from 'run_time' + 'lead_time'.
    If both 'timestamp' and ('run_time', 'lead_time') are present, validates their consistency.

    Parameters
    ----------
    df : pd.DataFrame
        Forecast DataFrame.

    Returns
    -------
    pd.DataFrame
        DataFrame with verified UTC 'timestamp' (valid time).
    """
    out = df.copy()

    has_run = "run_time" in out.columns and out["run_time"].notna().any()
    has_lead = "lead_time" in out.columns and out["lead_time"].notna().any()
    has_ts = "timestamp" in out.columns and out["timestamp"].notna().any()

    if has_run and has_lead:
        out["run_time"] = pd.to_datetime(out["run_time"], utc=True)
        computed_valid = out["run_time"] + pd.to_timedelta(out["lead_time"], unit="h")

        if not has_ts:
            out["timestamp"] = computed_valid
        else:
            out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
            # Check discrepancy (> 1 minute difference between timestamp and run_time + lead_time)
            diff_secs = (out["timestamp"] - computed_valid).abs().dt.total_seconds()
            mismatch_mask = diff_secs > 60
            if mismatch_mask.any():
                # Correct timestamp to canonical run_time + lead_time
                out.loc[mismatch_mask, "timestamp"] = computed_valid.loc[mismatch_mask]

    elif has_ts:
        out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)

    return out


def match_temporal_tolerance(
    forecast_df: pd.DataFrame,
    obs_df: pd.DataFrame,
    tolerance_minutes: int = 30,
) -> pd.DataFrame:
    """
    Pair forecast records with observations within a temporal tolerance window.

    Parameters
    ----------
    forecast_df : pd.DataFrame
        DataFrame with 'timestamp'.
    obs_df : pd.DataFrame
        DataFrame with 'timestamp'.
    tolerance_minutes : int, default 30
        Maximum allowable difference in minutes between forecast valid time and observation timestamp.

    Returns
    -------
    pd.DataFrame
        Matched DataFrame with 'time_difference_minutes' and 'is_temporal_matched'.
    """
    f_df = forecast_df.copy()
    o_df = obs_df.copy()

    f_df["timestamp"] = pd.to_datetime(f_df["timestamp"], utc=True)
    o_df["timestamp"] = pd.to_datetime(o_df["timestamp"], utc=True)

    # Use pd.merge_asof for nearest temporal matching
    f_df = f_df.sort_values("timestamp")
    o_df = o_df.sort_values("timestamp")

    tolerance = pd.Timedelta(minutes=tolerance_minutes)
    matched = pd.merge_asof(
        f_df,
        o_df,
        on="timestamp",
        direction="nearest",
        tolerance=tolerance,
        suffixes=("_forecast", "_obs"),
    )

    return matched
