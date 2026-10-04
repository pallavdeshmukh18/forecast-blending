"""
Validation and Quality Control for live operational weather data.
"""

from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd


PHYSICAL_BOUNDS = {
    "temperature": (-90.0, 60.0),    # °C
    "wind_speed": (0.0, 100.0),       # m/s
    "rainfall": (0.0, 1000.0),        # mm / 6h
}


class LiveDataValidationError(ValueError):
    """Raised when live data fails critical physical or schema validation."""
    pass


def validate_raw_provider_payload(raw_data: Any) -> Tuple[bool, List[str]]:
    """
    Validate that raw JSON response from live weather provider has required structure.
    
    Returns:
        (is_valid, list_of_error_messages)
    """
    errors = []
    if not isinstance(raw_data, dict):
        return False, ["Payload is not a JSON object"]
        
    if "current" not in raw_data and "hourly" not in raw_data:
        errors.append("Payload lacks both 'current' and 'hourly' weather sections")
        
    if "latitude" not in raw_data or "longitude" not in raw_data:
        errors.append("Payload lacks geographic coordinates (latitude, longitude)")
        
    return len(errors) == 0, errors


def validate_live_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Validate and enforce meteorological physical bounds on normalized live forecast DataFrame.
    
    Returns:
        (cleaned_df, quality_metrics_dict)
    """
    if df.empty:
        return df, {"status": "EMPTY", "valid_records": 0, "dropped_records": 0}
        
    clean_df = df.copy()
    initial_count = len(clean_df)
    
    # 1. Drop records with NaN in critical columns
    clean_df = clean_df.dropna(subset=["timestamp", "variable", "forecast_value", "lead_time"])
    
    # 2. Check and clip/filter physical bounds
    valid_mask = pd.Series(True, index=clean_df.index)
    
    for var_name, (min_val, max_val) in PHYSICAL_BOUNDS.items():
        var_mask = clean_df["variable"] == var_name
        if var_mask.any():
            out_of_bounds = var_mask & (
                (clean_df["forecast_value"] < min_val) | (clean_df["forecast_value"] > max_val)
            )
            if out_of_bounds.any():
                valid_mask = valid_mask & (~out_of_bounds)
                
    clean_df = clean_df[valid_mask].copy()
    dropped_count = initial_count - len(clean_df)
    
    metrics = {
        "status": "VALID" if dropped_count == 0 else "PARTIAL_FILTERED",
        "initial_records": initial_count,
        "valid_records": len(clean_df),
        "dropped_records": dropped_count,
    }
    
    return clean_df, metrics
