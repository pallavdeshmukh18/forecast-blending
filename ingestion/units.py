"""
Unit conversion and normalization layer for meteorological quantities.
Standard internal units:
- temperature: Celsius (°C)
- rainfall: millimeters (mm)
- wind_speed: meters per second (m/s)
"""

from typing import Dict, Optional, Union
import numpy as np
import pandas as pd

from configs import AppConfig, get_config

# Standard target units
TARGET_UNITS = {
    "temperature": "celsius",
    "rainfall": "mm",
    "wind_speed": "m/s",
}


def _clean_unit_str(unit_str: str) -> str:
    """Normalize unit string for matching."""
    if not isinstance(unit_str, str):
        return ""
    return unit_str.strip().lower().replace(" ", "").replace("_", "").replace("-", "").replace("^", "").replace("°", "")


def convert_temperature(values: Union[float, np.ndarray, pd.Series], from_unit: str, to_unit: str = "celsius") -> Union[float, np.ndarray, pd.Series]:
    """Convert temperature values between units."""
    u_from = _clean_unit_str(from_unit)
    u_to = _clean_unit_str(to_unit)

    if u_from == u_to or not u_from:
        return values

    # First convert from source to Celsius
    if u_from in ["k", "kelvin", "kelvins"]:
        celsius = values - 273.15
    elif u_from in ["f", "fahrenheit", "degf"]:
        celsius = (values - 32.0) * (5.0 / 9.0)
    elif u_from in ["c", "celsius", "degc"]:
        celsius = values
    else:
        raise ValueError(f"Unsupported temperature unit: '{from_unit}'. Expected Kelvin, Fahrenheit, or Celsius.")

    # Then convert from Celsius to target
    if u_to in ["c", "celsius", "degc"]:
        return celsius
    elif u_to in ["k", "kelvin", "kelvins"]:
        return celsius + 273.15
    elif u_to in ["f", "fahrenheit", "degf"]:
        return (celsius * 9.0 / 5.0) + 32.0
    else:
        raise ValueError(f"Unsupported target temperature unit: '{to_unit}'.")


def convert_rainfall(values: Union[float, np.ndarray, pd.Series], from_unit: str, to_unit: str = "mm") -> Union[float, np.ndarray, pd.Series]:
    """Convert rainfall / precipitation values between units."""
    u_from = _clean_unit_str(from_unit)
    u_to = _clean_unit_str(to_unit)

    if u_from == u_to or not u_from:
        return values

    # Convert from source to mm
    if u_from in ["m", "meter", "meters"]:
        mm = values * 1000.0
    elif u_from in ["in", "inch", "inches"]:
        mm = values * 25.4
    elif u_from in ["cm", "centimeter", "centimeters"]:
        mm = values * 10.0
    elif u_from in ["kgm2", "kg/m2", "kgm2", "kg/m^2"]:
        # 1 kg/m^2 of liquid water accumulation == 1 mm
        mm = values * 1.0
    elif u_from in ["mm", "millimeter", "millimeters"]:
        mm = values
    else:
        raise ValueError(f"Unsupported rainfall unit: '{from_unit}'. Expected m, cm, mm, inches, or kg/m^2.")

    # Convert from mm to target
    if u_to in ["mm", "millimeter", "millimeters"]:
        return mm
    elif u_to in ["m", "meter", "meters"]:
        return mm / 1000.0
    elif u_to in ["in", "inch", "inches"]:
        return mm / 25.4
    elif u_to in ["cm", "centimeter", "centimeters"]:
        return mm / 10.0
    else:
        raise ValueError(f"Unsupported target rainfall unit: '{to_unit}'.")


def convert_wind_speed(values: Union[float, np.ndarray, pd.Series], from_unit: str, to_unit: str = "m/s") -> Union[float, np.ndarray, pd.Series]:
    """Convert wind speed values between units."""
    u_from = _clean_unit_str(from_unit)
    u_to = _clean_unit_str(to_unit)

    if u_from == u_to or not u_from:
        return values

    # Convert from source to m/s
    if u_from in ["km/h", "kmh", "kph", "kmperhour", "kilometer/hour"]:
        mps = values / 3.6
    elif u_from in ["kt", "kts", "knot", "knots"]:
        mps = values * 0.5144444444444445
    elif u_from in ["mph", "mile/hour", "milesperhour"]:
        mps = values * 0.44704
    elif u_from in ["m/s", "mps", "meter/sec", "meterspersecond", "ms1"]:
        mps = values
    else:
        raise ValueError(f"Unsupported wind speed unit: '{from_unit}'. Expected m/s, km/h, knots, or mph.")

    # Convert from m/s to target
    if u_to in ["m/s", "mps", "meter/sec", "meterspersecond", "ms1"]:
        return mps
    elif u_to in ["km/h", "kmh", "kph", "kmperhour"]:
        return mps * 3.6
    elif u_to in ["kt", "kts", "knot", "knots"]:
        return mps / 0.5144444444444445
    elif u_to in ["mph", "mile/hour", "milesperhour"]:
        return mps / 0.44704
    else:
        raise ValueError(f"Unsupported target wind speed unit: '{to_unit}'.")


def convert_units(
    values: Union[float, np.ndarray, pd.Series],
    from_unit: str,
    to_unit: Optional[str] = None,
    variable: str = "temperature",
) -> Union[float, np.ndarray, pd.Series]:
    """
    Universal unit conversion function for meteorological variables.

    Parameters
    ----------
    values : float, np.ndarray, or pd.Series
        Input values to convert.
    from_unit : str
        Source unit string (e.g. 'K', 'km/h', 'meters').
    to_unit : Optional[str]
        Target unit. If None, uses internal standard unit for the given variable.
    variable : str
        Meteorological variable ('temperature', 'rainfall', 'wind_speed').

    Returns
    -------
    Converted value(s) in target unit.
    """
    var_clean = variable.strip().lower()
    target = to_unit or TARGET_UNITS.get(var_clean, "standard")

    if var_clean == "temperature":
        return convert_temperature(values, from_unit=from_unit, to_unit=target)
    elif var_clean == "rainfall":
        return convert_rainfall(values, from_unit=from_unit, to_unit=target)
    elif var_clean == "wind_speed":
        return convert_wind_speed(values, from_unit=from_unit, to_unit=target)
    else:
        # Unknown variable, return unchanged
        return values


def normalize_dataframe_units(
    df: pd.DataFrame,
    value_col: str = "forecast_value",
    variable_col: str = "variable",
    unit_col: Optional[str] = None,
    source_units: Optional[Dict[str, str]] = None,
    config: Optional[AppConfig] = None,
) -> pd.DataFrame:
    """
    Normalize physical values in a DataFrame to standard internal units.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    value_col : str, default 'forecast_value'
        Name of numerical value column (e.g. 'forecast_value' or 'observed_value').
    variable_col : str, default 'variable'
        Name of variable category column.
    unit_col : Optional[str]
        If present in DataFrame, specifies row-wise source unit.
    source_units : Optional[Dict[str, str]]
        Dictionary mapping variable names to their source unit (e.g. {'temperature': 'kelvin', 'rainfall': 'm'}).
    config : Optional[AppConfig]
        Application configuration.

    Returns
    -------
    pd.DataFrame
        DataFrame with normalized numerical values.
    """
    out = df.copy()
    if value_col not in out.columns or variable_col not in out.columns:
        return out

    cfg = config or get_config()
    target_units = {
        "temperature": cfg.units.temperature if cfg else "celsius",
        "rainfall": cfg.units.rainfall if cfg else "mm",
        "wind_speed": cfg.units.wind_speed if cfg else "m/s",
    }

    # Case 1: row-wise unit column
    if unit_col and unit_col in out.columns:
        for idx, row in out.iterrows():
            u = row[unit_col]
            var = row[variable_col]
            if pd.notna(u) and u:
                tgt = target_units.get(var, "standard")
                out.at[idx, value_col] = convert_units(row[value_col], from_unit=str(u), to_unit=tgt, variable=var)
        return out

    # Case 2: source units specified per variable
    if source_units:
        for var_name, from_u in source_units.items():
            mask = out[variable_col] == var_name
            if mask.any() and from_u:
                tgt = target_units.get(var_name, "standard")
                out.loc[mask, value_col] = convert_units(
                    out.loc[mask, value_col],
                    from_unit=from_u,
                    to_unit=tgt,
                    variable=var_name,
                )

    return out
