"""
Variable name standardization and mapping across meteorological providers.
"""

from typing import Dict, List, Optional
import pandas as pd

from configs import AppConfig, get_config

# Default alias dictionary mapping standard names to common provider aliases
DEFAULT_VARIABLE_ALIASES: Dict[str, List[str]] = {
    "temperature": [
        "temperature",
        "t2m",
        "t2m_degc",
        "t2m_k",
        "2m_temperature",
        "2t",
        "temp",
        "tmp",
        "t_2m",
        "air_temperature",
        "t_surface",
        "tas",
    ],
    "rainfall": [
        "rainfall",
        "rain",
        "tp",
        "precipitation",
        "precip",
        "total_precipitation",
        "total_precip",
        "total_precipitation_6hr",
        "total_precipitation_24hr",
        "tp_6hr",
        "tp_24hr",
        "apcp",
        "prate",
        "rain_mm",
        "accumulated_precipitation",
        "precip_accum",
        "pr",
    ],
    "wind_speed": [
        "wind_speed",
        "wind",
        "ws10",
        "10m_wind",
        "10m_wind_speed",
        "10m_windspeed",
        "wind_speed_10m",
        "10si",
        "si10",
        "wspd",
        "windspeed",
        "speed_10m",
        "wind_10m",
        "sfc_wind",
        "sfcwind",
    ],
}


def _build_lookup(config: Optional[AppConfig] = None) -> Dict[str, str]:
    """Build a fast reverse lookup dictionary from aliases to standard names."""
    cfg = config or get_config()
    alias_dict = DEFAULT_VARIABLE_ALIASES.copy()
    if cfg and cfg.variable_mappings:
        for std_name, aliases in cfg.variable_mappings.items():
            if std_name in alias_dict:
                alias_dict[std_name] = list(set(alias_dict[std_name] + aliases))
            else:
                alias_dict[std_name] = aliases

    lookup = {}
    for std_name, aliases in alias_dict.items():
        lookup[std_name.lower().replace("_", "").replace("-", "")] = std_name
        for alias in aliases:
            norm_alias = alias.lower().replace("_", "").replace("-", "")
            lookup[norm_alias] = std_name

    return lookup


def standardize_variable_name(raw_name: str, config: Optional[AppConfig] = None) -> str:
    """
    Map an external meteorological variable name to the project's internal standardized name.

    Parameters
    ----------
    raw_name : str
        Variable name from external dataset (e.g. 't2m', 'total_precipitation', 'ws10').
    config : Optional[AppConfig]
        Optional system configuration instance.

    Returns
    -------
    str
        Internal standardized name ('temperature', 'rainfall', 'wind_speed', or unchanged if unknown).
    """
    if not isinstance(raw_name, str):
        return str(raw_name)

    lookup = _build_lookup(config)
    norm_name = raw_name.strip().lower().replace("_", "").replace("-", "")
    return lookup.get(norm_name, raw_name.strip().lower())


def standardize_dataframe_variables(
    df: pd.DataFrame,
    variable_col: str = "variable",
    config: Optional[AppConfig] = None,
) -> pd.DataFrame:
    """
    Standardize the contents of the variable column in a long-format DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing a variable column.
    variable_col : str, default 'variable'
        Name of the column containing variable names.
    config : Optional[AppConfig]
        Optional system configuration instance.

    Returns
    -------
    pd.DataFrame
        DataFrame with standardized variable names.
    """
    if variable_col not in df.columns:
        return df

    out = df.copy()
    out[variable_col] = out[variable_col].astype(str).apply(lambda v: standardize_variable_name(v, config))
    return out
