"""
Staleness detection and freshness evaluation for live meteorological records.
"""

from datetime import datetime, timezone
from typing import Tuple

from .config import LiveConfig, get_live_config
from .schemas import DataFreshnessStatus


def evaluate_data_freshness(
    retrieved_at_iso: str,
    config: LiveConfig,
) -> Tuple[DataFreshnessStatus, float]:
    """
    Calculate the age of the data and determine its operational freshness status.
    
    Returns:
        (DataFreshnessStatus, age_in_minutes)
    """
    try:
        ts_clean = retrieved_at_iso.replace("Z", "+00:00")
        dt_retrieved = datetime.fromisoformat(ts_clean)
        if dt_retrieved.tzinfo is None:
            dt_retrieved = dt_retrieved.replace(tzinfo=timezone.utc)
            
        now_utc = datetime.now(timezone.utc)
        age_seconds = (now_utc - dt_retrieved).total_seconds()
        age_minutes = max(0.0, age_seconds / 60.0)
        
        if age_minutes < config.fresh_threshold_minutes:
            return DataFreshnessStatus.FRESH, round(age_minutes, 1)
        elif age_minutes < config.stale_threshold_minutes:
            return DataFreshnessStatus.STALE, round(age_minutes, 1)
        else:
            return DataFreshnessStatus.EXPIRED, round(age_minutes, 1)
            
    except Exception:
        return DataFreshnessStatus.UNAVAILABLE, 9999.0
