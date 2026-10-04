"""
Local bounded disk cache for live operational forecasts and observations.
"""

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import LiveConfig, get_live_config

logger = logging.getLogger(__name__)


class LiveDataCache:
    """
    Manages bounded local disk caching for live operational weather data.
    """

    def __init__(self, config: Optional[LiveConfig] = None):
        self.config = config or get_live_config()
        self.cache_dir = self.config.cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.latest_cache_file = self.cache_dir / "latest_live_payload.json"

    def save(self, payload: Dict[str, Any], key: Optional[str] = None) -> Path:
        """Save a live payload to cache with timestamp and prune old files."""
        now_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        fname = f"live_payload_{now_str}.json" if not key else f"live_payload_{key}_{now_str}.json"
        target_path = self.cache_dir / fname
        
        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
            
        # Also maintain latest pointer
        with open(self.latest_cache_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
            
        logger.info(f"LIVE_CACHE_SAVED: Saved live payload to {target_path}")
        self._prune_old_cache_files()
        return target_path

    def load_latest(self) -> Optional[Dict[str, Any]]:
        """Load the most recent cached live payload if available."""
        if self.latest_cache_file.exists():
            try:
                with open(self.latest_cache_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    logger.info("LIVE_CACHE_USED: Successfully loaded latest cached live payload")
                    return data
            except Exception as e:
                logger.warning(f"Failed to read latest cache file: {e}")
                
        # Fallback: scan directory for newest json file
        json_files = sorted(self.cache_dir.glob("live_payload_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        if json_files:
            try:
                with open(json_files[0], "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to read fallback cache file {json_files[0]}: {e}")
                
        return None

    def _prune_old_cache_files(self):
        """Keep only the newest max_cache_files in cache directory."""
        json_files = sorted(self.cache_dir.glob("live_payload_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        if len(json_files) > self.config.max_cache_files:
            for old_file in json_files[self.config.max_cache_files :]:
                try:
                    old_file.unlink()
                except Exception:
                    pass
