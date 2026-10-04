"""
Background scheduler for periodic live weather data refresh in FastAPI.
"""

import asyncio
from datetime import datetime, timezone
import logging
from typing import Optional

from .config import LiveConfig, get_live_config
from .service import LiveForecastService

logger = logging.getLogger(__name__)


class LiveForecastScheduler:
    """
    Asynchronous background scheduler for keeping live forecast caches warm and fresh.
    """

    def __init__(self, service: Optional[LiveForecastService] = None, config: Optional[LiveConfig] = None):
        self.config = config or get_live_config()
        self.service = service or LiveForecastService(self.config)
        self._task: Optional[asyncio.Task] = None
        self._is_running = False

    async def start(self):
        """Start the background refresh loop if live mode is enabled."""
        if not self.config.live_mode_enabled:
            logger.info("LIVE_SCHEDULER: Live mode is disabled; scheduler will not start.")
            return

        if self._is_running:
            logger.warning("LIVE_SCHEDULER: Scheduler is already running.")
            return

        self._is_running = True
        self._task = asyncio.create_task(self._refresh_loop())
        logger.info(f"LIVE_SCHEDULER_STARTED: Background loop active (interval: {self.config.refresh_interval_minutes}m)")

    async def stop(self):
        """Gracefully terminate the background refresh task."""
        if not self._is_running:
            return

        self._is_running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("LIVE_SCHEDULER_STOPPED: Background loop safely stopped.")

    async def _refresh_loop(self):
        """Infinite async loop refreshing live forecasts periodically."""
        # Initial warmup fetch
        try:
            logger.info("LIVE_SCHEDULER: Executing initial cache warmup...")
            # Run in thread pool to avoid blocking async event loop
            await asyncio.to_thread(self.service.get_live_forecast, force_refresh=False)
        except Exception as e:
            logger.warning(f"LIVE_SCHEDULER: Initial warmup warning: {e}")

        while self._is_running:
            try:
                sleep_seconds = max(60, self.config.refresh_interval_minutes * 60)
                await asyncio.sleep(sleep_seconds)
                if not self._is_running:
                    break
                logger.info("LIVE_SCHEDULER: Running scheduled live data refresh...")
                await asyncio.to_thread(self.service.get_live_forecast, force_refresh=True)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"LIVE_SCHEDULER_ERROR: Scheduled refresh encountered an error: {e}")
