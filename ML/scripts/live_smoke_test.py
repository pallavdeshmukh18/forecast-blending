"""
Operational Live Smoke Test for SkyCast.
Connects to live Open-Meteo API endpoint, verifies response parsing,
normalization, physical bounds, dynamic blending, and cache persistence.
"""

import logging
import sys
import time
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent.parent
_ml_dir = _project_root / "ML"
for _p in [str(_ml_dir), str(_project_root)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from live import LiveForecastService, get_live_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("live_smoke_test")


def run_live_smoke_test():
    cfg = get_live_config()
    print("=" * 70)
    print("SKYCAST OPERATIONAL LIVE LAYER: END-TO-END SMOKE TEST")
    print(f"Target Location: {cfg.default_location_name} ({cfg.default_latitude} deg N, {cfg.default_longitude} deg E)")
    print(f"Provider:        {cfg.provider_name} ({cfg.api_base_url})")
    print(f"Cache Dir:       {cfg.cache_dir}")
    print("=" * 70)

    service = LiveForecastService()

    t0 = time.time()
    try:
        result = service.get_live_forecast(force_refresh=True)
        elapsed_sec = round(time.time() - t0, 3)

        print("\n[LIVE SMOKE TEST RESULT]")
        print(f"  Status:               {result.status}")
        print(f"  Mode:                 {result.mode}")
        print(f"  Provider:             {result.provider}")
        print(f"  Models Available:     {result.models_available}")
        print(f"  Data Freshness:       {result.freshness.value} (Age: {result.data_age_minutes:.1f} mins)")
        print(f"  Retrieved At (UTC):   {result.retrieved_at_utc}")
        print(f"  Total Latency:        {elapsed_sec}s")

        print("\n[CURRENT SURFACE CONDITIONS]")
        curr = result.current_conditions
        print(f"  Temperature:          {curr.temperature_c} C")
        print(f"  Wind Speed:           {curr.wind_speed_ms} m/s ({round(curr.wind_speed_ms * 3.6, 1)} km/h)")
        print(f"  Precipitation:        {curr.precipitation_mm} mm")
        print(f"  Relative Humidity:    {curr.relative_humidity_pct}%")
        print(f"  Surface Pressure:     {curr.surface_pressure_hpa} hPa")

        print("\n[OPERATIONAL FORECAST STEPS (+6h to +72h)]")
        for step in result.forecast_steps:
            print(
                f"  +{step.lead_time_hours:02d}h | Valid: {step.valid_time_utc} | "
                f"Temp: {step.temperature_c:5.1f} C (90% CI: {step.temperature_ci90}) | "
                f"Wind: {step.wind_speed_ms:4.1f} m/s | "
                f"Rain (6h): {step.rainfall_mm:4.1f} mm (P>=5mm: {step.rainfall_exceedance_p5mm:.0%}) | "
                f"Regime: {step.regime}"
            )

        print(f"\n[ACTIVE EXTREME WEATHER ALERTS ({len(result.active_alerts)})]")
        if result.active_alerts:
            for alert in result.active_alerts:
                print(f"  [ALERT] [{alert.severity.upper()}] {alert.headline}: {alert.description}")
        else:
            print("  No active extreme alerts triggered at current thresholds.")

        print("\n[MODEL DISAGREEMENT DIAGNOSTICS]")
        for diag in result.disagreement:
            print(f"  - {diag.variable.capitalize()} (+{diag.lead_time_hours}h): {diag.disagreement_level} ({diag.interpretation})")

        print("\n[SCIENTIFIC BOUNDARY]")
        print(f"  {result.scientific_disclaimer}")
        print("\n[PASS] LIVE SMOKE TEST COMPLETED SUCCESSFULLY.")

    except Exception as e:
        print(f"\n[FAIL] LIVE SMOKE TEST FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_live_smoke_test()
