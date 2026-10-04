# WeatherBench 2 Real-Data Integration Report

**System**: Hybrid AI–NWP Multi-Model Forecast Blending System  
**Version**: `0.6.0`  
**Integration Status**: Complete & Verified  
**Date**: October 3, 2026  

---

## 1. Executive Summary

This report documents the successful integration of real meteorological dataset streams from **WeatherBench 2** (ECMWF & Google Cloud Open Datasets) into the multi-model forecasting pipeline.

The integration establishes a reproducible, configurable pipeline capable of ingesting ground-truth atmospheric reanalysis (ERA5) alongside deterministic physics-based Numerical Weather Prediction (ECMWF HRES) and data-driven Neural Weather Prediction (GraphCast v2, Pangu-Weather).

### Core Milestones Achieved:
1. **Zero Synthetic Leakage**: WeatherBench 2 ingestion operates exclusively on real external Zarr datasets. No synthetic fallbacks are invoked during real-data execution.
2. **Synthetic Baseline Isolation**: The synthetic Phase 6 benchmark remains 100% byte-for-byte reproducible (`SHA-256: 54cb659a7d8e86f0014114e616d2ecbf834d574b877b8db69326e6440c8223fd`).
3. **100% Test Suite Pass**: All 211 backend tests (199 existing + 12 new WeatherBench 2 tests) pass with zero errors and zero warnings.
4. **Benchmark Boundary Respected**: Stopped prior to executing the full Phase 6 real-data benchmark, strictly completing data ingestion, quality control, spatial/temporal alignment, and real-data smoke verification.

---

## 2. Dataset Architecture & Access

WeatherBench 2 datasets are hosted on Google Cloud Storage in cloud-optimized Zarr format with consolidated metadata (`.zmetadata`). Slicing and extraction are executed via anonymous public streaming (`gcsfs` / `xarray`).

### Dataset Identifiers & Endpoints:

| Role | Provider / Model | GCS Zarr Endpoint | Resolution / Mesh |
| :--- | :--- | :--- | :--- |
| **Observation Truth** | ECMWF ERA5 Reanalysis | `gs://weatherbench2/datasets/era5/1959-2022-6h-240x121_equiangular_with_poles_conservative.zarr` | 1.5° (240 × 121) |
| **Physics NWP** | ECMWF HRES Deterministic | `gs://weatherbench2/datasets/hres/2016-2022-0012-240x121_equiangular_with_poles_conservative.zarr` | 1.5° (240 × 121) |
| **Neural AI NWP** | Google DeepMind GraphCast v2 | `gs://weatherbench2/datasets/graphcast_v2/2020-240x121_equiangular_with_poles_conservative.zarr` | 1.5° (240 × 121) |
| **Neural AI NWP** | Huawei Cloud Pangu-Weather | `gs://weatherbench2/datasets/pangu/2018-2022_0012_240x121_equiangular_with_poles_conservative.zarr` | 1.5° (240 × 121) |

---

## 3. Spatial & Temporal Configurations

- **Geographic Domain**: South Asia / Indian Subcontinent Domain
  - Latitude Range: `[10.0°N, 30.0°N]` ($N=14$ grid points, step $1.5^\circ$)
  - Longitude Range: `[70.0°E, 90.0°E]` ($N=14$ grid points, step $1.5^\circ$)
  - Spatial Mesh Size: $14 \times 14 = 196$ geographical grid nodes
- **Temporal Verification Window**:
  - Sample Range: `2020-01-01T00:00:00Z` to `2020-01-03T23:59:59Z`
  - Temporal Frequency: 6-hourly cycle (`00:00`, `06:00`, `12:00`, `18:00` UTC)
- **Forecast Lead Times**:
  - Horizon Steps: $+6\text{h}$, $+12\text{h}$, $+24\text{h}$, $+48\text{h}$

---

## 4. Ingested Records & Pairing Statistics

The integration pipeline ingested and verified the following records:

```text
============================================================
WEATHERBENCH 2 SMOKE TEST INGESTION METRICS
============================================================
Forecast Models:          ECMWF HRES, GraphCast v2
Meteorological Variables: temperature, wind_speed, rainfall
Temporal Range:           2020-01-01 to 2020-01-03 (6-hour steps)
Spatial Grid Size:        196 grid points (14x14 equiangular)

Raw Records Extracted:
  - ERA5 Ground-Truth Observations: 7,056
  - ECMWF HRES Forecasts:          14,112
  - GraphCast v2 Forecasts:        28,224
  - Total Raw Forecast Records:    42,336

Quality Control Assessment:
  - Valid Forecast Records:        42,336 (100.0%)
  - Valid Observation Records:     7,056  (100.0%)
  - Flagged / Rejected Records:    0
  - Missing Values (NaN/Inf):      0

Verification Pairing:
  - Matched Verified Pairs:        29,400
  - Unmatched Forecasts:           12,936*
  - Missing Value Percentage:      0.00%
============================================================
```

*\*Note on Unmatched Forecasts: Forecasts initialized at late cycles (e.g. Jan 3 12:00 UTC) with lead times $+24\text{h}$ and $+48\text{h}$ verify at valid times outside the test observation extraction window (Jan 4–5), which is mathematically expected and demonstrates strict causal filtering.*

---

## 5. Variable Transformations & Physical Semantics

Every physical meteorological quantity is mapped into the project's canonical schema with verified units:

| Canonical Variable | WeatherBench 2 Source Variable | Source Raw Unit | Target Canonical Unit | Transformation Applied |
| :--- | :--- | :--- | :--- | :--- |
| **`temperature`** | `2m_temperature` | Kelvin ($K$) | Celsius ($^\circ\text{C}$) | $T_{^\circ\text{C}} = T_K - 273.15$ |
| **`rainfall`** | `total_precipitation_6hr` | meters ($m$) | millimeters ($mm$) | $P_{mm} = P_m \times 1000.0$ |
| **`wind_speed`** | `10m_wind_speed` or `10m_u`/`10m_v` | $\text{m/s}$ | $\text{m/s}$ | Direct $\text{si10}$ or $\sqrt{u_{10}^2 + v_{10}^2}$ |
| **`longitude`** | `longitude` | $[0.0, 358.5]^\circ$ | $[-180.0, 180.0]^\circ$ | $((lon + 180) \bmod 360) - 180$ |
| **`timestamp`** | `time` + `prediction_timedelta` | UTC datetime | UTC valid time | $\text{valid\_time} = \text{run\_time} + \text{lead\_time}$ |

---

## 6. What is Actually Available in WeatherBench 2

To maintain scientific rigor, model and variable coverage is transparently delineated:

### 1. Available Now:
- **ERA5**: `2m_temperature` (K), `10m_wind_speed` (m/s), `total_precipitation_6hr` (m), `mean_sea_level_pressure`, `geopotential`, `specific_humidity`.
- **ECMWF HRES**: `2m_temperature` (K), `10m_wind_speed` (m/s), `total_precipitation_6hr` (m).
- **GraphCast v2**: `2m_temperature` (K), `10m_wind_speed` (m/s), `10m_u_component_of_wind`, `10m_v_component_of_wind`, `total_precipitation_6hr` (m), `total_precipitation_24hr` (m).
- **Pangu-Weather**: `2m_temperature` (K), `10m_wind_speed` (m/s), `10m_u_component_of_wind`, `10m_v_component_of_wind`, upper-air geopotential and wind.

### 2. Not Available / Limitations:
- **Pangu Precipitation**: Pangu-Weather does not output precipitation accumulation fields; evaluating precipitation blending requires HRES + GraphCast or another rainfall-capable NWP model.
- **Hourly Accumulation**: Standard WeatherBench 2 conservative 1.5° stores provide 6-hourly accumulation increments rather than 1-hour station intervals.

---

## 7. Data Artifacts & Cryptographic Provenance

All raw extracts, standardized dataframes, verified pairs, and provenance files are stored with complete isolation from synthetic files:

```text
ML/
├── data/
│   ├── raw/
│   │   └── weatherbench2/
│   │       ├── weatherbench2_forecasts_raw.parquet
│   │       └── weatherbench2_observations_raw.parquet
│   ├── processed/
│   │   └── real/
│   │       └── weatherbench2/
│   │           ├── real_forecasts.parquet
│   │           ├── real_observations.parquet
│   │           └── real_verification_dataset.parquet
│   └── provenance/
│       └── weatherbench2/
│           ├── manifest.json
│           └── provenance.json
```

### Cryptographic SHA-256 Hashes:
- **`raw_forecasts.parquet`**: `0b529aed2d944c156b096e71696c0d8a3b7c4bd8ea149d814aefc2f28912941b`
- **`raw_observations.parquet`**: `35dca80a92a8b5225a81743708d5facf426632ea7cf3b66f950d988d01e1b96b`
- **`real_forecasts.parquet`**: `c4284c7281eeb621fdd7561ae8288f829ae9e6908dfda1f6dea91c3472622bf9`
- **`real_observations.parquet`**: `aa675f489d68a54f12b6824cd42b0e949e7099301529faa0c70b01dab28f27fc`
- **`real_verification_dataset.parquet`**: `001bd05e2b6b0b1f9e444a02cc533aa0ea31a35c047d6818d9900078cf5ce751`

---

## 8. Synthetic Baseline Preservation Check

- **Pre-Integration Synthetic Benchmark SHA-256**: `54cb659a7d8e86f0014114e616d2ecbf834d574b877b8db69326e6440c8223fd`
- **Post-Integration Synthetic Benchmark SHA-256**: `54cb659a7d8e86f0014114e616d2ecbf834d574b877b8db69326e6440c8223fd`
- **Status**: **100% Byte-for-Byte Identical (Unchanged)**

---

## 9. Conclusion & Next Steps

The WeatherBench 2 real meteorological data integration is complete and verified. The repository is in a ready state for real-data evaluation without risking synthetic baseline drift or data leakage.

**Stopping condition satisfied**: No full real-data benchmark was executed.
