# Hybrid AI–NWP Multi-Model Forecast Blending System

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/pytest-245%20passed-brightgreen.svg)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)]()

A scientifically rigorous meteorological framework designed to dynamically blend multi-model weather forecasts from Numerical Weather Prediction (NWP) systems, multi-member ensembles, and deep-learning neural weather models.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Scientific Principles & Blending Formulations](#2-scientific-principles--blending-formulations)
3. [System Architecture](#3-system-architecture)
4. [Data Ingestion, Quality Control & Provenance](#4-data-ingestion-quality-control--provenance)
5. [Atmospheric Regime Detection & Adaptive Blending](#5-atmospheric-regime-detection--adaptive-blending)
6. [Spatial CNN & Spatio-Temporal ConvGRU Memory](#6-spatial-cnn--spatio-temporal-convgru-memory)
7. [Probabilistic Forecasting & Extreme Calibration](#7-probabilistic-forecasting--extreme-calibration)
8. [Scientific Benchmark Results (WeatherBench 2 Audit)](#8-scientific-benchmark-results-weatherbench-2-audit)
9. [Repository Structure](#9-repository-structure)
10. [Installation & Setup](#10-installation--setup)
11. [Quickstart & Execution Guide](#11-quickstart--execution-guide)
12. [FastAPI Endpoints & API Reference](#12-fastapi-endpoints--api-reference)
13. [Automated Testing & Validation](#13-automated-testing--validation)
14. [Interactive Operational Portal & 3D WebGIS Workspace](#14-interactive-operational-portal--3d-webgis-workspace)

---

## 1. Project Overview

Weather forecasting accuracy varies significantly across model paradigms:
* **Physics-based NWP models** (e.g., NOAA GFS, ECMWF IFS, WRF) excel at medium-range synoptic dynamics and conserve mass and energy, but exhibit parameterization biases and topography smoothing.
* **Data-driven AI weather models** (e.g., GraphCast, Pangu-Weather, FourCastNet) achieve high computational efficiency and non-linear resolution in short horizons, but can suffer from spatial smoothing over extended lead times.
* **Ensemble Prediction Systems** provide probabilistic bounds but require post-processing and calibration.

The **Hybrid AI–NWP Multi-Model Forecast Blending System** dynamically determines optimal, convex model weights:

$$W_m(x, y, t) \ge 0, \quad \sum_{m=1}^M W_m(x, y, t) = 1.0$$

conditioned on:
* **Geographic region** (plains, coastal marine transitions, complex mountainous orography).
* **Forecast lead time horizon** (short-range 0–24h, medium-range 24–72h, extended-range >72h).
* **Climatological season & diurnal calendar cycles**.
* **Synoptic weather regimes** (convective storm, heavy rain, high wind, temperature extremes, stable).
* **Local spatial gradients and model disagreement fields**.
* **Rolling historical skill statistics**.

---

## 2. Scientific Principles & Blending Formulations

### 2.1 Baseline Equal-Weight Blend
For $N$ reporting models at spatiotemporal coordinate $(x, y, t)$ and lead horizon $\tau$:

$$F_{\text{blend}}(x, y, t, \tau) = \frac{1}{N} \sum_{i=1}^{N} F_i(x, y, t, \tau)$$

### 2.2 Inverse-Variance & Inverse-MAE Skill Weighting
Raw skill weights are computed from historical error variance $\sigma_i^2$ or mean absolute error $\text{MAE}_i$:

$$w_i = \frac{1 / (\sigma_i^2 + \epsilon)}{\sum_{j=1}^N 1 / (\sigma_j^2 + \epsilon)}$$

with weight smoothing toward the uninformative prior ($1/N$):

$$w_i' = (1 - \alpha) w_i + \alpha \frac{1}{N} \quad (\alpha = 0.10)$$

### 2.3 Constrained Simplex Ridge Regression
Models the relationship between member forecasts and ground truth observations:

$$\min_{\mathbf{w}} \frac{1}{2M} \|\mathbf{y} - (\beta_0 + \mathbf{X}\mathbf{w})\|_2^2 + \frac{\alpha}{2} \|\mathbf{w}\|_2^2 \quad \text{s.t.} \quad w_i \ge 0, \quad \sum_{i=1}^N w_i = 1$$

### 2.4 Hierarchical Skill Fallback
When sample counts in a granular context are below `minimum_samples` ($N < 20$), skill queries fall back deterministically:

```text
[Scope 5] model + variable + lead_time + region + regime
     ↓ (if N < minimum_samples)
[Scope 4] model + variable + lead_time + regime
     ↓ (if N < minimum_samples)
[Scope 3] model + variable + lead_time + region
     ↓ (if N < minimum_samples)
[Scope 2] model + variable + lead_time
     ↓ (if N < minimum_samples)
[Scope 1] model + variable
     ↓ (if N < minimum_samples)
[Scope 0] global model prior
```

---

## 3. System Architecture

```text
               REAL & SYNTHETIC DATA SOURCES
             (NWP, AI, AWS Stations, Radar, Satellite)
                             │
                             ▼
               Provider-Neutral Ingestion
                             │
                             ▼
               Quality Control & Provenance
               (reports/data_quality.json)
                             │
                             ▼
              Spatiotemporal Grid Alignment
                             │
                             ▼
                Atmospheric Feature Stack
                             │
              ┌──────────────┴──────────────┐
              ▼                             ▼
       Weather Regimes              Historical Skill
       (Rule & KMeans)             (Hierarchical DB)
              │                             │
              └──────────────┬──────────────┘
                             ▼
             Spatial CNN & ConvGRU Recurrent State
                             │
                             ▼
             Dynamic Convex Weights W_m(x,y,t)
                             │
                             ▼
                    Multi-Model Blend
                             │
              ┌──────────────┴──────────────┐
              ▼                             ▼
      Deterministic Blend          Probabilistic Spread
      (MAE, RMSE, Bias, r)       (EMOS, Zero-Inflated Gamma)
              │                             │
              └──────────────┬──────────────┘
                             ▼
               Extreme Exceedance Probabilities
               (Rain >= 25mm, Wind, Heatwave)
                             │
                             ▼
                FastAPI Operational Backend
                             │
                             ▼
                Streamlit Research Dashboard
```

---

## 4. Data Ingestion, Quality Control & Provenance

### 4.1 Supported Formats
* **CSV / TSV**: `.csv`, `.tsv`, `.txt`
* **Parquet**: `.parquet`, `.pq`
* **NetCDF**: `.nc`, `.nc4`, `.netcdf` (via `xarray` / `scipy`)
* **GRIB2**: `.grib`, `.grib2`, `.grb2` (via `cfgrib` / `ecCodes`)

### 4.2 Quality Control Diagnostics
Automated checks in `ingestion/quality_control.py`:
1. Coordinate validity: Latitude in `[-90, 90]`, Longitude in `[-180, 180]`
2. Physical bounds: Temperature `[-60°C, +65°C]`, Rain `[0, 600 mm]`, Wind `[0, 120 m/s]`
3. Duplicate timestamp and grid-cell detection
4. Temporal sequence gap diagnostics
5. Provenance audit tracking recorded to `reports/data_provenance.json` and `reports/data_quality.json`.

### 4.3 Verified Pairing
* `forecast_error` = `forecast_value` − `observed_value`
* `absolute_error` = `|forecast_error|`
* `squared_error` = `(forecast_error)²`
* `valid_time` = `initialization_time` + `lead_time`

---

## 5. Atmospheric Regime Detection & Adaptive Blending

### 5.1 Rule-Based Synoptic Regimes
* **heavy_rain**: Precipitation $\ge 50.0\text{ mm}$
* **convective**: Precipitation $\ge 10.0\text{ mm}$ with diurnal heating
* **high_wind**: Wind speed $\ge 15.0\text{ m/s}$
* **hot_extreme**: Temperature $\ge 38.0^\circ\text{C}$
* **cold_extreme**: Temperature $\le 0.0^\circ\text{C}$
* **transition**: Rapid temperature step $|\Delta T| \ge 4.0^\circ\text{C}$
* **stable**: Calm baseline conditions

### 5.2 Unsupervised K-Means Atmospheric Clustering
* Standardized feature pipeline across 9 atmospheric variables.
* Computes Silhouette scores and centroid profiles across $K \in [3, 4, 5, 6]$.

### 5.3 Neural Mixture-of-Experts Gating
* PyTorch Feed-Forward network mapping atmospheric state vectors to model routing logits.
* Channel-wise Softmax guarantees $W_m \ge 0$ and $\sum W_m = 1.0$.

---

## 6. Spatial CNN & Spatio-Temporal ConvGRU Memory

* **2D Spatial CNN Encoder** (`models/spatial_encoder.py`): Preserves fine spatial resolutions over complex terrain with same-padding convolutions.
* **Spatio-Temporal ConvGRU** (`models/temporal_encoder.py`, `models/spatiotemporal_gating.py`): Lightweight convolutional recurrent cell operating across sequential forecast cycles $[t-3, t-2, t-1, t]$, tracking moving convective fronts to predict dynamic spatial gating weights at $t + \text{lead}$.

---

## 7. Probabilistic Forecasting & Extreme Calibration

### 7.1 Gaussian EMOS
Parametric post-processing optimizing affine mean $\mu = a + b\bar{F}$ and spread variance $\sigma^2 = c + dS^2$ via Continuous Ranked Probability Score (CRPS) minimization.

### 7.2 Zero-Inflated Gamma EMOS
Decouples binary dry-day probability from positive rainfall intensity:
* $P(Y = 0) = \text{logit}^{-1}(\gamma_0 + \gamma_1 \bar{F})$
* $Y \mid (Y > 0) \sim \text{Gamma}(\alpha, \beta)$

### 7.3 GEV Tail Calibration
Generalized Extreme Value upper-tail modeling for severe weather return periods (enforced for $N \ge 50$).

---

## 8. Scientific Benchmark Results (WeatherBench 2 Audit)

The system was evaluated on the standardized **WeatherBench 2** conservative reanalysis archive (1.5° grid, South Asia domain: 10.5°N–30.0°N, 70.5°E–90.0°E, 2020 record) across 14 candidate multi-model blending and probabilistic calibration methodologies. The benchmark passed an independent scientific integrity audit (`PASS_WITH_LIMITATIONS`, zero test leakage, genuine model execution).

### 8.1 Key Scientific Findings
1. **Dominant Model Disparity**: Google DeepMind GraphCast v2 substantially outperforms ECMWF HRES across all three variables in South Asia (47.0% lower Temp MAE, 20.7% lower Wind MAE, 45.9% lower Rain MAE).
2. **Conditional Blending Optimization**: Equal-weight averaging ($0.5/0.5$) degrades skill relative to GraphCast alone when constituent errors differ widely. Adaptive skill blending (Inverse-Variance, Ridge) recovers single-model accuracy by dynamically allocating 73–81% weight to GraphCast.
3. **Statistically Established Temperature Gain**: Spatio-Temporal ConvGRU achieves the lowest overall 2m Temperature RMSE (**0.6760 °C** vs GraphCast 0.6856 °C), establishing a statistically significant ~1.4% relative error reduction under paired block-bootstrap testing ($p = 0.0001, 95\%\text{ CI } [+0.0043, +0.0056]^\circ\text{C}$).
4. **Wind Speed Multi-Model Synergy**: For 10m wind speed where model skills are comparable, Spatial CNN achieves the lowest numerical MAE (**0.3043 m/s** vs GraphCast 0.3413 m/s), although differences among top blends are not statistically significant ($p = 0.714$).
5. **Probabilistic Uncertainty Calibration**: Gaussian EMOS and Zero-Inflated Gamma EMOS deliver calibrated prediction intervals ($\text{PICP}_{90} \approx 89.7\%$, Rain Brier score $0.0382$).
6. **Error Diagnostics via Model Disagreement**: Physical-AI forecast disagreement $|F_{\text{HRES}} - F_{\text{GC}}|$ exhibits a robust positive association with forecast error ($r = 0.5310$ on rainfall, $r = 0.4498$ on temperature).

For the full 25-section report, cryptographic manifests, and complete tables, see [`ML/reports/final_research_report.md`](file:///c:/Users/Vedansh/forecast-blending/ML/reports/final_research_report.md).

---

## 9. Repository Structure

```text
forecast-blending/
├── api/
│   ├── __init__.py
│   └── main.py                     # FastAPI REST service & operational endpoints
├── configs/
│   ├── __init__.py                 # Pydantic configuration models
│   ├── config.yaml                 # Central system configuration
│   └── experiments/                # Experiment configuration manifests
├── dashboard/
│   └── app.py                      # Streamlit research dashboard
├── data/
│   ├── provenance.py               # Data provenance and audit tracker
│   ├── synthetic_spatial.py        # 2D gridded synthetic weather generator
│   ├── raw/                        # Raw weather archives
│   ├── processed/                  # Verification datasets and error DB
│   └── samples/                    # Sample test datasets
├── evaluation/
│   ├── error_database.py           # Historical verification error database
│   ├── metrics.py                  # MAE, RMSE, Bias, Correlation metrics
│   ├── probabilistic_metrics.py    # CRPS, CRPSS, PICP, MPIW, Spatial MAE
│   ├── calibration.py              # Brier score and reliability curves
│   ├── final_validation.py         # Multi-dimensional holdout validator
│   └── model_comparison.py         # Moving block bootstrap inference
├── extremes/
│   ├── rainfall.py                 # Extreme rainfall event detection
│   ├── heatwave.py                 # Heatwave hazard detection
│   └── wind.py                     # Gale wind detection
├── ingestion/
│   ├── schema.py                   # Pydantic forecast and observation schemas
│   ├── variable_mapping.py         # Variable alias dictionary
│   ├── units.py                    # Unit conversion and normalization
│   ├── quality_control.py          # Quality control validation and reporting
│   ├── readers.py                  # CSV, Parquet, NetCDF, GRIB2 readers
│   ├── providers/                  # Provider-neutral ingestion interfaces
│   └── synthetic.py                # Synthetic forecast/obs generator
├── models/
│   ├── baseline.py                 # Equal-weight multi-model blender
│   ├── skill_weighting.py          # Inverse-Variance & Inverse-MAE blenders
│   ├── adaptive_blending.py        # Constrained simplex Ridge regression
│   ├── regime_blending.py          # Regime-conditioned skill blender
│   ├── gating_model.py             # Neural gating mixture-of-experts
│   ├── spatial_blending.py         # 2D Spatial inverse-variance blender
│   ├── spatial_gating.py           # 2D Spatial CNN blender
│   ├── temporal_encoder.py         # ConvGRU spatio-temporal recurrent cell
│   ├── spatiotemporal_gating.py    # Spatio-Temporal CNN blender
│   ├── emos.py                     # Gaussian EMOS post-processing
│   └── extreme_calibration.py      # Zero-Inflated Gamma & GEV calibrator
├── pipeline/
│   └── forecast_pipeline.py        # Central operational pipeline orchestrator
├── reports/
│   ├── final_research_report.md    # Comprehensive 25-section research report
│   ├── final_model_comparison.csv  # Audited benchmark comparison results
│   ├── final_probability_metrics.csv # Extreme contingency & calibration metrics
│   └── provenance_manifest.json    # Dataset and model provenance metadata
├── tests/                          # 264 automated unit & integration tests
├── requirements.txt
└── README.md
```

---

## 9. Installation & Setup

### Prerequisites
* Python 3.11 or higher
* Git

### Setup Steps
```bash
# 1. Clone the repository
git clone https://github.com/pallavdeshmukh18/forecast-blending.git
cd forecast-blending

# 2. Create and activate virtual environment
python -m venv .venv

# On Windows (PowerShell):
.venv\Scripts\Activate.ps1
# On Linux / macOS:
source .venv/bin/activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 10. Real-Time Operational Forecasting (Live Mode)

SkyCast provides an operational live-data inference layer built around the validated research forecasting and blending framework.

### 10.1 Live Data Provider
* **Provider**: Open-Meteo Weather Forecast API (free, open-access, zero credentials required).
* **Endpoint**: `https://api.open-meteo.com/v1/forecast`
* **Data Resolution**: ~0.1° spatial resolution hourly surface analysis & multi-day numerical model forecasts.

### 10.2 Variables & Normalization
All live variables are strictly converted and validated in UTC:
* **2m Temperature**: Normalized to **°C** (supports Kelvin and Fahrenheit conversions).
* **10m Wind Speed**: Normalized to **m/s** (supports km/h, knots, mph, and u/v vector magnitude $\sqrt{u_{10}^2 + v_{10}^2}$).
* **Precipitation**: Ingests hourly precipitation and aggregates rolling 6-hour accumulation windows ($\text{mm} / 6\text{h}$) to match the meteorological evaluation standard.

### 10.3 Forecast Horizons & Supported Lead Times
* Operational Horizons: **+6h, +12h, +24h, +48h, +72h**.

### 10.4 Refresh Frequency & Bounded Caching
* **Periodic Refresh**: Automatically fetched and refreshed every **15 minutes** (configurable via `LIVE_REFRESH_MINUTES`).
* **Disk Cache**: Stored in `ML/data/live_cache/live_payload_YYYYMMDD_HHMMSS.json`. Bounded to the newest 24 files with automatic FIFO pruning. Completely isolated from research benchmark datasets.

### 10.5 Data Freshness & Staleness Detection
Every operational payload records `retrieved_at` in UTC and computes data age in minutes:
* **🟢 LIVE / FRESH**: Data age $< 60$ minutes.
* **🟡 STALE**: Data age between $60$ and $180$ minutes.
* **🔴 EXPIRED / UNAVAILABLE**: Data age $> 180$ minutes.

### 10.6 Operational Fallback Hierarchy
In production environments, weather data availability can vary. SkyCast implements a deterministic, fail-safe fallback hierarchy:
```text
LIVE OPERATIONAL DATA FETCH
       ↓ (if successful)
Adaptive SkyCast Blending & EMOS Calibration
       ↓ (if provider fails / times out)
Latest Fresh Local Disk Cache Hit
       ↓ (if fresh cache unavailable)
Stale Cache Fallback (Clearly labeled as STALE in UI)
       ↓ (if cache missing or expired)
Explicit Offline / Unavailable Result State (Never substitutes historical benchmark data)
```

### 10.7 Research Mode vs. Live Mode (Strict Separation)
| Aspect | Research Benchmark Mode | Live Operational Mode |
| :--- | :--- | :--- |
| **Data Source** | WeatherBench 2 (ECMWF HRES, GraphCast v2, ERA5) | Open-Meteo Operational Forecast & Analysis |
| **Time Period** | Historical 2020–2023 Multi-Year Split | Real-time / Latest UTC Timestamps |
| **Verification** | Reanalysis verification with ERA5 Ground Truth | Real-time observation updates |
| **Model Scope** | 14 Research Blending Formulations | Adaptive Blending & EMOS Calibrated Ensembles |
| **Scientific Role** | Statistically audited multi-year benchmark | Operational deployment inference gateway |

### 10.8 Critical Scientific Disclaimers
> **IMPORTANT NOTE**:
> The WeatherBench 2 benchmark is a historical scientific evaluation and is not itself a real-time data feed.
> 
> Operational live forecasts are not automatically equivalent to the research benchmark and should not be interpreted as independently validated operational forecasting skill.
> 
> GraphCast v2 and ECMWF HRES research models evaluated in the scientific benchmark require WeatherBench 2 specific multi-level spatial grids and are not claimed to run in the unauthenticated single-point live feed.

### 10.9 How to Start Live Mode
```bash
# 1. Start FastAPI backend with live operational router enabled
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# 2. Run live smoke test
python ML/scripts/live_smoke_test.py

# 3. Start React Frontend Dashboard
cd frontend
npm run dev
```

---

## 11. Quickstart & Execution Guide

### 1. Run Complete Test Suite (264 Tests)
```bash
python -m pytest Backend/tests
```

### 2. Run Final Multi-Model Benchmark & Generate Reports
```bash
python ML/scripts/run_phase6_final_experiment.py
```

### 3. Launch Operational FastAPI Backend
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```
API Documentation: `http://localhost:8000/docs`

### 4. Launch React Frontend Dashboard
```bash
cd frontend
npm run dev
```
Dashboard URL: `http://localhost:5173`

---

## 12. FastAPI Endpoints & API Reference

### 12.1 Live Operational Endpoints
| Endpoint | Method | Description |
| :--- | :---: | :--- |
| `/api/live/status` | `GET` | Operational connection status, active provider, data freshness, and data age |
| `/api/live/current` | `GET` | Current surface meteorological conditions (Temp, Wind, Rain, Humidity, Pressure) |
| `/api/live/forecast` | `GET` | Multi-horizon operational forecast (+6h to +72h) with adaptive blending & EMOS 90% CI |
| `/api/live/alerts` | `GET` | Active extreme weather alerts (Heavy rain $\ge 15\text{mm}$, Extreme rain $\ge 35\text{mm}$, Gale wind $\ge 17.2\text{m/s}$, Heatwave $\ge 40^\circ\text{C}$) |
| `/api/live/disagreement`| `GET` | Mean $|\text{ECMWF IFS} - \text{GFS}|$ over the next 24 h for temperature, wind and rainfall (Open-Meteo multi-model); falls back to "unavailable" diagnostics if the provider cannot be reached. GraphCast is not part of the live feed |
| `/api/live/wind-grid` | `GET` | 8×8 regular grid over 6–36°N, 67–98°E with wind components ($u_{10}, v_{10}$), temperature, precipitation, humidity, MSL pressure, dew point and cloud cover. Query: `hour` (0–72, forecast hour ahead), `include_disagreement` (adds $|\text{ECMWF IFS} - \text{GFS}|$ per cell), `force_refresh`. Returns `503` if the provider is unreachable (no synthetic fallback) |
| `/api/live/metadata` | `GET` | Live layer configuration, variables, horizons, and scientific boundaries |

### 12.2 Research & Evaluation Endpoints
| Endpoint | Method | Description |
| :--- | :---: | :--- |
| `/health` | `GET` | Service status and version check |
| `/config` | `GET` | System configuration parameters and spatial/temporal definitions |
| `/synthetic/generate` | `POST` | Deterministic synthetic observations and forecasts |
| `/blend/baseline` | `POST` | Equal-weight multi-model blending |
| `/blending/weights` | `GET` | Dynamically computed model blending weights |
| `/blending/forecast`| `POST` | Skill-weighted adaptive blended forecast |
| `/blending/comparison`| `GET` | Walk-forward validation comparison results |
| `/regimes/classify` | `POST` | Rule-based and KMeans weather regime classification |
| `/evaluation/phase4-benchmark` | `GET` | Multi-model benchmark across all candidate blenders |
| `/spatial/grid` | `GET` | Standardized 2D spatial grid metadata |
| `/spatial/disagreement` | `GET` | 2D spatial model disagreement fields |

---

## 13. Automated Testing & Validation

The test suite contains **264 automated tests** covering 100% of data pipelines, spatial models, temporal gating, calibration, live ingestion, staleness detection, and backend APIs:

```bash
python -m pytest Backend/tests -v
```

### Coverage Highlights:
* `test_live_operational_pipeline.py`: Real-time ingestion, UTC parsing, physical bounds validation, bounded disk cache pruning, staleness evaluation, extreme alert detection, fallback hierarchy, and research isolation.
* `test_real_benchmark_execution.py` & `test_phase6_final_results.py`: Bit-for-bit synthetic baseline verification (`54cb659a...`) and audited WeatherBench 2 metrics.
* `test_real_pipeline.py`: Provider-neutral data ingestion, NetCDF, GRIB2, Parquet, and CSV readers.
* `test_extreme_calibration.py`: Gamma EMOS, Zero-Inflated Gamma calibration, GEV tail fitting, sample size safeguards.
* `test_temporal_encoder.py` & `test_spatiotemporal_gating.py`: ConvGRU recurrent memory and convex spatial weights $W_m \ge 0, \sum W_m = 1$.
* `test_final_leakage.py`: Temporal isolation tests verifying historical weights remain invariant to future test observation perturbations.
* `test_robustness.py`: Missing model renormalization, corrupted model downweighting, missing observation safety.
* `test_pipeline_orchestrator.py`: End-to-end `ForecastPipelineOrchestrator` execution.
* `test_api.py` & `test_api_final.py`: Full endpoint contract testing for FastAPI service.

---

## 14. Interactive Operational Portal & 3D WebGIS Workspace

The **Weather Map** tab of the dashboard (`frontend/src/components/dashboard/WeatherMapWorkspace.jsx`) is a MapLibre GL workspace in which every control is bound to map state or a backend call.

### 14.1 Map Controls
| Control | Behaviour |
| :--- | :--- |
| **3D** | Toggles camera pitch between $0^\circ$ (flat 2D) and $55^\circ$ with a $-15^\circ$ bearing (1 s eased transition). This is a perspective tilt of the raster basemap; no terrain elevation model is loaded |
| **+ / −** | Zoom in / out |
| **Locate** | Browser geolocation → flies to the position at zoom 7 and opens a popup with the live `/api/live/forecast` result for those coordinates |
| **particles** | Toggles the wind streamline canvas (active on the *Wind* layer) |
| **Overlay slider** | Sets heatmap opacity, $0 \le \alpha \le 0.75$ |
| **km/h · m/s · knots** | Converts wind legend ticks, popup values and live-forecast wind; $1\ \text{m/s} = 3.6\ \text{km/h} = 1.94384\ \text{kn}$ |
| **Settings (gear)** | Shows / hides city labels |
| **Cycle picker** | `00/06/12/18 UTC` – shifts timeline valid times and the live-grid hour (see limitations) |
| **Lead pills `+24h/+48h/+72h`** | Select the forecast horizon and move the timeline |
| **Timeline** | Day/hour ticks over $t+0 \dots t+72$ h in 12 h steps; **Play** advances one step per second. Each step re-requests the live grid for that hour and re-runs the pipeline for the nearest archive lead |

### 14.2 Layers
Wind, Rainfall (mm / 6 h), Temperature (°C), Uncertainty (ensemble spread), Model weights, Extremes (pulsing markers where the live grid breaches heavy rain $\ge 15$ mm, gale $\ge 17.2$ m/s or heat $\ge 40^\circ$C) and Run change ($F_{\text{lead}} - F_{\text{previous lead}}$). **More layers** adds Relative Humidity, Mean Sea-Level Pressure, Dew Point and Cloud Cover (a model-derived infrared proxy, **not** satellite imagery) from the live grid. Switching layers swaps the heatmap colour ramp, weight stops, radius and legend; the layer rail scrolls inside its own container so it never overlaps the bottom-right control dock.

### 14.3 Windy-Style Particle Vector Flow
`frontend/src/windField.js` implements an HTML5 Canvas particle engine:

* The live grid supplies $u_{10} = -s\sin\theta$, $v_{10} = -s\cos\theta$ (with $\theta$ the direction the wind blows *from*), bilinearly interpolated onto a pixel-space lookup that is rebuilt whenever the view changes.
* Speed $= \sqrt{u_{10}^2 + v_{10}^2}$; particles advect each frame, live for 50–100 frames and respawn, with fading trails (`destination-out` compositing).
* Colour by speed: $<5$ m/s `#00f2fe`, $5$–$15$ m/s `#4facfe`, $15$–$25$ m/s `#f6d365`, $>25$ m/s `#ff0844`.
* The landing-page map (`InteractiveMapSection.jsx`, Leaflet + Esri dark canvas) uses the same engine and can also switch to precipitation, temperature and |ECMWF − GFS| disagreement fields.
* If the backend is unreachable the workspace falls back to a procedural decorative flow; it is not data-driven in that state.

### 14.4 Blending Methodology Switcher
The weight bars come from `GET /blending/weights`; the map field comes from `POST /forecast/run`.

| Button | Pipeline method | Weights shown | Formulation |
| :--- | :--- | :--- | :--- |
| **Regime blend** | `regime_blender` | `neural_gating` | Regime-conditioned mixture of experts. On the synthetic Phase-4 evaluation set (rainfall, +24 h) it gives ≈ 73 % GraphCast v2, 20 % ECMWF HRES, 8 % GFS |
| **Inverse variance** | `inverse_variance` | `inverse_variance` | $w_i = \dfrac{1/\sigma_i^2}{\sum_j 1/\sigma_j^2}$ |
| **Equal weight** | `equal_weight` | uniform | $F = \frac{1}{N}\sum_i F_i$ |
| **More… (modal)** | see below | see below | Advanced research blenders |

The **Advanced blenders** dialog lists Spatial CNN Encoder, Spatio-Temporal ConvGRU, Constrained Simplex Ridge, Gaussian EMOS and Zero-Inflated Gamma Calibration. The forecast pipeline only implements equal-weight, inverse-variance, inverse-MAE and regime blending, so these five run as the **regime blender**; the telemetry pill reports the effective method (`ran as regime_blender`). Their weight bars use the closest fitted blender (`neural_gating`, `ridge_constrained` or `regime_inverse_variance`). Weights quoted above are properties of the synthetic evaluation data and are not live-skill estimates.

### 14.5 Operational Menu Drawer
The top-right **Menu** button opens a slide-over drawer:

* **Basemap** – OpenStreetMap Light, Esri Dark Canvas or Esri Satellite Imagery (three raster layers toggled by visibility, so data layers persist).
* **System health** – provider, freshness (`fresh` < 60 min, `stale` 60–180 min), data age, provider latency (ms), active grid nodes (e.g. 1,053) and engine version, read from `/api/live/status` and `/api/live/metadata`.
* **Export** – downloads a JSON file with the pipeline forecast, live grid, blend weights, run settings and the selected location's live forecast. A second action opens the FastAPI Swagger docs.
* **Display preferences** – city labels and hazard markers.

The landing page and Overview tab also show a status badge (🟢 LIVE / FRESH, 🟡 STALE, 🔴 OFFLINE / CACHE FALLBACK) with a UTC timestamp and a 5-minute auto-refresh countdown.

### 14.6 Limitations
* The pipeline map layers use the synthetic forecast archive, whose lead times are $+6, +12, +24, +48$ h. Other timeline steps snap to the nearest of these; beyond $+48$ h the pill shows `archive max +48h`. The live grid, by contrast, is real Open-Meteo data for the requested hour.
* The cycle picker does not select archived cycles – the archive has no per-cycle runs.
* *Model weights* is a uniform tint (weights are global per run, not spatial) and *Run change* compares leads, not consecutive runs.
* One pipeline run takes roughly 10–20 s; results are cached per variable / lead / method for the session.
* The backend CORS allow-list covers frontend origins on ports 3000 and 5173.
