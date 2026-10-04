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
│   ├── final_project_report.md     # Comprehensive 22-section research report
│   ├── final_model_comparison.csv  # 14-model benchmark results
│   ├── final_probability_metrics.csv # Extreme contingency & calibration metrics
│   ├── data_provenance.json        # Dataset provenance metadata
│   └── data_quality.json           # Data quality control report
├── scripts/
│   ├── run_phase6_final_experiment.py # Final 14-model benchmark runner
│   ├── build_verification_dataset.py  # Verification dataset builder
│   └── validate_real_dataset.py       # Dataset QC validator
├── tests/                          # 199 automated unit & integration tests
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

## 10. Quickstart & Execution Guide

### 1. Run Complete Test Suite (199 Tests)
```bash
python -m pytest -q
```

### 2. Run Final Multi-Model Benchmark & Generate Reports
```bash
python scripts/run_phase6_final_experiment.py
```

### 3. Launch Operational FastAPI Backend
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```
API Documentation: `http://localhost:8000/docs`

### 4. Launch Streamlit Verification Dashboard
```bash
streamlit run dashboard/app.py
```
Dashboard URL: `http://localhost:8501`

---

## 11. FastAPI Endpoints & API Reference

| Endpoint | Method | Description |
| :--- | :---: | :--- |
| `/health` | `GET` | Service status and version check |
| `/metadata` | `GET` | Project metadata, available models, variables, lead times, and regimes |
| `/forecast/run` | `POST` | Central end-to-end forecast pipeline returning structured `ForecastResult` |
| `/forecast/blend` | `POST` | Dynamic multi-model blend for ad-hoc forecast arrays |
| `/forecast/probability` | `POST` | Calibrated quantiles, spread, and predictive distributions |
| `/forecast/extremes` | `POST` | Grid-cell extreme exceedance probabilities (Zero-Inflated Gamma EMOS) |
| `/verify` | `POST` | Forecast verification and paired error metric evaluation |
| `/models` | `GET` | Specifications of available NWP and AI members |
| `/models/weights` | `GET` | Active dynamic model weights |
| `/regimes` | `GET` | Weather regime classification definitions |
| `/diagnostics` | `GET` | Quality control reports and pipeline health |
| `/metrics` | `GET` | Verification error scores across models |

---

## 12. Automated Testing & Validation

The test suite contains **199 automated tests** covering 100% of data pipelines, spatial models, temporal gating, calibration, and backend APIs:

```bash
python -m pytest -v
```

### Coverage Highlights:
* `test_real_pipeline.py`: Provider-neutral data ingestion, NetCDF, GRIB2, Parquet, and CSV readers.
* `test_extreme_calibration.py`: Gamma EMOS, Zero-Inflated Gamma calibration, GEV tail fitting, sample size safeguards.
* `test_temporal_encoder.py` & `test_spatiotemporal_gating.py`: ConvGRU recurrent memory and convex spatial weights $W_m \ge 0, \sum W_m = 1$.
* `test_final_leakage.py`: Temporal isolation tests verifying historical weights remain invariant to future test observation perturbations.
* `test_robustness.py`: Missing model renormalization, corrupted model downweighting, missing observation safety.
* `test_pipeline_orchestrator.py`: End-to-end `ForecastPipelineOrchestrator` execution.
* `test_api_final.py`: Full endpoint contract testing for FastAPI service.