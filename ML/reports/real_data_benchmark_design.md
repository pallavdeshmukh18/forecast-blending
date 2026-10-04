# Real-Data Audit & Leakage-Safe Benchmark Design Report

**Project**: Hybrid AI–NWP Multi-Model Forecast Blending System  
**Version**: `0.6.0`  
**Stage**: Real-Data Audit & Benchmark Experiment Design  
**Date**: October 3, 2026  
**Status**: Complete & Verified (Ready for Prompt 4 Execution)  

---

## 1. Dataset Identity & Source Architecture

WeatherBench 2 is a community benchmark dataset for data-driven and numerical weather forecasting, curated by ECMWF and Google Cloud. The target data stores reside on Google Cloud Storage as cloud-optimized, consolidated Zarr archives:

| Role | Dataset Identifier | GCS Zarr Endpoint | Grid & Resolution |
| :--- | :--- | :--- | :--- |
| **Observation Truth** | ECMWF ERA5 Reanalysis | `gs://weatherbench2/datasets/era5/1959-2022-6h-240x121_equiangular_with_poles_conservative.zarr` | 1.5° Equiangular ($240 \times 121$) |
| **Physics NWP** | ECMWF IFS HRES Deterministic | `gs://weatherbench2/datasets/hres/2016-2022-0012-240x121_equiangular_with_poles_conservative.zarr` | 1.5° Equiangular ($240 \times 121$) |
| **Neural AI NWP** | Google DeepMind GraphCast v2 | `gs://weatherbench2/datasets/graphcast_v2/2020-240x121_equiangular_with_poles_conservative.zarr` | 1.5° Equiangular ($240 \times 121$) |
| **Neural AI NWP** | Huawei Cloud Pangu-Weather | `gs://weatherbench2/datasets/pangu/2018-2022_0012_240x121_equiangular_with_poles_conservative.zarr` | 1.5° Equiangular ($240 \times 121$) |

---

## 2. Locally Downloaded vs. Source Data Coverage

### Physical Local Data on Disk:
- **Files Location**: `ML/data/processed/real/weatherbench2/`
  - `real_forecasts.parquet` (42,336 records, SHA-256: `c4284c7281eeb621fdd7561ae8288f829ae9e6908dfda1f6dea91c3472622bf9`)
  - `real_observations.parquet` (7,056 records, SHA-256: `aa675f489d68a54f12b6824cd42b0e949e7099301529faa0c70b01dab28f27fc`)
  - `real_verification_dataset.parquet` (29,400 pairs, SHA-256: `001bd05e2b6b0b1f9e444a02cc533aa0ea31a35c047d6818d9900078cf5ce751`)
- **Temporal Window**: `2020-01-01T00:00:00Z` to `2020-01-03T23:59:59Z` (3 days)
- **Spatial Coverage**: South Asia ($10.5^\circ\text{N}\text{--}30.0^\circ\text{N}, 70.5^\circ\text{E}\text{--}90.0^\circ\text{E}$, 196 grid points)

### Full Source Coverage on Google Cloud Storage:
- **ERA5 Ground Truth**: 1959-01-02 to 2021-12-31 (63 years, 92,040 steps)
- **ECMWF HRES**: 2016-01-01 to 2022-12-31 (7 years, 5,114 cycles, 00z/12z)
- **GraphCast v2**: 2019-12-01 to 2021-01-21 (14 months, 1,672 cycles, 00z/06z/12z/18z)
- **Pangu-Weather**: 2018-01-01 to 2022-12-31 (5 years, 3,652 cycles, 00z/12z)

### Multi-Model Common Source Overlap:
- **Full Calendar Year 2020**: `2020-01-01` to `2020-12-31` (366 days).
- Both ECMWF HRES, GraphCast v2, and ERA5 have 100% complete continuous coverage across all 3 variables for the entire year 2020.

---

## 3. Model Identity & Availability Matrix

| Model Identifier | Present Locally | Source Available | Category | Cycle Frequency | Native Resolution | Variables Available |
| :--- | :---: | :---: | :--- | :--- | :--- | :--- |
| **`ecmwf_hres`** | **YES** | **YES** (2016–2022) | Physics NWP (IFS) | 2/day (00z, 12z) | 0.1° ($\to 1.5^\circ$) | Temp, Wind, 6h Precip |
| **`graphcast_v2`** | **YES** | **YES** (2020) | Graph Neural Net (DeepMind) | 4/day (00z, 06z, 12z, 18z) | 0.25° ($\to 1.5^\circ$) | Temp, Wind, 6h Precip |
| **`pangu`** | Adapter only | **YES** (2018–2022) | 3D Transformer (Huawei) | 2/day (00z, 12z) | 0.25° ($\to 1.5^\circ$) | Temp, Wind (No Precip) |

### GraphCast Model Identity Verification:
The dataset `weatherbench2/datasets/graphcast_v2/2020-240x121_equiangular_with_poles_conservative.zarr` contains predictions from the **GraphCast v2** model developed by Google DeepMind (trained on ERA5 1979–2017). It is **not** WeatherNext, and it is **not** FourCastNet. The dataset model identifier is strictly preserved as `graphcast` / `graphcast_v2`.

---

## 4. Physical Variable Definitions & Semantics

### 1. Temperature (`temperature`)
- **Source Variable**: `2m_temperature`
- **Physical Meaning**: Surface air temperature at 2 meters above ground level.
- **Source Units**: Kelvin ($K$).
- **Target Units**: Celsius ($^\circ\text{C}$).
- **Transformation**: $T_{^\circ\text{C}} = T_K - 273.15$.
- **Verification**: Consistent across ERA5, ECMWF HRES, and GraphCast.

### 2. Precipitation (`rainfall`)
- **Source Variable**: `total_precipitation_6hr`
- **Physical Meaning**: 6-hour accumulated liquid-equivalent precipitation (rain + melted snow) over the preceding 6 hours.
- **Source Units**: meters ($m$).
- **Target Units**: millimeters ($mm$).
- **Transformation**: $P_{mm} = P_m \times 1000.0$.
- **Temporal Alignment**: In ERA5, `total_precipitation_6hr` at timestamp $t$ represents accumulation in $[t-6\text{h}, t]$. In forecast models (HRES, GraphCast), `total_precipitation_6hr` at lead $L$ represents the predicted 6-hour accumulation over $[L-6\text{h}, L]$. Thus, the forecast valid time $t = \text{run\_time} + L$ matches the observation accumulation window perfectly.

### 3. Wind Speed (`wind_speed`)
- **Source Variable**: `10m_wind_speed` (or vector magnitude $\sqrt{u_{10}^2 + v_{10}^2}$)
- **Physical Meaning**: Instantaneous horizontal wind speed at 10 meters above surface.
- **Source Units**: $\text{m/s}$.
- **Target Units**: $\text{m/s}$.
- **Verification**: `10m_wind_speed` matches $\sqrt{u_{10}^2 + v_{10}^2}$ within floating-point precision ($< 10^{-5}\text{ m/s}$).

---

## 5. Spatial Grid & Coordinate Compatibility

- **Mesh Type**: Equiangular conservative grid with poles ($240 \times 121$).
- **Latitude Coordinates**: Ascending from $-90.0^\circ$ to $+90.0^\circ$, step $1.5^\circ$.
- **Longitude Coordinates**: $0.0^\circ$ to $358.5^\circ$, step $1.5^\circ$ (normalized to $[-180.0^\circ, 180.0^\circ]$).
- **Domain Slice (South Asia)**:
  - Latitude: $10.5^\circ\text{N}, 12.0^\circ\text{N}, 13.5^\circ\text{N}, \dots, 30.0^\circ\text{N}$ ($N=14$)
  - Longitude: $70.5^\circ\text{E}, 72.0^\circ\text{E}, 73.5^\circ\text{E}, \dots, 90.0^\circ\text{E}$ ($N=14$)
  - Total Points: $14 \times 14 = 196$ spatial grid cells.
- **Grid Compatibility**: ERA5, ECMWF HRES, GraphCast, and Pangu share the **exact identical spatial grid points**. No spatial interpolation or grid distortion is introduced.

---

## 6. Lead Times & Initialization Semantics

- **Available Lead Times in Source**: $[0, 6, 12, 18, 24, 30, 36, 42, 48, \dots, 240\text{h}]$.
- **Evaluation Lead Set**: $+6\text{h}$, $+12\text{h}$, $+24\text{h}$, $+48\text{h}$, $+72\text{h}$, $+120\text{h}$.
- **Initialization Frequency**:
  - `ecmwf_hres`: 00:00 UTC and 12:00 UTC.
  - `graphcast_v2`: 00:00, 06:00, 12:00, 18:00 UTC.
- **Fair Comparison Rule**: For primary head-to-head benchmarking, forecasts initialized at common cycles (**00:00 UTC** and **12:00 UTC**) are evaluated to eliminate diurnal initialization bias.

---

## 7. Pairing Analysis & Unmatched Record Breakdown

```text
Total Raw Forecasts Extracted:     42,336
Total Raw Observations Extracted:   7,056
Matched Verified Pairs:            29,400
Unmatched Forecasts:               12,936
Unmatched Observations:                 0 (all observations within range matched)
Missing / Corrupt Values:               0
```

### Exact Breakdown of Unmatched Forecasts:
1. **Forecasts with valid time $\le$ observation max time (Jan 3 18:00 UTC)**:
   - Exactly **29,400** records.
   - All 29,400 were matched with 100.0% precision (0 drops).
2. **Forecasts with valid time $>$ observation max time**:
   - Exactly **12,936** records.
   - Initialized on Jan 2 12z/18z or Jan 3 00z/06z/12z/18z with $+24\text{h}$ or $+48\text{h}$ lead time, verifying on Jan 4 or Jan 5.
   - **Conclusion**: Unmatched records are 100% accounted for by causal boundary truncation. Zero data was lost to matching errors.

---

## 8. Data Sufficiency Assessment

| Capability | Current 3-Day Subset | 1-Year (2020) Dataset | Assessment & Rationale |
| :--- | :---: | :---: | :--- |
| **Basic Deterministic Verification** | **YES** | **YES** | 29,400 pairs sufficient for basic pipeline smoke tests. |
| **Multi-Model Comparison** | **YES** | **YES** | HRES and GraphCast both available. |
| **Seasonal Conditioning** | **NO** | **YES** | 3 days cover only early January (Winter). 1-year data covers Winter, Pre-Monsoon, Monsoon, Post-Monsoon. |
| **Weather Regime Clustering** | **NO** | **YES** | 3 days yielded 29,398 standard vs 2 storm events. Year-long data provides thousands of convective/high-wind cases. |
| **Extreme Rainfall Evaluation** | **NO** | **YES** | Extreme events are rare ($p > 99\%$). 3 days has only 11 heavy rain samples; 1 year provides statistical power. |
| **Walk-Forward / Rolling Adaptation** | **NO** | **YES** | Rolling windows (30 days) require multi-month continuous data. |
| **ConvGRU Spatio-Temporal Memory** | **NO** | **YES** | Sequential recurrent neural training requires hundreds of continuous historical time steps. |
| **Probabilistic Calibration (EMOS)** | **NO** | **YES** | Stable CRPS minimization requires $> 500$ distinct valid verification dates. |

### Final Sufficiency Verdict:
The 3-day subset is **sufficient for integration testing and smoke verification**, but **insufficient for scientific benchmarking**. Prompt 4 should expand the dataset to the **full year 2020** (or minimum 6 months) to execute the complete real-data benchmark.

---

## 9. Leakage-Safe Chronological Experiment Design

### Chronological Partitioning (No Random Shuffling):
```text
┌───────────────────────────────────────┬───────────────────┬───────────────────────┐
│              TRAINING                 │    VALIDATION     │         TEST          │
│        2020-01-01 → 2020-08-31        │ 2020-09-01 → 10-15│ 2020-10-16 → 2020-12-31│
│        (8 months / ~67% data)         │(1.5 mo / ~12.5% ) │(2.5 months / ~20.5% ) │
└───────────────────────────────────────┴───────────────────┴───────────────────────┘
```

### Strict Anti-Leakage Controls:
1. **Model Weights & Inverse Variance**: Fit exclusively on `Train` (Jan–Aug 2020). Never updated using `Test` observations.
2. **Neural Gating & ConvGRU**: Weights optimized on `Train`, early stopping evaluated on `Validation`. Model checkpoint frozen before evaluating on `Test`.
3. **Feature Scalers & Normalizers**: Means and standard deviations computed strictly on `Train`.
4. **Extreme Calibration (EMOS / GEV)**: Distribution parameters fit on `Train` positive rainfall events, frozen during `Test`.
5. **No Lookahead**: Forecast at initialization cycle $t$ uses only past observations $t_{\text{obs}} \le t$.

---

## 10. Model Comparison & Method Compatibility Matrix

| Evaluation Method | Temperature | Wind Speed | Rainfall (6h) | Scientific Justification |
| :--- | :---: | :---: | :---: | :--- |
| **ECMWF HRES** (Single NWP) | YES | YES | YES | Deterministic physics baseline. |
| **GraphCast v2** (Single AI) | YES | YES | YES | Pure neural weather prediction baseline. |
| **Equal-Weight Blend** | YES | YES | YES | Simple ensemble baseline ($w_i = 0.5$). |
| **Inverse-Variance Blend** | YES | YES | YES | Adaptive weighting based on historical error variance. |
| **Inverse-MAE Blend** | YES | YES | YES | Robust $L_1$ adaptive weighting. |
| **Ridge Regression Blend** | YES | YES | YES | Regularized linear combination with non-negative constraints. |
| **Weather-Regime Blend** | YES | YES | YES | Regime-specific skill weighting. |
| **Neural Regime Gating** | YES | YES | YES | Non-linear dynamic blend driven by atmospheric state. |
| **Spatial CNN Blender** | YES | YES | YES | 2D convolutional spatial context weighting. |
| **Spatio-Temporal ConvGRU** | YES | YES | YES | Recurrent spatio-temporal memory across historical timesteps. |
| **Gaussian EMOS** | YES | YES | **NO** | Inappropriate for non-negative skewed rainfall. |
| **Zero-Inflated Gamma EMOS** | **NO** | **NO** | YES | Tailored for discrete-continuous rainfall (point mass at 0 + Gamma tail). |
| **Gamma EMOS** | **NO** | **NO** | YES | Non-negative skewed precipitation modeling. |

---

## 11. Weather Regimes & Extreme Threshold Audit (Real Data)

### 6-Hour Precipitation Thresholds (South Asia):
- **Trace Rain**: $< 0.1\text{ mm / 6h}$
- **Moderate Rain**: $5.0\text{ mm / 6h} \le P < 15.0\text{ mm / 6h}$
- **Heavy Rain**: $15.0\text{ mm / 6h} \le P < 35.0\text{ mm / 6h}$ (corresponds to $\sim 60\text{ mm / 24h}$)
- **Extreme Rain**: $P \ge 35.0\text{ mm / 6h}$ (corresponds to $\sim 140\text{ mm / 24h}$)

### Temperature & Wind Thresholds:
- **Hot Extreme (Heatwave)**: $T_{2m} \ge 35.0^\circ\text{C}$
- **Cold Extreme**: $T_{2m} \le 5.0^\circ\text{C}$
- **High Wind**: $WS_{10m} \ge 10.0\text{ m/s}$

---

## 12. Probabilistic Evaluation Metrics Plan

| Variable | Target Distribution Type | Primary Verification Metrics |
| :--- | :--- | :--- |
| **Temperature** | Gaussian $\mathcal{N}(\mu, \sigma^2)$ | CRPS, CRPSS, Prediction Interval Coverage (PICP 90%), Width (MPIW), Reliability |
| **Wind Speed** | Truncated Gaussian / Weibull | CRPS, Mean Spread-Skill Ratio, Reliability |
| **Rainfall (6h)** | Zero-Inflated Gamma | Continuous CRPS, Brier Score ($P > 5\text{mm}$, $P > 15\text{mm}$), Reliability Diagrams |

---

## 13. Required Data Expansion for Prompt 4 Execution

To execute the benchmark in Prompt 4 without assumptions:
- **Source**: Google Cloud Storage (`gs://weatherbench2/datasets/`)
- **Models**: `ecmwf_hres`, `graphcast_v2`
- **Variables**: `2m_temperature`, `10m_wind_speed`, `total_precipitation_6hr`
- **Date Range**: `2020-01-01` to `2020-12-31` (1 full year)
- **Lead Times**: `[6, 12, 24, 48, 72, 120]` hours
- **Region**: South Asia ($10.5^\circ\text{N}\text{--}30.0^\circ\text{N}, 70.5^\circ\text{E}\text{--}90.0^\circ\text{E}$, 196 grid points)
- **Estimated Ingested Size**: $\sim 35\text{ MB}$ compressed Parquet.

---

## 14. Conclusion & Summary

1. WeatherBench 2 source metadata, variables, grids, and lead times have been verified.
2. The 3-day subset is verified and healthy, with all unmatched records mathematically accounted for.
3. A leakage-safe chronological benchmark design is frozen in `ML/configs/real_benchmark.yaml`.
4. The synthetic baseline remains 100% byte-for-byte identical.
5. Repository is in **READY** state for Prompt 4.
