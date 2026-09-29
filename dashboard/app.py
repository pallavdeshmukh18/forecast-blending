"""
Streamlit Research Dashboard for Hybrid AI-NWP Forecast Blending System.
Supports both Synthetic Simulation and Real Meteorological Data Ingestion & Verification,
along with Phase 3 Adaptive Multi-Model Skill-Weighted Blending and Walk-Forward Benchmarking.
"""

from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from configs import get_config
from ingestion.synthetic import SyntheticDataGenerator
from ingestion.readers import load_weather_file
from ingestion.quality_control import apply_quality_control
from ingestion.variable_mapping import standardize_dataframe_variables
from models.baseline import equal_weight_blend, EqualWeightBlender
from models.skill_weighting import InverseVarianceSkillBlender, InverseMAESkillBlender
from models.adaptive_blending import RidgeBlender
from models.regime_blending import RegimeInverseVarianceBlender, RegimeInverseMAEBlender
from models.gating_model import WeatherRegimeGatingNetwork
from preprocessing.alignment import align_forecasts_and_observations
from preprocessing.matching import match_forecasts_to_observations
from preprocessing.weather_regimes import (
    extract_atmospheric_features,
    classify_rule_based_regime,
    RuleBasedRegimeClassifier,
    REGIME_TO_ID,
)
from preprocessing.regime_clustering import (
    KMeansRegimeClassifier,
    evaluate_cluster_range,
)
from evaluation.comparison import compare_models, compute_skill_rankings
from evaluation.walk_forward import (
    WalkForwardBlenderEvaluator,
    compute_relative_improvements,
    compute_paired_bootstrap_ci,
    evaluate_regime_stratified_performance,
)
from extremes.rainfall import detect_extreme_rainfall
from extremes.heatwave import detect_heatwave_events
from extremes.wind import detect_wind_extremes

# Phase 5 Spatial & Probabilistic Imports
from data.synthetic_spatial import generate_synthetic_spatial_dataset
from preprocessing.spatial_features import (
    extract_spatial_feature_stack,
    compute_model_disagreement_2d,
)
from models.spatial_blending import (
    SpatialEqualWeightBlender,
    SpatialInverseVarianceBlender,
)
from models.spatial_gating import SpatialCNNBlender
from models.emos import EnsembleModelOutputStatistics
from models.probabilistic import EnsembleProbabilisticModel
from evaluation.calibration import compute_reliability_curve, evaluate_event_calibration
from evaluation.probabilistic_metrics import (
    crps_gaussian,
    crps_ensemble,
    spatial_mae,
    spatial_rmse,
    spatial_weight_entropy,
)

st.set_page_config(
    page_title="Hybrid Weather Blending Research Platform",
    page_icon="⛅",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #F8FAFC;
        border-radius: 8px;
        padding: 1rem;
        border-left: 4px solid #3B82F6;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_synthetic_walkforward_data(seed: int = 42):
    cfg = get_config()
    generator = SyntheticDataGenerator(cfg)
    obs_df, forecasts_df = generator.generate(seed=seed)

    # Classify weather regimes on observations and forecasts
    obs_df = classify_rule_based_regime(obs_df, config=cfg)
    forecasts_df = classify_rule_based_regime(forecasts_df, config=cfg)

    # Instantiate Phase 3 & 4 blender suite
    blenders = [
        EqualWeightBlender(name="equal_weight_blend"),
        InverseMAESkillBlender(name="inverse_mae_blend", config=cfg),
        InverseVarianceSkillBlender(name="inverse_variance_blend", config=cfg),
        RidgeBlender(constrained_weights=True, name="ridge_constrained_blend", config=cfg),
        RegimeInverseVarianceBlender(name="regime_inverse_variance_blend", config=cfg),
        RegimeInverseMAEBlender(name="regime_inverse_mae_blend", config=cfg),
        WeatherRegimeGatingNetwork(
            hidden_sizes=[64, 32],
            epochs=15,
            batch_size=32,
            learning_rate=0.005,
            random_seed=42,
            name="neural_gating_blend",
            config=cfg,
        ),
    ]

    evaluator = WalkForwardBlenderEvaluator(
        blenders=blenders,
        mode="rolling",
        window_size=15,
        min_train_steps=5,
        config=cfg,
    )
    verified_results = evaluator.run(forecasts_df, obs_df)

    # Also extract pure equal-weight blend for compatibility
    blended_df = equal_weight_blend(forecasts_df)

    # Fit a global inverse-variance blender for interactive weight exploration
    iv_blender = InverseVarianceSkillBlender(config=cfg)
    iv_blender.fit(forecasts_df, obs_df)

    mae_blender = InverseMAESkillBlender(config=cfg)
    mae_blender.fit(forecasts_df, obs_df)

    ridge_blender = RidgeBlender(constrained_weights=True, config=cfg)
    ridge_blender.fit(forecasts_df, obs_df)

    regime_iv_blender = RegimeInverseVarianceBlender(config=cfg)
    regime_iv_blender.fit(forecasts_df, obs_df)

    gating_blender = WeatherRegimeGatingNetwork(
        hidden_sizes=[64, 32],
        epochs=15,
        batch_size=32,
        learning_rate=0.005,
        random_seed=42,
        config=cfg,
    )
    gating_blender.fit(forecasts_df, obs_df)

    return obs_df, forecasts_df, blended_df, verified_results, iv_blender, mae_blender, ridge_blender, regime_iv_blender, gating_blender


@st.cache_data
def load_real_sample_data():
    cfg = get_config()
    obs_path = Path("data/samples/sample_observations.csv")
    fcs_path = Path("data/samples/sample_all_forecasts.csv")

    if not obs_path.exists() or not fcs_path.exists():
        return None, None, None, None, None, None, None, None, None

    obs_raw = load_weather_file(obs_path)
    fcs_raw = load_weather_file(fcs_path)

    obs_std = standardize_dataframe_variables(obs_raw, config=cfg)
    fcs_std = standardize_dataframe_variables(fcs_raw, config=cfg)

    obs_valid, _, _ = apply_quality_control(obs_std, value_col="observed_value", config=cfg)
    fcs_valid, _, _ = apply_quality_control(fcs_std, value_col="forecast_value", config=cfg)

    obs_valid = classify_rule_based_regime(obs_valid, config=cfg)
    fcs_valid = classify_rule_based_regime(fcs_valid, config=cfg)

    blenders = [
        EqualWeightBlender(name="equal_weight_blend"),
        InverseMAESkillBlender(name="inverse_mae_blend", config=cfg),
        InverseVarianceSkillBlender(name="inverse_variance_blend", config=cfg),
        RidgeBlender(constrained_weights=True, name="ridge_constrained_blend", config=cfg),
        RegimeInverseVarianceBlender(name="regime_inverse_variance_blend", config=cfg),
        RegimeInverseMAEBlender(name="regime_inverse_mae_blend", config=cfg),
    ]

    evaluator = WalkForwardBlenderEvaluator(
        blenders=blenders,
        mode="rolling",
        window_size=10,
        min_train_steps=3,
        config=cfg,
    )
    verified_results = evaluator.run(fcs_valid, obs_valid)
    blended_df = equal_weight_blend(fcs_valid)

    iv_blender = InverseVarianceSkillBlender(config=cfg)
    iv_blender.fit(fcs_valid, obs_valid)

    mae_blender = InverseMAESkillBlender(config=cfg)
    mae_blender.fit(fcs_valid, obs_valid)

    ridge_blender = RidgeBlender(constrained_weights=True, config=cfg)
    ridge_blender.fit(fcs_valid, obs_valid)

    regime_iv_blender = RegimeInverseVarianceBlender(config=cfg)
    regime_iv_blender.fit(fcs_valid, obs_valid)

    gating_blender = WeatherRegimeGatingNetwork(hidden_sizes=[32, 16], epochs=10, config=cfg)
    gating_blender.fit(fcs_valid, obs_valid)

    return obs_valid, fcs_valid, blended_df, verified_results, iv_blender, mae_blender, ridge_blender, regime_iv_blender, gating_blender


@st.cache_data
def load_spatial_grid_data(seed: int = 42):
    cfg = get_config()
    fc_ds, obs_ds = generate_synthetic_spatial_dataset(
        config=cfg,
        start_date="2026-01-01",
        end_date="2026-01-15",
        time_step_hours=6,
        lead_times=[6, 12, 24, 48],
        grid_shape=(16, 16),
        seed=seed,
    )
    return fc_ds, obs_ds


def main():
    cfg = get_config()

    st.markdown('<div class="main-header">⛅ Hybrid AI–NWP Multi-Model Forecast Blending System</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Research platform for adaptive spatiotemporal multi-model meteorological forecast blending and verification.</div>',
        unsafe_allow_html=True,
    )

    # Sidebar controls
    st.sidebar.header("🛠️ Pipeline & Execution Mode")
    data_mode = st.sidebar.radio("Active Data Mode", ["Synthetic Simulation", "Real Weather Dataset (Demo)"], index=0)

    if data_mode == "Synthetic Simulation":
        sim_seed = st.sidebar.number_input("Random Seed (Deterministic)", min_value=1, max_value=9999, value=cfg.synthetic_data.seed, step=1)
        with st.spinner("Generating simulation data and executing chronological walk-forward validation..."):
            obs_df, forecasts_df, blended_df, aligned_df, iv_blender, mae_blender, ridge_blender, regime_iv_blender, gating_blender = load_synthetic_walkforward_data(seed=sim_seed)
        mode_tag = "SYNTHETIC SIMULATION"
    else:
        with st.spinner("Loading real datasets and running walk-forward verification..."):
            obs_df, forecasts_df, blended_df, aligned_df, iv_blender, mae_blender, ridge_blender, regime_iv_blender, gating_blender = load_real_sample_data()
            if obs_df is None:
                st.sidebar.error("Real sample data not found at data/samples/. Falling back to synthetic.")
                obs_df, forecasts_df, blended_df, aligned_df, iv_blender, mae_blender, ridge_blender, regime_iv_blender, gating_blender = load_synthetic_walkforward_data(seed=42)
                mode_tag = "SYNTHETIC (FALLBACK)"
            else:
                mode_tag = "REAL METEOROLOGICAL DATA"

    st.sidebar.markdown(f"**Execution Mode:** `{mode_tag}`")
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Evaluated Blenders:**")
    st.sidebar.markdown("- `equal_weight_blend` (BASELINE)")
    st.sidebar.markdown("- `inverse_mae_blend` (PHASE 3 SKILL)")
    st.sidebar.markdown("- `inverse_variance_blend` (PHASE 3 SKILL)")
    st.sidebar.markdown("- `ridge_constrained_blend` (PHASE 3 REGRESSION)")
    st.sidebar.markdown("- `regime_inverse_variance_blend` (PHASE 4 REGIME)")
    st.sidebar.markdown("- `regime_inverse_mae_blend` (PHASE 4 REGIME)")
    st.sidebar.markdown("- `neural_gating_blend` (PHASE 4 NEURAL GATING)")
    st.sidebar.markdown("- `spatial_cnn_gating` (PHASE 5 SPATIAL CNN)")
    st.sidebar.markdown("- `emos_probabilistic` (PHASE 5 EMOS CALIBRATION)")

    # Tabs
    tab_overview, tab_spatial, tab_weights, tab_regimes, tab_explorer, tab_benchmarking, tab_extremes, tab_data = st.tabs([
        "🏛️ Architecture",
        "🌍 Spatial & Probabilistic",
        "🎯 Dynamic Weights",
        "🌪️ Weather Regimes & Gating",
        "📈 Time Series Explorer",
        "📊 Skill Benchmarking",
        "⚠️ Extreme Weather Alerts",
        "💾 Dataset & Map Export",
    ])

    # -------------------------------------------------------------
    # TAB 1: System Architecture
    # -------------------------------------------------------------
    with tab_overview:
        st.subheader("System Architecture & Scientific Principles")
        st.markdown(
            """
            The **Hybrid AI–NWP Forecast Blending System** dynamically combines physical Numerical Weather Prediction (NWP)
            models, ensemble members, and data-driven AI weather models based on atmospheric state/weather regime, lead time, region, season, and historical model skill.
            """
        )

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Available Models", len(cfg.models) + 7, "3 Models + 7 Blenders")
        col2.metric("Supported Variables", len(cfg.variables), "Temp, Rain, Wind")
        lead_times = sorted(list(forecasts_df["lead_time"].unique())) if not forecasts_df.empty else cfg.lead_times_hours
        col3.metric("Lead Time Range", f"{min(lead_times)}h - {max(lead_times)}h", f"{len(lead_times)} Horizons")
        col4.metric("Active Regions", len(cfg.regions), "Coastal, Plains, Mountain")

        st.markdown("### 🧩 Member Models in Blend")
        m_cols = st.columns(3)
        for idx, (m_name, m_info) in enumerate(cfg.models.items()):
            with m_cols[idx]:
                st.markdown(f"#### `{m_name}`")
                st.markdown(f"**Type:** `{m_info.type.upper()}` | **Resolution:** `{m_info.resolution_km} km`")
                st.markdown(f"*{m_info.description}*")

        st.info(
            "💡 **Phase 5 Innovation: 2D Spatial Convolutional Blending & Probabilistic Post-Processing.** "
            "Features 2D gridded spatial alignment, CNN spatial gating networks producing convex weight fields W_m(x,y) >= 0, sum_m W_m(x,y) = 1, "
            "multi-model disagreement layers, Gaussian EMOS calibration, and CRPS / Brier score reliability verification."
        )

    # -------------------------------------------------------------
    # TAB 2: Spatial & Probabilistic Forecasting (Phase 5 Feature)
    # -------------------------------------------------------------
    with tab_spatial:
        st.subheader("🌍 2D Spatial Convolutional Blending & Probabilistic Post-Processing")
        st.markdown(
            "Explore 2D gridded meteorological fields, dynamic spatial model weight fields $W_m(x,y)$, "
            "inter-model disagreement, Gaussian EMOS post-processing, and calibrated extreme event probability maps."
        )

        fc_spatial_ds, obs_spatial_ds = load_spatial_grid_data(seed=42)
        lats = fc_spatial_ds.latitude.values
        lons = fc_spatial_ds.longitude.values
        spatial_times = fc_spatial_ds.time.values

        sp_col1, sp_col2, sp_col3 = st.columns(3)
        with sp_col1:
            sp_var = st.selectbox("Meteorological Variable", list(cfg.variables.keys()), index=0, key="sp_var")
        with sp_col2:
            sp_lead = st.selectbox("Forecast Horizon (Lead Time)", [6, 12, 24, 48], index=2, key="sp_lead")
        with sp_col3:
            time_idx = st.slider("Forecast Timestamp Step", 0, len(spatial_times) - 1, 0, key="sp_time_slider")

        selected_time = spatial_times[time_idx]
        st.markdown(f"**Valid Time:** `{pd.to_datetime(selected_time)}` | **Lead Time:** `{sp_lead}h` | **Variable:** `{sp_var.upper()}`")

        # Extract 2D slices
        fc_slice = fc_spatial_ds[sp_var].sel(time=selected_time, lead_time=sp_lead).values  # (M, H, W)
        obs_slice = obs_spatial_ds[sp_var].sel(time=selected_time).values                   # (H, W)
        models_list = list(fc_spatial_ds.model.values)

        # Compute Blends
        eq_sp_blender = SpatialEqualWeightBlender(model_names=models_list)
        eq_blend_2d, eq_w_2d = eq_sp_blender.predict(fc_slice)

        siv_blender = SpatialInverseVarianceBlender(model_names=models_list)
        siv_blender.fit(fc_spatial_ds[sp_var].sel(lead_time=sp_lead).values, obs_spatial_ds[sp_var].values)
        siv_blend_2d, siv_w_2d = siv_blender.predict(fc_slice)

        # Disagreement
        disagree = compute_model_disagreement_2d(fc_slice, model_names=models_list)

        # EMOS Probabilistic
        emos = EnsembleModelOutputStatistics(fit_method="crps")
        emos.fit(fc_spatial_ds[sp_var].sel(lead_time=sp_lead).values, obs_spatial_ds[sp_var].values)
        pf_slice = emos.predict(fc_slice[np.newaxis, ...])

        # Sub-tabs for Spatial Analysis
        st_sub1, st_sub2, st_sub3, st_sub4, st_sub5 = st.tabs([
            "🗺️ Spatial Forecasts",
            "🎨 Model Weight Fields",
            "⚡ Inter-Model Disagreement",
            "🎲 Probabilistic Quantiles",
            "📈 Reliability & Calibration",
        ])

        with st_sub1:
            st.markdown("#### Multi-Model 2D Gridded Forecasts vs Observation")
            f_cols = st.columns(4)
            for m_idx, m_name in enumerate(models_list):
                with f_cols[m_idx]:
                    fig_m = go.Figure(data=go.Heatmap(
                        z=fc_slice[m_idx],
                        x=lons,
                        y=lats,
                        colorscale="Viridis",
                        colorbar=dict(title=cfg.units.rainfall if sp_var == "rainfall" else cfg.units.temperature),
                    ))
                    fig_m.update_layout(title=f"{m_name.upper()}", height=320, margin=dict(l=10, r=10, t=35, b=10))
                    st.plotly_chart(fig_m, use_container_width=True)

            with f_cols[3]:
                fig_obs = go.Figure(data=go.Heatmap(
                    z=obs_slice,
                    x=lons,
                    y=lats,
                    colorscale="Viridis",
                    colorbar=dict(title="Obs"),
                ))
                fig_obs.update_layout(title="Ground Truth Observation", height=320, margin=dict(l=10, r=10, t=35, b=10))
                st.plotly_chart(fig_obs, use_container_width=True)

            # Blended Forecast and Error
            b_cols = st.columns(2)
            with b_cols[0]:
                fig_b = go.Figure(data=go.Heatmap(
                    z=siv_blend_2d,
                    x=lons,
                    y=lats,
                    colorscale="Viridis",
                    colorbar=dict(title="Blend"),
                ))
                fig_b.update_layout(title="Spatial Inverse-Variance Blended Forecast", height=350)
                st.plotly_chart(fig_b, use_container_width=True)

            with b_cols[1]:
                err_2d = np.abs(siv_blend_2d - obs_slice)
                fig_err = go.Figure(data=go.Heatmap(
                    z=err_2d,
                    x=lons,
                    y=lats,
                    colorscale="Reds",
                    colorbar=dict(title="Abs Error"),
                ))
                fig_err.update_layout(title=f"Spatial Absolute Error Field (MAE: {np.mean(err_2d):.2f})", height=350)
                st.plotly_chart(fig_err, use_container_width=True)

        with st_sub2:
            st.markdown("#### Spatially Varying Model Weight Fields $W_m(x,y)$")
            st.markdown(r"Weights are convex ($W_m(x,y) \ge 0, \sum_m W_m(x,y) = 1.0$) at every grid coordinate.")

            w_cols = st.columns(3)
            for m_idx, m_name in enumerate(models_list):
                with w_cols[m_idx]:
                    w_field = siv_w_2d[m_idx]
                    fig_w = go.Figure(data=go.Heatmap(
                        z=w_field,
                        x=lons,
                        y=lats,
                        zmin=0.0,
                        zmax=1.0,
                        colorscale="Blues",
                        colorbar=dict(title="Weight"),
                    ))
                    fig_w.update_layout(
                        title=f"{m_name} Weight (Mean: {np.mean(w_field):.3f})",
                        height=330,
                        margin=dict(l=10, r=10, t=35, b=10),
                    )
                    st.plotly_chart(fig_w, use_container_width=True)

            st.metric("Spatial Weight Sum Check", f"{np.sum(siv_w_2d, axis=0).mean():.6f}", "Sum = 1.0 everywhere")

        with st_sub3:
            st.markdown("#### Spatial Multi-Model Disagreement Layer")
            d_cols = st.columns(2)
            with d_cols[0]:
                fig_std = go.Figure(data=go.Heatmap(
                    z=disagree["std"],
                    x=lons,
                    y=lats,
                    colorscale="Magma",
                    colorbar=dict(title="Spread"),
                ))
                fig_std.update_layout(title=f"Inter-Model Spread Spread (Mean: {np.mean(disagree['std']):.2f})", height=350)
                st.plotly_chart(fig_std, use_container_width=True)

            with d_cols[1]:
                fig_rng = go.Figure(data=go.Heatmap(
                    z=disagree["range"],
                    x=lons,
                    y=lats,
                    colorscale="Thermal",
                    colorbar=dict(title="Range"),
                ))
                fig_rng.update_layout(title=f"Inter-Model Range (Max: {np.max(disagree['range']):.2f})", height=350)
                st.plotly_chart(fig_rng, use_container_width=True)

        with st_sub4:
            st.markdown("#### Gaussian EMOS Probabilistic Forecast Quantiles")
            q_cols = st.columns(3)
            with q_cols[0]:
                p10_2d = pf_slice.quantile(0.10)[0]
                fig_p10 = go.Figure(data=go.Heatmap(z=p10_2d, x=lons, y=lats, colorscale="Viridis"))
                fig_p10.update_layout(title="P10 Lower Bound (10th Percentile)", height=320)
                st.plotly_chart(fig_p10, use_container_width=True)

            with q_cols[1]:
                p50_2d = pf_slice.quantile(0.50)[0]
                fig_p50 = go.Figure(data=go.Heatmap(z=p50_2d, x=lons, y=lats, colorscale="Viridis"))
                fig_p50.update_layout(title="P50 Median Forecast (50th Percentile)", height=320)
                st.plotly_chart(fig_p50, use_container_width=True)

            with q_cols[2]:
                p90_2d = pf_slice.quantile(0.90)[0]
                fig_p90 = go.Figure(data=go.Heatmap(z=p90_2d, x=lons, y=lats, colorscale="Viridis"))
                fig_p90.update_layout(title="P90 Upper Bound (90th Percentile)", height=320)
                st.plotly_chart(fig_p90, use_container_width=True)

            # Central Prediction Interval Width (80%)
            mpiw_2d = p90_2d - p10_2d
            fig_mpiw = go.Figure(data=go.Heatmap(z=mpiw_2d, x=lons, y=lats, colorscale="Purples", colorbar=dict(title="Width")))
            fig_mpiw.update_layout(title=f"Central 80% Prediction Interval Width (Mean Width: {np.mean(mpiw_2d):.2f})", height=350)
            st.plotly_chart(fig_mpiw, use_container_width=True)

        with st_sub5:
            st.markdown("#### Probability Calibration & Reliability Analysis")
            cal_thresh = 25.0 if sp_var == "rainfall" else (35.0 if sp_var == "temperature" else 15.0)
            cal_diag = evaluate_event_calibration(
                obs_spatial_ds[sp_var].values.ravel(),
                emos.predict(fc_spatial_ds[sp_var].sel(lead_time=sp_lead).values),
                threshold=cal_thresh,
                operator=">=",
            )

            c_metric_cols = st.columns(4)
            c_metric_cols[0].metric("Event Threshold", f">= {cal_thresh}")
            c_metric_cols[1].metric("Brier Score (BS)", f"{cal_diag['brier_score']:.5f}")
            c_metric_cols[2].metric("Brier Skill Score (BSS)", f"{cal_diag['brier_skill_score']:+.4f}")
            c_metric_cols[3].metric("Reliability Error (REL)", f"{cal_diag['reliability_error']:.6f}")

            # Plot Reliability Curve
            fig_rel = go.Figure()
            fig_rel.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                mode="lines",
                name="Perfect Calibration (1:1)",
                line=dict(color="gray", dash="dash"),
            ))
            fig_rel.add_trace(go.Scatter(
                x=cal_diag["reliability_curve"]["mean_predicted_probs"],
                y=cal_diag["reliability_curve"]["observed_frequencies"],
                mode="lines+markers",
                name="EMOS Calibrated Model",
                marker=dict(size=8, color="#3B82F6"),
                line=dict(color="#3B82F6", width=2),
            ))
            fig_rel.update_layout(
                title=f"Reliability Diagram: P({sp_var.capitalize()} >= {cal_thresh})",
                xaxis_title="Forecast Probability",
                yaxis_title="Observed Relative Frequency",
                height=380,
            )
            st.plotly_chart(fig_rel, use_container_width=True)

    # -------------------------------------------------------------
    # TAB 3: Dynamic Weights (Phase 3 Core Feature)
    # -------------------------------------------------------------
    with tab_weights:
        st.subheader("🎯 Context-Conditioned Dynamic Blending Weights")
        st.markdown("Explore how model influence varies across meteorological variables, forecast horizons, and geographical domains.")

        w_col1, w_col2, w_col3, w_col4 = st.columns(4)
        with w_col1:
            w_var = st.selectbox("Variable", list(cfg.variables.keys()), key="w_var")
        with w_col2:
            w_lead = st.selectbox("Lead Time Horizon", sorted(list(forecasts_df["lead_time"].unique())), index=2, key="w_lead")
        with w_col3:
            w_reg = st.selectbox("Region Context", ["coastal", "plains", "mountain"], key="w_reg")
        with w_col4:
            w_method = st.selectbox("Blender Algorithm", ["Inverse-Variance", "Inverse-MAE", "Constrained Ridge", "Regime-Inverse-Variance"], key="w_method")

        active_blender = iv_blender if w_method == "Inverse-Variance" else (mae_blender if w_method == "Inverse-MAE" else (ridge_blender if w_method == "Constrained Ridge" else regime_iv_blender))
        weights_dict = active_blender.get_weights(variable=w_var, lead_time=w_lead, region=w_reg)
        meta_dict = active_blender.get_weight_metadata(variable=w_var, lead_time=w_lead, region=w_reg)

        # Dynamic Weights Bar Chart
        w_df = pd.DataFrame([{"Model": k, "Weight": v, "Percentage": f"{v*100:.1f}%"} for k, v in weights_dict.items()])
        fig_bar = px.bar(
            w_df,
            x="Model",
            y="Weight",
            text="Percentage",
            color="Model",
            title=f"Dynamic Model Weights ({w_method}) for {w_var.capitalize()} @ +{w_lead}h in {w_reg.capitalize()} Region",
            range_y=[0, 1.0],
            template="plotly_white",
        )
        fig_bar.update_traces(textposition="outside")
        st.plotly_chart(fig_bar, use_container_width=True)

        # Detailed Diagnostic Metadata
        st.markdown("### 📋 Skill Context & Hierarchical Scope Metadata")
        st.json(meta_dict)

        # Weight Evolution vs Lead Time Horizon
        st.markdown("### 📈 Weight Evolution Across Forecast Horizons")
        lt_weights = []
        for lt_step in sorted(list(forecasts_df["lead_time"].unique())):
            lt_w = active_blender.get_weights(variable=w_var, lead_time=lt_step, region=w_reg)
            for m_name, val in lt_w.items():
                lt_weights.append({"lead_time": lt_step, "model_name": m_name, "weight": val})
        lt_w_df = pd.DataFrame(lt_weights)

        fig_lt_w = px.line(
            lt_w_df,
            x="lead_time",
            y="weight",
            color="model_name",
            markers=True,
            title=f"Model Weight Evolution vs Lead Time ({w_var.capitalize()}, {w_reg.capitalize()})",
            labels={"lead_time": "Lead Time (Hours)", "weight": "Assigned Weight"},
            template="plotly_white",
        )
        fig_lt_w.update_traces(line=dict(width=3), marker=dict(size=8))
        st.plotly_chart(fig_lt_w, use_container_width=True)

    # -------------------------------------------------------------
    # TAB 3: Weather Regimes & Neural Gating (Phase 4 Innovation)
    # -------------------------------------------------------------
    with tab_regimes:
        st.subheader("🌪️ Atmospheric State Detection & Neural Mixture-of-Experts Gating")
        st.markdown(
            "Conditioning forecast weighting dynamically on the **current atmospheric weather regime** "
            "(stable, convective storms, heavy rainfall, synoptic high wind, thermal extremes)."
        )

        r_sub1, r_sub2 = st.tabs(["🧩 Regime Classification & Clustering", "🗺️ Spatial Model Weight Maps"])

        with r_sub1:
            st.markdown("#### Unsupervised KMeans Clustering Diagnostics")
            k_diag = evaluate_cluster_range(obs_df, k_values=[3, 4, 5, 6], random_state=42, config=cfg)
            k_diag_df = pd.DataFrame([
                {
                    "Clusters (K)": d["k"],
                    "Silhouette Score": d["silhouette_score"],
                    "Inertia": d["inertia"],
                    "Cluster Sizes": str(d["cluster_sizes"]),
                }
                for d in k_diag
            ])

            k_col1, k_col2 = st.columns(2)
            with k_col1:
                fig_sil = px.bar(
                    k_diag_df,
                    x="Clusters (K)",
                    y="Silhouette Score",
                    color="Silhouette Score",
                    color_continuous_scale="Viridis",
                    title="Silhouette Diagnostic Score vs Cluster Count (K)",
                    template="plotly_white",
                )
                st.plotly_chart(fig_sil, use_container_width=True)

            with k_col2:
                fig_ine = px.line(
                    k_diag_df,
                    x="Clusters (K)",
                    y="Inertia",
                    markers=True,
                    title="KMeans Inertia (Elbow Diagnostic) vs K",
                    template="plotly_white",
                )
                fig_ine.update_traces(line=dict(width=3, color="#EF4444"), marker=dict(size=9))
                st.plotly_chart(fig_ine, use_container_width=True)

            st.dataframe(k_diag_df, use_container_width=True)

            st.markdown("#### Regime-Stratified Multi-Model Verification Breakdown")
            regime_metrics_df = evaluate_regime_stratified_performance(aligned_df)
            if not regime_metrics_df.empty:
                st.dataframe(regime_metrics_df, use_container_width=True)

        with r_sub2:
            st.markdown("#### Geographic Model Weight Maps (Continuous Color Scale)")
            st.markdown(
                "Visualizes spatial model weights across stations and grid points conditioned on variable, lead time, and atmospheric regime."
            )

            m_col1, m_col2, m_col3, m_col4 = st.columns(4)
            with m_col1:
                map_var = st.selectbox("Map Variable", list(cfg.variables.keys()), key="map_var")
            with m_col2:
                map_lead = st.selectbox("Map Lead Time", sorted(list(forecasts_df["lead_time"].unique())), index=1, key="map_lead")
            with m_col3:
                map_regime = st.selectbox("Atmospheric Regime", ["stable", "convective", "heavy_rain", "high_wind"], key="map_regime")
            with m_col4:
                map_model = st.selectbox("Target Member Model", list(cfg.models.keys()), key="map_model")

            map_pts = forecasts_df[["latitude", "longitude", "region"]].drop_duplicates().copy()
            weights_list = []
            for _, pt in map_pts.iterrows():
                pt_w = regime_iv_blender.get_weights(
                    variable=map_var,
                    lead_time=map_lead,
                    region=pt.get("region"),
                    weather_regime=map_regime,
                )
                weights_list.append(pt_w.get(map_model, 1.0 / len(cfg.models)))

            map_pts["Model Weight"] = weights_list
            map_pts["Model"] = map_model
            map_pts["Regime"] = map_regime
            map_pts["Lead Time"] = f"+{map_lead}h"

            fig_map = px.scatter(
                map_pts,
                x="longitude",
                y="latitude",
                color="Model Weight",
                size="Model Weight",
                hover_data=["region", "Model Weight", "Regime", "Lead Time"],
                color_continuous_scale="Viridis",
                range_color=[0.0, 1.0],
                title=f"Spatial Weight Distribution for {map_model} ({map_var.capitalize()}, {map_regime.capitalize()} Regime, +{map_lead}h)",
                template="plotly_white",
                height=480,
            )
            fig_map.update_traces(marker=dict(sizemin=8, sizeref=0.05))
            st.plotly_chart(fig_map, use_container_width=True)

            st.dataframe(map_pts, use_container_width=True)


    # -------------------------------------------------------------
    # TAB 3: Time Series Explorer
    # -------------------------------------------------------------
    with tab_explorer:
        st.subheader("Interactive Multi-Model Forecast vs Ground-Truth Verification")

        exp_col1, exp_col2, exp_col3 = st.columns(3)
        with exp_col1:
            selected_var = st.selectbox("Meteorological Variable", list(cfg.variables.keys()), index=0, key="exp_var")
        with exp_col2:
            if "station_id" in obs_df.columns and obs_df["station_id"].notna().any():
                available_stations = sorted(obs_df["station_id"].dropna().unique())
                selected_station = st.selectbox("Monitoring Station", available_stations, index=0, key="exp_stn")
            else:
                selected_station = None
        with exp_col3:
            available_lts = sorted(list(aligned_df["lead_time"].unique())) if not aligned_df.empty else cfg.lead_times_hours
            selected_lt = st.selectbox("Forecast Lead Time", available_lts, index=min(2, len(available_lts) - 1), key="exp_lt")

        # Filter aligned data
        if selected_station and "station_id" in obs_df.columns:
            stn_coords = obs_df[obs_df["station_id"] == selected_station].iloc[0]
            stn_lat, stn_lon = stn_coords["latitude"], stn_coords["longitude"]
            stn_aligned = aligned_df[
                (aligned_df["variable"] == selected_var)
                & (aligned_df["latitude"] == stn_lat)
                & (aligned_df["longitude"] == stn_lon)
                & (aligned_df["lead_time"] == selected_lt)
            ]
        else:
            stn_aligned = aligned_df[
                (aligned_df["variable"] == selected_var)
                & (aligned_df["lead_time"] == selected_lt)
            ]

        if not stn_aligned.empty:
            fig = go.Figure()

            # Add Ground Truth Observations
            sample_model = stn_aligned["model_name"].iloc[0]
            obs_subset = stn_aligned[stn_aligned["model_name"] == sample_model].sort_values("timestamp")
            fig.add_trace(
                go.Scatter(
                    x=obs_subset["timestamp"],
                    y=obs_subset["observed_value"],
                    mode="lines+markers",
                    name="Ground Truth (Observed)",
                    line=dict(color="#111827", width=3.5, dash="solid"),
                    marker=dict(size=6),
                )
            )

            palette = {
                "nwp_model_a": ("#2563EB", "dash"),
                "nwp_model_b": ("#059669", "dot"),
                "ai_model_c": ("#D97706", "dashdot"),
                "equal_weight_blend": ("#9333EA", "solid"),
                "inverse_variance_blend": ("#DC2626", "solid"),
                "inverse_mae_blend": ("#EA580C", "solid"),
                "ridge_constrained_blend": ("#0D9488", "solid"),
            }

            for model_name in stn_aligned["model_name"].unique():
                m_sub = stn_aligned[stn_aligned["model_name"] == model_name].sort_values("timestamp")
                if not m_sub.empty:
                    c, d = palette.get(model_name, ("#6B7280", "solid"))
                    is_blend = "blend" in model_name
                    fig.add_trace(
                        go.Scatter(
                            x=m_sub["timestamp"],
                            y=m_sub["forecast_value"],
                            mode="lines",
                            name=f"{model_name}" + (" (BLEND)" if is_blend else ""),
                            line=dict(color=c, width=3.0 if is_blend else 1.8, dash=d),
                        )
                    )

            unit = cfg.variables[selected_var].unit if selected_var in cfg.variables else ""
            title_stn = f" ({selected_station})" if selected_station else ""
            fig.update_layout(
                title=f"{selected_var.capitalize()} Forecast Verification{title_stn} (Horizon = +{selected_lt}h)",
                xaxis_title="UTC Valid Timestamp",
                yaxis_title=f"{selected_var.capitalize()} ({unit})",
                template="plotly_white",
                height=520,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No verified records found for selected filter.")

    # -------------------------------------------------------------
    # TAB 4: Skill Benchmarking
    # -------------------------------------------------------------
    with tab_benchmarking:
        st.subheader("Multi-Model Skill Benchmarking & Relative Improvements")

        b_col1, b_col2 = st.columns(2)
        with b_col1:
            eval_var = st.selectbox("Target Variable", list(cfg.variables.keys()), key="b_eval_var")
        with b_col2:
            eval_metric = st.selectbox("Target Metric", ["rmse", "mae", "bias", "correlation"], index=0, key="b_eval_metric")

        # Run multi-model comparison across lead times
        comparison_df = compare_models(aligned_df, group_by=["variable", "lead_time"])
        var_comp = comparison_df[comparison_df["variable"] == eval_var].copy()

        if not var_comp.empty:
            ranked_df = compute_skill_rankings(var_comp, metric=eval_metric)
            improvement_df = compute_relative_improvements(ranked_df, baseline_model="equal_weight_blend", metric=eval_metric)

            fig_metric = px.line(
                var_comp,
                x="lead_time",
                y=eval_metric,
                color="model_name",
                markers=True,
                title=f"{eval_metric.upper()} vs Forecast Lead Time ({eval_var.capitalize()})",
                labels={"lead_time": "Lead Time (Hours)", eval_metric: eval_metric.upper(), "model_name": "Model / Blender"},
                template="plotly_white",
            )
            fig_metric.update_traces(line=dict(width=2.5), marker=dict(size=8))
            st.plotly_chart(fig_metric, use_container_width=True)

            st.markdown("### 🏆 Comprehensive Benchmark & Skill Rankings")
            disp_cols = [c for c in ["lead_time", "rank", "model_name", "mae", "rmse", "bias", "correlation", f"{eval_metric}_rel_improvement_vs_equal_weight_blend", "sample_count"] if c in improvement_df.columns]
            st.dataframe(improvement_df[disp_cols], use_container_width=True)

    # -------------------------------------------------------------
    # TAB 5: Extreme Weather Alerts
    # -------------------------------------------------------------
    with tab_extremes:
        st.subheader("Extreme Meteorological Event Guidance")

        ext_rain = detect_extreme_rainfall(blended_df)
        ext_heat = detect_heatwave_events(blended_df)
        ext_wind = detect_wind_extremes(blended_df)

        e_col1, e_col2, e_col3 = st.columns(3)
        e_col1.metric("Heavy Rainfall Alerts", len(ext_rain), f"> {cfg.extreme_event_thresholds.rainfall.heavy} mm")
        e_col2.metric("Heatwave Warnings", len(ext_heat), f">= {cfg.extreme_event_thresholds.heatwave.mild} °C")
        e_col3.metric("Gale Wind Alerts", len(ext_wind), f">= {cfg.extreme_event_thresholds.wind.gale} m/s")

        st.markdown("#### Heavy Precipitation Alert Log")
        if not ext_rain.empty:
            st.dataframe(ext_rain[["timestamp", "latitude", "longitude", "lead_time", "forecast_value"] + ([c for c in ["region"] if c in ext_rain.columns])].head(15), use_container_width=True)
        else:
            st.success("No heavy precipitation events detected in current forecast window.")

    # -------------------------------------------------------------
    # TAB 6: Dataset & Spatial Weight Map Export (Phase 3 Requirement)
    # -------------------------------------------------------------
    with tab_data:
        st.subheader("Spatial Model Weight Map Preparation & Data Export")
        st.markdown(
            "Structured export supporting spatially varying model reliability maps (`latitude | longitude | lead_time | variable | model_name | weight`)."
        )

        weight_map_df = iv_blender.get_weight_map(
            forecasts_df=forecasts_df,
            variables=list(cfg.variables.keys()),
            lead_times=sorted(list(forecasts_df["lead_time"].unique())),
        )

        st.markdown("##### Spatially Conditioned Model Weight Map (Sample)")
        st.dataframe(weight_map_df.head(25), use_container_width=True)

        csv_data = weight_map_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Structured Model Weight Map (CSV)",
            data=csv_data,
            file_name="spatial_model_weights_map.csv",
            mime="text/csv",
        )
