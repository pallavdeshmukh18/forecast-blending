"""
Unsupervised Synoptic Weather Regime Clustering using Standardized KMeans.
Strictly prevents temporal leakage by fitting scalers and cluster centroids only on training data.
Provides silhouette diagnostics, centroid interpretability profiles, and transition metrics.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

from configs import AppConfig, get_config
from preprocessing.weather_regimes import extract_atmospheric_features


DEFAULT_CLUSTER_FEATURES = [
    "temperature",
    "rainfall",
    "wind_speed",
    "diurnal_sin",
    "diurnal_cos",
    "seasonal_sin",
    "seasonal_cos",
]


class KMeansRegimeClassifier:
    """
    Unsupervised KMeans weather regime classifier with strict temporal isolation.
    """

    def __init__(
        self,
        n_clusters: int = 4,
        features: Optional[List[str]] = None,
        random_state: int = 42,
        config: Optional[AppConfig] = None,
    ):
        self.n_clusters = n_clusters
        self.features = features or DEFAULT_CLUSTER_FEATURES
        self.random_state = random_state
        self.config = config or get_config()

        self.scaler = StandardScaler()
        self.kmeans: Optional[KMeans] = None
        self.is_fitted: bool = False
        self.active_features: List[str] = []
        self.diagnostics: Dict[str, Any] = {}
        self.cluster_profiles: Dict[int, Dict[str, Any]] = {}
        self.cluster_names: Dict[int, str] = {}

    def _prepare_feature_matrix(self, df: pd.DataFrame, is_training: bool = False) -> np.ndarray:
        """Extract and clean feature matrix matching active features."""
        feat_df = extract_atmospheric_features(df, config=self.config)

        if is_training:
            # Determine available numerical features
            self.active_features = [f for f in self.features if f in feat_df.columns]
            if not self.active_features:
                # Fallback to any present numerical columns
                numeric_candidates = ["temperature", "rainfall", "wind_speed", "observed_value", "forecast_value"]
                self.active_features = [c for c in numeric_candidates if c in feat_df.columns]

        if not self.active_features:
            raise ValueError("No valid atmospheric features found for regime clustering.")

        X = feat_df[self.active_features].copy()
        # Fill missing with median
        for col in self.active_features:
            if X[col].isna().any():
                fill_val = X[col].median() if not is_training else 0.0
                X[col] = X[col].fillna(fill_val if pd.notna(fill_val) else 0.0)

        return X.values

    def fit(self, training_df: pd.DataFrame) -> "KMeansRegimeClassifier":
        """
        Fit feature scaler and KMeans clustering strictly on historical training data.
        """
        if training_df.empty:
            raise ValueError("Cannot fit KMeansRegimeClassifier on empty training dataset.")

        X_raw = self._prepare_feature_matrix(training_df, is_training=True)
        if len(X_raw) < self.n_clusters:
            # Adjust K if training samples are fewer than requested clusters
            self.n_clusters = max(1, len(X_raw))

        # 1. Fit scaler strictly on training slice
        X_scaled = self.scaler.fit_transform(X_raw)

        # 2. Fit KMeans
        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=10,
        )
        labels = self.kmeans.fit_predict(X_scaled)
        self.is_fitted = True

        # 3. Compute Clustering Diagnostics
        inertia = float(self.kmeans.inertia_)
        sil_score = None
        if len(X_scaled) > self.n_clusters and self.n_clusters > 1:
            try:
                # Subsample if dataset is very large for speed
                sample_size = min(2000, len(X_scaled))
                sil_score = float(
                    silhouette_score(
                        X_scaled,
                        labels,
                        sample_size=sample_size,
                        random_state=self.random_state,
                    )
                )
            except Exception:
                sil_score = 0.0

        unique_labels, counts = np.unique(labels, return_counts=True)
        cluster_sizes = {int(k): int(v) for k, v in zip(unique_labels, counts)}

        self.diagnostics = {
            "n_clusters": self.n_clusters,
            "inertia": round(inertia, 2),
            "silhouette_score": round(sil_score, 4) if sil_score is not None else None,
            "cluster_sizes": cluster_sizes,
            "features": self.active_features,
            "sample_count": len(X_scaled),
        }

        # 4. Generate Centroid Interpretability Profiles
        self._build_cluster_profiles(training_df, labels)

        return self

    def _build_cluster_profiles(self, df: pd.DataFrame, labels: np.ndarray):
        """Compute human-interpretable centroid metrics and qualitative regime names."""
        feat_df = extract_atmospheric_features(df, config=self.config)
        feat_df["cluster"] = labels

        profiles = {}
        names = {}

        for c_id in range(self.n_clusters):
            c_data = feat_df[feat_df["cluster"] == c_id]
            if c_data.empty:
                continue

            mean_temp = float(c_data["temperature"].mean()) if "temperature" in c_data.columns else np.nan
            mean_rain = float(c_data["rainfall"].mean()) if "rainfall" in c_data.columns else np.nan
            mean_wind = float(c_data["wind_speed"].mean()) if "wind_speed" in c_data.columns else np.nan

            # Qualitative naming
            name_parts = []
            if pd.notna(mean_rain) and mean_rain > 10.0:
                name_parts.append("heavy_rain")
            elif pd.notna(mean_rain) and mean_rain > 2.0:
                name_parts.append("convective")

            if pd.notna(mean_wind) and mean_wind > 10.0:
                name_parts.append("high_wind")

            if pd.notna(mean_temp) and mean_temp > 32.0:
                name_parts.append("hot")
            elif pd.notna(mean_temp) and mean_temp < 10.0:
                name_parts.append("cold")

            if not name_parts:
                regime_tag = f"cluster_{c_id}_stable"
            else:
                regime_tag = f"cluster_{c_id}_" + "_".join(name_parts)

            names[c_id] = regime_tag
            profiles[c_id] = {
                "regime_name": regime_tag,
                "mean_temperature": round(mean_temp, 2) if pd.notna(mean_temp) else None,
                "mean_rainfall": round(mean_rain, 2) if pd.notna(mean_rain) else None,
                "mean_wind_speed": round(mean_wind, 2) if pd.notna(mean_wind) else None,
                "size": len(c_data),
                "sample_count": len(c_data),
            }

        self.cluster_profiles = profiles
        self.cluster_names = names

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Assign cluster-based weather regimes to input DataFrame using fitted scaler and KMeans.
        """
        if not self.is_fitted or self.kmeans is None:
            raise RuntimeError("KMeansRegimeClassifier must be fitted before predict() can be called.")

        if df.empty:
            return df.copy()

        X_raw = self._prepare_feature_matrix(df, is_training=False)
        X_scaled = self.scaler.transform(X_raw)
        labels = self.kmeans.predict(X_scaled)

        out_df = df.copy()
        out_df["regime_id"] = labels
        out_df["weather_regime"] = [self.cluster_names.get(int(i), f"cluster_{i}") for i in labels]

        # Distance to centroid as confidence proxy
        distances = self.kmeans.transform(X_scaled)  # (N, K)
        min_dist = np.min(distances, axis=1)
        # Soft confidence: exp(-dist)
        conf = np.exp(-0.5 * min_dist)
        out_df["regime_confidence"] = np.round(conf, 4)

        return out_df

    def get_diagnostics(self) -> Dict[str, Any]:
        """Return diagnostic metrics including inertia, silhouette score, and cluster sizes."""
        return self.diagnostics

    def get_cluster_profiles(self) -> Dict[int, Dict[str, Any]]:
        """Return human-interpretable centroid profiles for each cluster."""
        return self.cluster_profiles


def compute_regime_transition_frequency(df: pd.DataFrame, regime_col: str = "weather_regime") -> float:
    """
    Calculate transition rate (fraction of sequential time steps where the regime changes).
    """
    if df.empty or regime_col not in df.columns or len(df) < 2:
        return 0.0

    regimes = df[regime_col].dropna().values
    if len(regimes) < 2:
        return 0.0

    transitions = np.sum(regimes[1:] != regimes[:-1])
    return float(transitions / (len(regimes) - 1))


def evaluate_cluster_range(
    df: pd.DataFrame,
    k_values: Optional[List[int]] = None,
    features: Optional[List[str]] = None,
    random_state: int = 42,
    config: Optional[AppConfig] = None,
) -> List[Dict[str, Any]]:
    """
    Evaluate multiple cluster counts K in [3, 4, 5, 6] to generate diagnostic metrics
    (inertia, silhouette score, cluster size distribution).
    """
    ks = k_values or [3, 4, 5, 6]
    results = []

    for k in ks:
        clf = KMeansRegimeClassifier(
            n_clusters=k,
            features=features,
            random_state=random_state,
            config=config,
        )
        try:
            clf.fit(df)
            diag = clf.get_diagnostics()
            results.append({
                "k": k,
                "silhouette_score": diag.get("silhouette_score", 0.0),
                "inertia": diag.get("inertia", 0.0),
                "cluster_sizes": diag.get("cluster_sizes", {}),
                "profiles": clf.get_cluster_profiles(),
            })
        except Exception as e:
            results.append({
                "k": k,
                "silhouette_score": None,
                "inertia": None,
                "error": str(e),
            })

    return results


__all__ = [
    "KMeansRegimeClassifier",
    "compute_regime_transition_frequency",
    "evaluate_cluster_range",
]

