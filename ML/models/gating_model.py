"""
Neural Weather Regime Gating Network using PyTorch Mixture-of-Experts.
Dynamically routes multi-model forecast weights based on atmospheric state features,
synoptic regimes, lead times, and spatiotemporal coordinates.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from configs import AppConfig, get_config
from ingestion.schema import validate_forecast_dataframe
from preprocessing.matching import match_forecasts_to_observations
from preprocessing.weather_regimes import extract_atmospheric_features, classify_rule_based_regime, REGIME_TO_ID
from models.base import BaseBlender


class GatingPyTorchModule(nn.Module):
    """
    PyTorch Feed-Forward Neural Gating Network with Softmax routing weights.
    """

    def __init__(
        self,
        input_dim: int,
        num_models: int,
        hidden_sizes: List[int] = [64, 32],
        dropout: float = 0.10,
        use_residual_head: bool = False,
    ):
        super().__init__()
        self.use_residual_head = use_residual_head
        layers = []
        prev_dim = input_dim

        for h in hidden_sizes:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h

        self.backbone = nn.Sequential(*layers)
        self.weight_head = nn.Linear(prev_dim, num_models)
        self.softmax = nn.Softmax(dim=-1)

        if self.use_residual_head:
            self.residual_head = nn.Linear(prev_dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        feat = self.backbone(x)
        logits = self.weight_head(feat)
        weights = self.softmax(logits)
        residual = self.residual_head(feat) if self.use_residual_head else None
        return weights, residual


class WeatherRegimeGatingNetwork(BaseBlender):
    """
    Atmospheric state and regime-aware neural gating blender.
    Learns dynamic model combination weights directly from observed targets.
    """

    def __init__(
        self,
        hidden_sizes: Optional[List[int]] = None,
        dropout: float = 0.10,
        learning_rate: float = 0.001,
        batch_size: int = 64,
        epochs: int = 100,
        early_stopping_patience: int = 10,
        weight_decay: float = 1e-4,
        use_residual_head: bool = False,
        random_seed: int = 42,
        name: str = "neural_gating_blend",
        config: Optional[AppConfig] = None,
    ):
        super().__init__(name=name)
        self.config = config or get_config()
        ng_cfg = self.config.neural_gating

        self.hidden_sizes = hidden_sizes or ng_cfg.hidden_sizes
        self.dropout = dropout if dropout is not None else ng_cfg.dropout
        self.learning_rate = learning_rate if learning_rate is not None else ng_cfg.learning_rate
        self.batch_size = batch_size if batch_size is not None else ng_cfg.batch_size
        self.epochs = epochs if epochs is not None else ng_cfg.epochs
        self.early_stopping_patience = early_stopping_patience if early_stopping_patience is not None else ng_cfg.early_stopping_patience
        self.weight_decay = weight_decay if weight_decay is not None else ng_cfg.weight_decay
        self.use_residual_head = use_residual_head if use_residual_head is not None else ng_cfg.use_residual_head
        self.random_seed = random_seed if random_seed is not None else ng_cfg.random_seed

        self.scaler = StandardScaler()
        self.model: Optional[GatingPyTorchModule] = None
        self.known_models: List[str] = []
        self.feature_names: List[str] = []
        self.region_list: List[str] = list(self.config.regions.keys())
        self.training_history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}
        self.historical_residual_std: float = 1.0

        # Set PyTorch & NumPy seeds
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)

    @property
    def module(self) -> Optional[GatingPyTorchModule]:
        """Alias for self.model."""
        return self.model

    @module.setter
    def module(self, val: Optional[GatingPyTorchModule]):
        self.model = val

    def _extract_feature_vector(self, df: pd.DataFrame, is_training: bool = False) -> Tuple[np.ndarray, List[str]]:
        """Extract continuous and categorical atmospheric features."""
        feat_df = extract_atmospheric_features(df, config=self.config)

        if "weather_regime" not in feat_df.columns or feat_df["weather_regime"].isna().all():
            feat_df = classify_rule_based_regime(feat_df, config=self.config)

        # 1. Numerical continuous features
        numeric_cols = ["latitude", "longitude", "lead_time", "diurnal_sin", "diurnal_cos", "seasonal_sin", "seasonal_cos"]
        for extra in ["temperature", "rainfall", "wind_speed", "surface_pressure", "relative_humidity"]:
            if extra in feat_df.columns:
                numeric_cols.append(extra)

        if is_training:
            self.feature_names = [c for c in numeric_cols if c in feat_df.columns]

        # Ensure all required features are present in feat_df with default 0.0
        for c in self.feature_names:
            if c not in feat_df.columns:
                feat_df[c] = 0.0

        X_num = feat_df[self.feature_names].fillna(0.0).values

        # 2. One-hot encode region
        region_dummies = np.zeros((len(feat_df), len(self.region_list)))
        if "region" in feat_df.columns:
            for idx, reg in enumerate(feat_df["region"].values):
                if str(reg) in self.region_list:
                    region_dummies[idx, self.region_list.index(str(reg))] = 1.0

        # 3. One-hot encode weather_regime
        n_regimes = len(REGIME_TO_ID)
        regime_dummies = np.zeros((len(feat_df), n_regimes))
        if "weather_regime" in feat_df.columns:
            for idx, regm in enumerate(feat_df["weather_regime"].values):
                r_id = REGIME_TO_ID.get(str(regm), -1)
                if r_id >= 0:
                    regime_dummies[idx, r_id] = 1.0

        X_combined = np.hstack([X_num, region_dummies, regime_dummies])
        return X_combined, self.feature_names

    def fit_verification_data(self, verified_df: pd.DataFrame) -> "WeatherRegimeGatingNetwork":
        """
        Train neural gating network on wide-format multi-model verified pairs.
        Uses chronological 80/20 train/validation split with early stopping.
        """
        if verified_df.empty:
            self.is_fitted = True
            return self

        df = verified_df.copy()
        if "weather_regime" not in df.columns or df["weather_regime"].isna().all():
            df = classify_rule_based_regime(df, config=self.config)

        self.known_models = sorted(list(df["model_name"].dropna().unique()))
        n_models = len(self.known_models)
        if n_models < 1:
            self.is_fitted = True
            return self

        # Pivot to wide format: One row per spatiotemporal forecast with columns for each model
        index_cols = ["timestamp", "latitude", "longitude", "lead_time", "variable"]
        meta_cols = [c for c in ["region", "weather_regime", "observed_value"] if c in df.columns]
        wide_index = index_cols + [c for c in meta_cols if c not in index_cols]

        wide_df = df.pivot_table(
            index=[c for c in wide_index if c in df.columns],
            columns="model_name",
            values="forecast_value",
            aggfunc="first",
        ).reset_index()
        wide_df.columns.name = None

        # Require all model forecasts and observed target
        complete_wide = wide_df.dropna(subset=self.known_models + ["observed_value"]).sort_values("timestamp").reset_index(drop=True)

        if len(complete_wide) < 10:
            # Not enough data for deep network, fallback to base fitting
            self.is_fitted = True
            return self

        # Extract features and targets
        X_raw, _ = self._extract_feature_vector(complete_wide, is_training=True)
        F_models = complete_wide[self.known_models].values  # (N, num_models)
        y_target = complete_wide["observed_value"].values.reshape(-1, 1)  # (N, 1)

        # Chronological 80% train / 20% validation split
        split_idx = max(int(0.80 * len(complete_wide)), len(complete_wide) - 10)
        X_train_raw, X_val_raw = X_raw[:split_idx], X_raw[split_idx:]
        F_train, F_val = F_models[:split_idx], F_models[split_idx:]
        y_train, y_val = y_target[:split_idx], y_target[split_idx:]

        # Fit feature scaler strictly on training split
        X_train = self.scaler.fit_transform(X_train_raw)
        X_val = self.scaler.transform(X_val_raw)

        # Build PyTorch Module
        input_dim = X_train.shape[1]
        self.model = GatingPyTorchModule(
            input_dim=input_dim,
            num_models=n_models,
            hidden_sizes=self.hidden_sizes,
            dropout=self.dropout,
            use_residual_head=self.use_residual_head,
        )

        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        criterion = nn.MSELoss()

        train_dataset = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(F_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32),
        )
        val_dataset = TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(F_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.float32),
        )

        train_loader = DataLoader(train_dataset, batch_size=min(self.batch_size, len(train_dataset)), shuffle=True)

        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0

        # Training Loop
        self.model.train()
        for epoch in range(self.epochs):
            train_epoch_losses = []
            for b_x, b_f, b_y in train_loader:
                optimizer.zero_grad()
                w, res = self.model(b_x)
                # Predicted blend: F_blend = sum(w_i * F_i)
                pred_blend = torch.sum(w * b_f, dim=-1, keepdim=True)
                if res is not None:
                    pred_blend = pred_blend + res

                loss = criterion(pred_blend, b_y)
                loss.backward()
                optimizer.step()
                train_epoch_losses.append(loss.item())

            # Validation evaluation
            self.model.eval()
            with torch.no_grad():
                val_x, val_f, val_y = val_dataset.tensors
                val_w, val_res = self.model(val_x)
                val_pred = torch.sum(val_w * val_f, dim=-1, keepdim=True)
                if val_res is not None:
                    val_pred = val_pred + val_res
                val_loss = criterion(val_pred, val_y).item()

            train_loss = float(np.mean(train_epoch_losses))
            self.training_history["train_loss"].append(train_loss)
            self.training_history["val_loss"].append(val_loss)

            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.early_stopping_patience:
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        self.model.eval()
        # Compute residual std
        with torch.no_grad():
            all_x = torch.tensor(self.scaler.transform(X_raw), dtype=torch.float32)
            all_w, all_res = self.model(all_x)
            all_pred = torch.sum(all_w * torch.tensor(F_models, dtype=torch.float32), dim=-1, keepdim=True)
            if all_res is not None:
                all_pred = all_pred + all_res
            residuals = y_target - all_pred.numpy()
            self.historical_residual_std = float(np.std(residuals)) if len(residuals) > 1 else 1.0

        self.is_fitted = True
        return self

    def fit(self, historical_forecasts: pd.DataFrame, historical_observations: pd.DataFrame) -> "WeatherRegimeGatingNetwork":
        """Align forecasts and observations, then train neural gating network."""
        verified = match_forecasts_to_observations(historical_forecasts, historical_observations, config=self.config)
        return self.fit_verification_data(verified)

    def predict(self, forecasts_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate dynamically routed neural blended forecasts with diagnostic uncertainty.
        """
        df = validate_forecast_dataframe(forecasts_df, allow_nans=True)

        if "weather_regime" not in df.columns or df["weather_regime"].isna().all():
            df = classify_rule_based_regime(df, config=self.config)

        if not self.is_fitted or self.model is None:
            # Fallback to equal weighting if untrained
            from models.baseline import equal_weight_blend
            return equal_weight_blend(df, output_model_name=self.name)

        group_cols = ["timestamp", "latitude", "longitude", "lead_time", "variable"]
        meta_cols = ["run_time", "region", "season", "weather_regime"]
        active_meta = [c for c in meta_cols if c in df.columns]
        all_group_cols = group_cols + active_meta

        records = []
        for keys, grp in df.groupby(all_group_cols, dropna=False, observed=True):
            if not isinstance(keys, tuple):
                keys = (keys,)
            key_dict = dict(zip(all_group_cols, keys))

            valid_grp = grp[grp["forecast_value"].notna()].copy()
            if valid_grp.empty:
                continue

            avail_models = list(valid_grp["model_name"].values)
            avail_values = valid_grp["forecast_value"].values

            # Extract feature vector for single record slice
            slice_df = pd.DataFrame([key_dict])
            X_slice_raw, _ = self._extract_feature_vector(slice_df, is_training=False)
            X_slice = self.scaler.transform(X_slice_raw)

            # Neural Gating Inference
            self.model.eval()
            with torch.no_grad():
                tensor_x = torch.tensor(X_slice, dtype=torch.float32)
                weights_tensor, res_tensor = self.model(tensor_x)
                w_all = weights_tensor.numpy()[0]  # (num_known_models,)
                res_val = float(res_tensor.numpy()[0, 0]) if res_tensor is not None else 0.0

            # Map weights to available models
            model_weights = dict(zip(self.known_models, w_all))
            w_avail = np.array([model_weights.get(m, 1.0 / len(avail_models)) for m in avail_models])

            if np.sum(w_avail) > 0:
                w_avail = w_avail / np.sum(w_avail)
            else:
                w_avail = np.ones(len(avail_models)) / len(avail_models)

            blended_val = float(np.sum(w_avail * avail_values) + res_val)
            spread = float(np.sqrt(np.sum(w_avail * (avail_values - blended_val) ** 2))) if len(avail_values) > 1 else 0.0

            rec = {
                **key_dict,
                "model_name": self.name,
                "forecast_value": round(blended_val, 4),
                "forecast_spread": round(spread, 4),
                "historical_residual_std": round(self.historical_residual_std, 4),
            }
            records.append(rec)

        if not records:
            return pd.DataFrame()

        out_df = pd.DataFrame(records)
        return validate_forecast_dataframe(out_df, allow_nans=False)

    def get_weights(
        self,
        variable: str,
        lead_time: Optional[int] = None,
        region: Optional[str] = None,
        weather_regime: Optional[str] = None,
        available_models: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """Compute neural gating softmax weights for given atmospheric context."""
        models = available_models or self.known_models
        if not models:
            return {}

        if not self.is_fitted or self.model is None:
            return {m: 1.0 / len(models) for m in models}

        slice_dict = {
            "timestamp": datetime(2026, 1, 15, 12, 0),
            "latitude": 25.0,
            "longitude": 75.0,
            "lead_time": lead_time or 24,
            "variable": variable,
            "region": region or "plains",
            "weather_regime": weather_regime or "stable",
        }
        slice_df = pd.DataFrame([slice_dict])
        X_slice_raw, _ = self._extract_feature_vector(slice_df, is_training=False)
        X_slice = self.scaler.transform(X_slice_raw)

        self.model.eval()
        with torch.no_grad():
            w_all = self.model(torch.tensor(X_slice, dtype=torch.float32))[0].numpy()[0]

        all_w_dict = dict(zip(self.known_models, w_all))
        raw_w = np.array([all_w_dict.get(m, 1.0 / len(models)) for m in models])
        if np.sum(raw_w) > 0:
            norm_w = raw_w / np.sum(raw_w)
        else:
            norm_w = np.ones(len(models)) / len(models)

        return dict(zip(models, [float(w) for w in norm_w]))

    def get_weight_metadata(
        self,
        variable: str,
        lead_time: Optional[int] = None,
        region: Optional[str] = None,
        weather_regime: Optional[str] = None,
        available_models: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Return diagnostic neural gating metadata."""
        models = available_models or self.known_models
        weights = self.get_weights(variable, lead_time, region, weather_regime, available_models=models)
        return {
            "blender": self.name,
            "architecture": f"Dense({'-'.join(map(str, self.hidden_sizes))})-Softmax",
            "use_residual_head": self.use_residual_head,
            "variable": variable,
            "lead_time": lead_time,
            "region": region,
            "weather_regime": weather_regime,
            "weights": {m: round(w, 4) for m, w in weights.items()},
            "historical_residual_std": round(self.historical_residual_std, 4),
        }


# Alias for backward compatibility
RegimeGatingBlender = WeatherRegimeGatingNetwork

__all__ = [
    "GatingPyTorchModule",
    "WeatherRegimeGatingNetwork",
    "RegimeGatingBlender",
]
