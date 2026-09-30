"""
Spatio-Temporal Mixture-of-Experts Gating Network and Blender for Phase 6.
Combines 2D Spatial CNN representations with sequential ConvGRU memory to compute dynamic 2D model weights W_m(x,y,t).
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from configs import AppConfig, get_config
from models.temporal_encoder import SpatioTemporalEncoder


class SpatioTemporalGatingNetwork(nn.Module):
    """
    Spatio-Temporal Gating Network for Multi-Model Forecast Blending.
    Maps historical sequence of spatial features [t-3, t-2, t-1, t] to 2D model weights W_m(x,y,t).
    """

    def __init__(
        self,
        in_channels: int,
        num_models: int,
        spatial_channels: Optional[List[int]] = None,
        hidden_dim: int = 32,
        kernel_size: int = 3,
        dropout: float = 0.10,
        use_residual_head: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_models = num_models
        self.use_residual_head = use_residual_head

        # 1. Spatio-Temporal Encoder
        self.encoder = SpatioTemporalEncoder(
            in_channels=in_channels,
            spatial_channels=spatial_channels,
            hidden_dim=hidden_dim,
            kernel_size=kernel_size,
            dropout=dropout,
        )

        # 2. 1x1 Conv Gating Head
        self.gating_head = nn.Conv2d(
            in_channels=self.encoder.out_channels,
            out_channels=num_models,
            kernel_size=1,
            bias=True,
        )

        # 3. Softmax over candidate models (dim=1)
        self.softmax = nn.Softmax(dim=1)

        # 4. Optional Residual Head
        if self.use_residual_head:
            self.residual_head = nn.Conv2d(
                in_channels=self.encoder.out_channels,
                out_channels=1,
                kernel_size=1,
                bias=True,
            )
        else:
            self.residual_head = None

    def forward(
        self,
        x_seq: torch.Tensor,
        f_target: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Parameters
        ----------
        x_seq : torch.Tensor of shape (B, T_seq, in_channels, H, W)
        f_target : torch.Tensor of shape (B, num_models, H, W)

        Returns
        -------
        blended : torch.Tensor of shape (B, 1, H, W)
        weights : torch.Tensor of shape (B, num_models, H, W)
        """
        # Spatio-temporal context feature map (B, hidden_dim, H, W)
        feat = self.encoder(x_seq)

        # Per-cell model logits
        logits = self.gating_head(feat)

        # Non-negative weights summing to 1 across models
        weights = self.softmax(logits)

        # Weighted blend
        blended = torch.sum(weights * f_target, dim=1, keepdim=True)

        if self.residual_head is not None:
            blended = blended + self.residual_head(feat)

        return blended, weights


class SpatioTemporalCNNBlender:
    """
    High-level training and prediction interface for Spatio-Temporal forecast blending.
    """

    def __init__(
        self,
        num_models: int = 3,
        in_channels: int = 16,
        sequence_length: int = 4,
        model_names: Optional[List[str]] = None,
        spatial_channels: Optional[List[int]] = None,
        hidden_dim: int = 32,
        kernel_size: int = 3,
        dropout: float = 0.10,
        learning_rate: float = 0.001,
        epochs: int = 40,
        batch_size: int = 8,
        early_stopping_patience: int = 6,
        weight_decay: float = 1e-4,
        random_seed: int = 42,
        use_residual_head: bool = False,
    ):
        self.num_models = num_models
        self.in_channels = in_channels
        self.sequence_length = sequence_length
        self.model_names = model_names or [f"model_{i}" for i in range(num_models)]
        self.spatial_channels = spatial_channels or [32, 32, 32]
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.early_stopping_patience = early_stopping_patience
        self.weight_decay = weight_decay
        self.random_seed = random_seed
        self.use_residual_head = use_residual_head

        # Normalization parameters (per channel mean & std across sequence)
        self.feature_means: Optional[np.ndarray] = None
        self.feature_stds: Optional[np.ndarray] = None

        self.network: Optional[SpatioTemporalGatingNetwork] = None
        self.history: List[Dict[str, float]] = []

    def _normalize_features(self, x_seq: np.ndarray, fit: bool = False) -> np.ndarray:
        """
        Normalize sequential features along channel dimension (N, T_seq, C, H, W).
        Fitted strictly on training data.
        """
        if fit:
            # Compute mean and std per channel across batch, time, and spatial dimensions
            self.feature_means = np.mean(x_seq, axis=(0, 1, 3, 4), keepdims=True).astype(np.float32)
            self.feature_stds = np.std(x_seq, axis=(0, 1, 3, 4), keepdims=True).astype(np.float32)
            self.feature_stds = np.where(self.feature_stds < 1e-6, 1.0, self.feature_stds)

        if self.feature_means is None or self.feature_stds is None:
            return x_seq

        return (x_seq - self.feature_means) / self.feature_stds

    def fit(
        self,
        X_seq_train: np.ndarray,
        F_train: np.ndarray,
        Y_train: np.ndarray,
        X_seq_val: Optional[np.ndarray] = None,
        F_val: Optional[np.ndarray] = None,
        Y_val: Optional[np.ndarray] = None,
        verbose: bool = False,
    ) -> "SpatioTemporalCNNBlender":
        """
        Train the Spatio-Temporal Gating Network on sequential historical grids.

        Parameters
        ----------
        X_seq_train : np.ndarray of shape (N, T_seq, C, H, W)
        F_train : np.ndarray of shape (N, M, H, W) - target cycle model forecasts
        Y_train : np.ndarray of shape (N, 1, H, W) or (N, H, W) - target ground truth
        """
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)

        if Y_train.ndim == 3:
            Y_train = Y_train[:, np.newaxis, :, :]
        if Y_val is not None and Y_val.ndim == 3:
            Y_val = Y_val[:, np.newaxis, :, :]

        self.sequence_length = X_seq_train.shape[1]
        self.in_channels = X_seq_train.shape[2]
        self.num_models = F_train.shape[1]

        # 1. Fit normalizer on training set ONLY
        X_train_norm = self._normalize_features(X_seq_train, fit=True)
        if X_seq_val is not None:
            X_val_norm = self._normalize_features(X_seq_val, fit=False)

        # 2. Instantiate Network
        self.network = SpatioTemporalGatingNetwork(
            in_channels=self.in_channels,
            num_models=self.num_models,
            spatial_channels=self.spatial_channels,
            hidden_dim=self.hidden_dim,
            kernel_size=self.kernel_size,
            dropout=self.dropout,
            use_residual_head=self.use_residual_head,
        )
        self.network.train()

        optimizer = optim.Adam(
            self.network.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        criterion = nn.MSELoss()

        t_X_train = torch.from_numpy(X_train_norm).float()
        t_F_train = torch.from_numpy(F_train).float()
        t_Y_train = torch.from_numpy(Y_train).float()

        if X_seq_val is not None and F_val is not None and Y_val is not None:
            t_X_val = torch.from_numpy(X_val_norm).float()
            t_F_val = torch.from_numpy(F_val).float()
            t_Y_val = torch.from_numpy(Y_val).float()
        else:
            t_X_val, t_F_val, t_Y_val = None, None, None

        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0
        N = X_seq_train.shape[0]

        for epoch in range(self.epochs):
            self.network.train()
            indices = np.random.permutation(N)
            epoch_loss = 0.0
            n_batches = 0

            for start_idx in range(0, N, self.batch_size):
                batch_idx = indices[start_idx : start_idx + self.batch_size]
                b_x = t_X_train[batch_idx]
                b_f = t_F_train[batch_idx]
                b_y = t_Y_train[batch_idx]

                optimizer.zero_grad()
                b_pred, _ = self.network(b_x, b_f)
                loss = criterion(b_pred, b_y)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            train_loss = epoch_loss / max(1, n_batches)

            val_loss = train_loss
            if t_X_val is not None:
                self.network.eval()
                with torch.no_grad():
                    val_pred, _ = self.network(t_X_val, t_F_val)
                    val_loss = criterion(val_pred, t_Y_val).item()

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {k: v.cpu().clone() for k, v in self.network.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.early_stopping_patience:
                        if verbose:
                            print(f"Early stopping triggered at epoch {epoch + 1}")
                        break

            self.history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss})

        if best_state is not None:
            self.network.load_state_dict(best_state)

        self.network.eval()
        return self

    def predict(
        self,
        X_seq: np.ndarray,
        F_target: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute blended spatial forecast field and 2D spatial model weights.

        Parameters
        ----------
        X_seq : np.ndarray of shape (N, T_seq, C, H, W)
        F_target : np.ndarray of shape (N, M, H, W)

        Returns
        -------
        blended : np.ndarray of shape (N, H, W)
        weights : np.ndarray of shape (N, M, H, W)
        """
        if self.network is None:
            raise RuntimeError("SpatioTemporalCNNBlender must be fitted before predict().")

        self.network.eval()
        X_norm = self._normalize_features(X_seq, fit=False)

        with torch.no_grad():
            t_x = torch.from_numpy(X_norm).float()
            t_f = torch.from_numpy(F_target).float()
            pred, weights = self.network(t_x, t_f)

            blended_np = pred.squeeze(1).cpu().numpy()
            weights_np = weights.cpu().numpy()

        return blended_np, weights_np

    def get_weights(self, X_seq: np.ndarray, F_target: np.ndarray) -> np.ndarray:
        """
        Get spatial model weights W_m(x,y,t) of shape (N, M, H, W).
        """
        _, weights = self.predict(X_seq, F_target)
        return weights
