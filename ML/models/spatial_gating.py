"""
Spatial Neural Gating Network & Spatial CNN Blender for Phase 5.

Implements a Mixture-of-Experts Spatial Gating Network that produces spatially varying,
convex model weights W_m(x,y) >= 0, sum_m W_m(x,y) = 1 at every grid cell, and blends
multi-model 2D forecast fields dynamically conditioned on local spatial atmospheric states.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from configs import AppConfig, get_config
from models.spatial_encoder import SpatialCNNEncoder


class SpatialGatingNetwork(nn.Module):
    """
    PyTorch 2D Spatial Mixture-of-Experts Gating Network.

    Given a multi-channel spatial input tensor X (B, C, H, W) and candidate model forecast fields
    F (B, M, H, W), the network encodes spatial features, computes model logits per grid cell,
    applies a channel-wise Softmax over the M models to guarantee W_m(x,y) >= 0 and sum_m W_m(x,y) = 1,
    and blends the forecasts: F_blend(x,y) = sum_m W_m(x,y) * F_m(x,y).
    """

    def __init__(
        self,
        in_channels: int,
        num_models: int,
        channels: Optional[List[int]] = None,
        kernel_size: int = 3,
        dropout: float = 0.10,
        use_residual_head: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_models = num_models
        self.use_residual_head = use_residual_head

        # 1. Spatial CNN Encoder
        self.encoder = SpatialCNNEncoder(
            in_channels=in_channels,
            channels=channels,
            kernel_size=kernel_size,
            dropout=dropout,
        )

        # 2. 1x1 Conv Spatial Gating Head mapping to num_models logits
        self.gating_head = nn.Conv2d(
            in_channels=self.encoder.out_channels,
            out_channels=num_models,
            kernel_size=1,
            bias=True,
        )

        # 3. Softmax over model dimension (dim=1)
        self.softmax = nn.Softmax(dim=1)

        # 4. Optional Residual Head (disabled by default for pure convex forecast blending)
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
        x: torch.Tensor,
        f: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor of shape (B, in_channels, H, W)
        f : torch.Tensor of shape (B, num_models, H, W)

        Returns
        -------
        blended : torch.Tensor of shape (B, 1, H, W)
        weights : torch.Tensor of shape (B, num_models, H, W)
        """
        # Encode spatial context
        feat = self.encoder(x)

        # Compute per-grid-cell model logits
        logits = self.gating_head(feat)

        # Convex weights via Softmax over models
        weights = self.softmax(logits)

        # Weighted multi-model forecast blend
        blended = torch.sum(weights * f, dim=1, keepdim=True)

        if self.residual_head is not None:
            residual = self.residual_head(feat)
            blended = blended + residual

        return blended, weights


class SpatialCNNBlender:
    """
    High-level blender interface for training and evaluating the Spatial Gating Network.
    Includes feature standardization fitted strictly on training data, early stopping,
    and CPU training.
    """

    def __init__(
        self,
        num_models: int = 3,
        in_channels: int = 16,
        model_names: Optional[List[str]] = None,
        channels: Optional[List[int]] = None,
        kernel_size: int = 3,
        dropout: float = 0.10,
        learning_rate: float = 0.001,
        epochs: int = 50,
        batch_size: int = 16,
        early_stopping_patience: int = 8,
        weight_decay: float = 1e-4,
        random_seed: int = 42,
        use_residual_head: bool = False,
    ):
        self.num_models = num_models
        self.in_channels = in_channels
        self.model_names = model_names or [f"model_{i}" for i in range(num_models)]
        self.channels = channels or [32, 32, 64]
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.early_stopping_patience = early_stopping_patience
        self.weight_decay = weight_decay
        self.random_seed = random_seed
        self.use_residual_head = use_residual_head

        # Normalization parameters (per channel mean & std)
        self.feature_means: Optional[np.ndarray] = None
        self.feature_stds: Optional[np.ndarray] = None

        self.network: Optional[SpatialGatingNetwork] = None
        self.history: List[Dict[str, float]] = []

    def _normalize_features(self, x: np.ndarray, fit: bool = False) -> np.ndarray:
        """
        Normalize features along channel dimension (B, C, H, W).
        Fitted strictly on training data.
        """
        if fit:
            # x has shape (N, C, H, W)
            # Compute mean and std per channel
            self.feature_means = np.mean(x, axis=(0, 2, 3), keepdims=True).astype(np.float32)
            self.feature_stds = np.std(x, axis=(0, 2, 3), keepdims=True).astype(np.float32)
            # Avoid division by zero
            self.feature_stds = np.where(self.feature_stds < 1e-6, 1.0, self.feature_stds)

        if self.feature_means is None or self.feature_stds is None:
            return x

        return (x - self.feature_means) / self.feature_stds

    def fit(
        self,
        X_train: np.ndarray,
        F_train: np.ndarray,
        Y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        F_val: Optional[np.ndarray] = None,
        Y_val: Optional[np.ndarray] = None,
        verbose: bool = False,
    ) -> "SpatialCNNBlender":
        """
        Train the Spatial Gating Network on historical training grids.

        Parameters
        ----------
        X_train : np.ndarray (N_train, C, H, W) - feature fields
        F_train : np.ndarray (N_train, M, H, W) - candidate model forecasts
        Y_train : np.ndarray (N_train, 1, H, W) or (N_train, H, W) - ground truth
        X_val, F_val, Y_val : optional validation grids for early stopping
        """
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)

        if Y_train.ndim == 3:
            Y_train = Y_train[:, np.newaxis, :, :]
        if Y_val is not None and Y_val.ndim == 3:
            Y_val = Y_val[:, np.newaxis, :, :]

        self.in_channels = X_train.shape[1]
        self.num_models = F_train.shape[1]

        # 1. Fit normalizer on training set ONLY
        X_train_norm = self._normalize_features(X_train, fit=True)
        if X_val is not None:
            X_val_norm = self._normalize_features(X_val, fit=False)

        # 2. Instantiate Network
        self.network = SpatialGatingNetwork(
            in_channels=self.in_channels,
            num_models=self.num_models,
            channels=self.channels,
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

        if X_val is not None and F_val is not None and Y_val is not None:
            t_X_val = torch.from_numpy(X_val_norm).float()
            t_F_val = torch.from_numpy(F_val).float()
            t_Y_val = torch.from_numpy(Y_val).float()
        else:
            t_X_val, t_F_val, t_Y_val = None, None, None

        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0
        N = X_train.shape[0]

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

            # Validation loss evaluation
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
        X: np.ndarray,
        F: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute blended spatial forecast field and 2D spatial model weights.

        Parameters
        ----------
        X : np.ndarray (N, C, H, W)
        F : np.ndarray (N, M, H, W)

        Returns
        -------
        blended : np.ndarray of shape (N, H, W)
        weights : np.ndarray of shape (N, M, H, W)
        """
        if self.network is None:
            raise RuntimeError("SpatialCNNBlender must be fitted before predict().")

        self.network.eval()
        X_norm = self._normalize_features(X, fit=False)

        with torch.no_grad():
            t_x = torch.from_numpy(X_norm).float()
            t_f = torch.from_numpy(F).float()
            pred, weights = self.network(t_x, t_f)

            blended_np = pred.squeeze(1).cpu().numpy()
            weights_np = weights.cpu().numpy()

        return blended_np, weights_np

    def get_weights(self, X: np.ndarray, F: np.ndarray) -> np.ndarray:
        """
        Get spatial model weights W_m(x,y) of shape (N, M, H, W).
        """
        _, weights = self.predict(X, F)
        return weights
