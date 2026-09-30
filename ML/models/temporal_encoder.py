"""
Spatio-Temporal Memory and Temporal Encoder Module for Phase 6.
Implements ConvGRU and CNN+GRU temporal architectures for learning weather motion and sequential forecast dynamics.
"""

from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.spatial_encoder import SpatialCNNEncoder


class ConvGRUCell(nn.Module):
    """
    2D Convolutional GRU cell that processes spatial features over sequential time steps
    while preserving 2D spatial dimensions (H, W).
    """

    def __init__(self, in_channels: int, hidden_dim: int, kernel_size: int = 3):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        padding = kernel_size // 2

        # Reset and update gates combined convolution
        self.conv_gates = nn.Conv2d(
            in_channels=in_channels + hidden_dim,
            out_channels=2 * hidden_dim,
            kernel_size=kernel_size,
            padding=padding,
            bias=True,
        )

        # Candidate hidden state convolution
        self.conv_candidate = nn.Conv2d(
            in_channels=in_channels + hidden_dim,
            out_channels=hidden_dim,
            kernel_size=kernel_size,
            padding=padding,
            bias=True,
        )

    def forward(self, x: torch.Tensor, h_prev: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor of shape (B, in_channels, H, W)
        h_prev : torch.Tensor of shape (B, hidden_dim, H, W) or None

        Returns
        -------
        h_next : torch.Tensor of shape (B, hidden_dim, H, W)
        """
        B, _, H, W = x.shape
        if h_prev is None:
            h_prev = torch.zeros(B, self.hidden_dim, H, W, device=x.device, dtype=x.dtype)

        combined = torch.cat([x, h_prev], dim=1)
        gates = self.conv_gates(combined)
        r_gate, z_gate = torch.chunk(gates, 2, dim=1)
        r = torch.sigmoid(r_gate)
        z = torch.sigmoid(z_gate)

        candidate_input = torch.cat([x, r * h_prev], dim=1)
        n = torch.tanh(self.conv_candidate(candidate_input))

        h_next = (1.0 - z) * n + z * h_prev
        return h_next


class SpatioTemporalEncoder(nn.Module):
    """
    Spatio-Temporal Feature Encoder.
    Processes a sequence of historical spatial grids [t-3, t-2, t-1, t] through a 2D Spatial CNN
    followed by a recurrent ConvGRU memory layer.
    """

    def __init__(
        self,
        in_channels: int,
        spatial_channels: Optional[List[int]] = None,
        hidden_dim: int = 32,
        kernel_size: int = 3,
        dropout: float = 0.10,
    ):
        super().__init__()
        if spatial_channels is None:
            spatial_channels = [32, 32, 32]

        self.in_channels = in_channels
        self.hidden_dim = hidden_dim

        # 1. 2D Spatial CNN Encoder (weights shared across all time steps)
        self.spatial_cnn = SpatialCNNEncoder(
            in_channels=in_channels,
            channels=spatial_channels,
            kernel_size=kernel_size,
            dropout=dropout,
        )

        # 2. Recurrent ConvGRU cell over sequence
        self.conv_gru = ConvGRUCell(
            in_channels=self.spatial_cnn.out_channels,
            hidden_dim=hidden_dim,
            kernel_size=kernel_size,
        )

        self.out_channels = hidden_dim

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        """
        Forward pass across temporal sequence.

        Parameters
        ----------
        x_seq : torch.Tensor of shape (B, T_seq, in_channels, H, W)

        Returns
        -------
        h_final : torch.Tensor of shape (B, hidden_dim, H, W) - final temporal state representation
        """
        B, T_seq, C, H, W = x_seq.shape

        # Reshape to run CNN on all frames concurrently: (B * T, C, H, W)
        x_flat = x_seq.reshape(B * T_seq, C, H, W)
        spatial_feats = self.spatial_cnn(x_flat)  # (B * T, C_spatial, H, W)
        C_spatial = spatial_feats.shape[1]
        spatial_feats = spatial_feats.reshape(B, T_seq, C_spatial, H, W)

        # Recurrent pass over time steps
        h = None
        for t in range(T_seq):
            xt = spatial_feats[:, t, :, :, :]
            h = self.conv_gru(xt, h)

        return h
