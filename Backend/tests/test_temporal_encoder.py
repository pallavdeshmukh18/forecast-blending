"""
Unit tests for Spatio-Temporal ConvGRU cell and SpatioTemporalEncoder.
"""

import torch
import numpy as np
import pytest

from models.temporal_encoder import ConvGRUCell, SpatioTemporalEncoder


def test_conv_gru_cell_forward():
    B, C_in, H, W = 2, 16, 8, 8
    hidden_dim = 32

    cell = ConvGRUCell(in_channels=C_in, hidden_dim=hidden_dim, kernel_size=3)
    x = torch.randn(B, C_in, H, W)

    # Initial step (no prior hidden state)
    h1 = cell(x, None)
    assert h1.shape == (B, hidden_dim, H, W)

    # Next step with prior hidden state
    x2 = torch.randn(B, C_in, H, W)
    h2 = cell(x2, h1)
    assert h2.shape == (B, hidden_dim, H, W)
    assert not torch.allclose(h1, h2)


def test_spatiotemporal_encoder_sequence():
    B, T_seq, C_in, H, W = 2, 4, 12, 8, 8
    hidden_dim = 24

    encoder = SpatioTemporalEncoder(
        in_channels=C_in,
        spatial_channels=[16, 16],
        hidden_dim=hidden_dim,
        kernel_size=3,
        dropout=0.0,
    )

    x_seq = torch.randn(B, T_seq, C_in, H, W)
    h_out = encoder(x_seq)

    assert h_out.shape == (B, hidden_dim, H, W)
    assert not torch.isnan(h_out).any()
