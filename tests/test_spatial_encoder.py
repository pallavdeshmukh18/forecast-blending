"""
Tests for Spatial CNN Encoder Module (Phase 5).
"""

import torch
import pytest

from models.spatial_encoder import SpatialCNNEncoder


def test_spatial_cnn_encoder_shape():
    encoder = SpatialCNNEncoder(in_channels=12, channels=[16, 32, 64], kernel_size=3, dropout=0.1)
    x = torch.randn(4, 12, 16, 16)
    out = encoder(x)
    assert out.shape == (4, 64, 16, 16)


def test_spatial_cnn_encoder_eval_mode():
    encoder = SpatialCNNEncoder(in_channels=4, channels=[8, 16], kernel_size=3, dropout=0.2)
    encoder.eval()
    x = torch.randn(2, 4, 8, 8)
    with torch.no_grad():
        out1 = encoder(x)
        out2 = encoder(x)
    assert torch.allclose(out1, out2)
