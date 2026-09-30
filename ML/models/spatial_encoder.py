"""
Spatial CNN Encoder Module for Phase 5.

Implements a lightweight, CPU-efficient 2D Convolutional Neural Network encoder
that extracts multi-scale spatial representations while preserving 2D spatial dimensions.
"""

from typing import List, Optional
import torch
import torch.nn as nn


class SpatialCNNEncoder(nn.Module):
    """
    Lightweight 2D CNN encoder for gridded meteorological feature fields.
    Preserves spatial grid resolution (H, W) using same-padding convolutions.
    """

    def __init__(
        self,
        in_channels: int,
        channels: Optional[List[int]] = None,
        kernel_size: int = 3,
        dropout: float = 0.10,
    ):
        super().__init__()
        if channels is None:
            channels = [32, 32, 64]

        self.in_channels = in_channels
        self.channels = channels
        self.out_channels = channels[-1]

        layers: List[nn.Module] = []
        prev_c = in_channels
        padding = kernel_size // 2

        for i, c in enumerate(channels):
            layers.append(
                nn.Conv2d(
                    in_channels=prev_c,
                    out_channels=c,
                    kernel_size=kernel_size,
                    padding=padding,
                    bias=True,
                )
            )
            layers.append(nn.BatchNorm2d(c))
            layers.append(nn.ReLU())
            if dropout > 0 and i < len(channels) - 1:
                layers.append(nn.Dropout2d(p=dropout))
            prev_c = c

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor of shape (Batch, in_channels, H, W)

        Returns
        -------
        torch.Tensor of shape (Batch, out_channels, H, W)
        """
        return self.net(x)
