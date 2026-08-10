"""
Convolutional Block Attention Module (CBAM) for Phyto Project.
Groundnut Plant Disease Classification (Edge-AI Framework).

Implements Channel Attention and Spatial Attention modules as proposed by Woo et al. (ECCV 2018).
"""

import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """
    Channel Attention Module (CAM) of CBAM.
    Aggregates spatial information via MaxPool and AvgPool, passes through shared MLP,
    and produces channel-wise attention weights via Sigmoid.
    """

    def __init__(self, in_channels: int, reduction: int = 16) -> None:
        super().__init__()
        mid_channels = max(1, in_channels // reduction)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, in_channels, kernel_size=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        out = self.sigmoid(avg_out + max_out)
        return x * out


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module (SAM) of CBAM.
    Aggregates channel information via MaxPool and AvgPool along channel axis,
    applies a 7x7 Conv2d, and produces spatial attention map via Sigmoid.
    """

    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.sigmoid(self.conv(spatial_cat))
        return x * out


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module combining Channel Attention and Spatial Attention sequentially.
    """

    def __init__(self, in_channels: int, reduction: int = 16, kernel_size: int = 7) -> None:
        super().__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction=reduction)
        self.spatial_attention = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x
