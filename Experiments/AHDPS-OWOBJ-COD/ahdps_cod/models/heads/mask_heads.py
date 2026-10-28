from __future__ import annotations

import torch.nn as nn


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0) -> None:
        layers = [
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(p=dropout))
        super().__init__(*layers)


class MaskHead(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.block = ConvBlock(in_channels, hidden_channels, dropout=dropout)
        self.head = nn.Conv2d(hidden_channels, 1, kernel_size=1)

    def forward(self, x):
        return self.head(self.block(x))
