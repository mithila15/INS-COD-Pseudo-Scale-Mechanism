from __future__ import annotations

import torch
import torch.nn as nn


class ConvBNAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int | None = None):
        if padding is None:
            padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = ConvBNAct(channels, channels, 3, stride=stride)
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + residual
        return self.act(out)


class DownsampleStage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_blocks: int = 2) -> None:
        super().__init__()
        blocks: list[nn.Module] = [ConvBNAct(in_channels, out_channels, 3, stride=2)]
        blocks.extend([ResidualBlock(out_channels) for _ in range(num_blocks)])
        self.stage = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stage(x)


class SimplePyramidBackbone(nn.Module):
    def __init__(self, in_channels: int = 3, widths: tuple[int, int, int, int] = (64, 128, 256, 512)) -> None:
        super().__init__()
        w2, w3, w4, w5 = widths
        self.stem = nn.Sequential(
            ConvBNAct(in_channels, w2 // 2, 3, stride=2),
            ConvBNAct(w2 // 2, w2 // 2, 3, stride=1),
        )
        self.stage2 = DownsampleStage(w2 // 2, w2, num_blocks=2)  # /4
        self.stage3 = DownsampleStage(w2, w3, num_blocks=2)       # /8
        self.stage4 = DownsampleStage(w3, w4, num_blocks=2)       # /16
        self.stage5 = DownsampleStage(w4, w5, num_blocks=2)       # /32

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.stem(x)
        c2 = self.stage2(x)
        c3 = self.stage3(c2)
        c4 = self.stage4(c3)
        c5 = self.stage5(c4)
        return {"c2": c2, "c3": c3, "c4": c4, "c5": c5}
