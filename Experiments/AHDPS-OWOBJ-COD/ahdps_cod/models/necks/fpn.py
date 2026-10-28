from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FPN(nn.Module):
    def __init__(self, in_channels: tuple[int, int, int, int], out_channels: int) -> None:
        super().__init__()
        self.lateral = nn.ModuleDict({
            "c2": nn.Conv2d(in_channels[0], out_channels, 1),
            "c3": nn.Conv2d(in_channels[1], out_channels, 1),
            "c4": nn.Conv2d(in_channels[2], out_channels, 1),
            "c5": nn.Conv2d(in_channels[3], out_channels, 1),
        })
        self.smooth = nn.ModuleDict({
            "p2": nn.Conv2d(out_channels, out_channels, 3, padding=1),
            "p3": nn.Conv2d(out_channels, out_channels, 3, padding=1),
            "p4": nn.Conv2d(out_channels, out_channels, 3, padding=1),
            "p5": nn.Conv2d(out_channels, out_channels, 3, padding=1),
        })

    def forward(self, feats: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        c2, c3, c4, c5 = feats["c2"], feats["c3"], feats["c4"], feats["c5"]
        p5 = self.lateral["c5"](c5)
        p4 = self.lateral["c4"](c4) + F.interpolate(p5, size=c4.shape[-2:], mode="nearest")
        p3 = self.lateral["c3"](c3) + F.interpolate(p4, size=c3.shape[-2:], mode="nearest")
        p2 = self.lateral["c2"](c2) + F.interpolate(p3, size=c2.shape[-2:], mode="nearest")
        return {
            "p2": self.smooth["p2"](p2),
            "p3": self.smooth["p3"](p3),
            "p4": self.smooth["p4"](p4),
            "p5": self.smooth["p5"](p5),
        }
