from __future__ import annotations

import random
from typing import Sequence

import numpy as np
import torch
from PIL import Image


def _normalize(image: torch.Tensor, mean: Sequence[float], std: Sequence[float]) -> torch.Tensor:
    mean_t = torch.tensor(mean, dtype=image.dtype).view(3, 1, 1)
    std_t = torch.tensor(std, dtype=image.dtype).view(3, 1, 1)
    return (image - mean_t) / std_t


def pil_to_image_tensor(image: Image.Image) -> torch.Tensor:
    arr = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def pil_to_mask_tensor(mask: Image.Image) -> torch.Tensor:
    arr = np.asarray(mask, dtype=np.float32)
    arr = (arr > 127.0).astype(np.float32)
    return torch.from_numpy(arr).unsqueeze(0).contiguous()


class TrainPairTransform:
    def __init__(
        self,
        image_size: int,
        mean: Sequence[float],
        std: Sequence[float],
        hflip_prob: float = 0.5,
    ) -> None:
        self.image_size = int(image_size)
        self.mean = tuple(mean)
        self.std = tuple(std)
        self.hflip_prob = float(hflip_prob)

    def __call__(self, image: Image.Image, mask: Image.Image) -> tuple[torch.Tensor, torch.Tensor]:
        image = image.resize((self.image_size, self.image_size), resample=Image.BILINEAR)
        mask = mask.resize((self.image_size, self.image_size), resample=Image.NEAREST)
        if random.random() < self.hflip_prob:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        image_t = _normalize(pil_to_image_tensor(image), self.mean, self.std)
        mask_t = pil_to_mask_tensor(mask)
        return image_t, mask_t


class EvalPairTransform:
    def __init__(self, image_size: int, mean: Sequence[float], std: Sequence[float]) -> None:
        self.image_size = int(image_size)
        self.mean = tuple(mean)
        self.std = tuple(std)

    def __call__(self, image: Image.Image, mask: Image.Image) -> tuple[torch.Tensor, torch.Tensor]:
        image = image.resize((self.image_size, self.image_size), resample=Image.BILINEAR)
        mask = mask.resize((self.image_size, self.image_size), resample=Image.NEAREST)
        image_t = _normalize(pil_to_image_tensor(image), self.mean, self.std)
        mask_t = pil_to_mask_tensor(mask)
        return image_t, mask_t
