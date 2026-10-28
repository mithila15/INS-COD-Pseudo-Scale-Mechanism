from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

from PIL import Image
from torch.utils.data import Dataset

from .common import SampleRecord


class CODBinaryDataset(Dataset):
    def __init__(
        self,
        records: Sequence[SampleRecord],
        transform: Callable | None = None,
    ) -> None:
        self.records = list(records)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        image = Image.open(record.image_path).convert("RGB")
        mask = Image.open(record.mask_path).convert("L")
        orig_size = image.size[::-1]  # H, W
        if self.transform is not None:
            image_t, mask_t = self.transform(image, mask)
        else:
            raise RuntimeError("Transform must be provided for CODBinaryDataset.")

        return {
            "image": image_t,
            "mask": mask_t,
            "dataset_name": record.dataset_name,
            "sample_id": record.sample_id,
            "image_path": str(Path(record.image_path)),
            "mask_path": str(Path(record.mask_path)),
            "orig_size": orig_size,
        }
