from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SampleRecord:
    image_path: str
    mask_path: str
    dataset_name: str
    sample_id: str
