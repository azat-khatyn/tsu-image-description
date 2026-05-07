# file: src/tsu_image_description/data/dataset.py

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset

from tsu_image_description.config import ProjectConfig, DatasetSourceConfig
from tsu_image_description.data.io import load_dataset_records
from tsu_image_description.data.schema import DatasetRecord


def _safe_open_image(path: str | Path) -> Image.Image:
    image = Image.open(path)
    return image.convert("RGB")


class MultiSourceArchiveDataset(Dataset):
    """
    Единый датасет для:
    - локального gold набора
    - Europeana / NYPL-подобных архивных изображений
    - WikiArt
    - ArtEmis
    """

    def __init__(
        self,
        config: ProjectConfig,
        split: str = "train",
        image_transform=None,
        text_language: str = "ru",
        include_sources: Optional[List[str]] = None,
        oversample_by_weight: bool = True,
    ) -> None:
        self.config = config
        self.split = split
        self.image_transform = image_transform
        self.text_language = text_language
        self.include_sources = include_sources
        self.oversample_by_weight = oversample_by_weight

        self.records: List[DatasetRecord] = self._load_records()
        self.indexes: List[int] = self._build_sampling_index()

    def _iter_sources(self) -> List[DatasetSourceConfig]:
        sources = []
        for source in self.config.dataset_sources:
            if not source.enabled:
                continue
            if self.include_sources and source.name not in self.include_sources:
                continue
            sources.append(source)
        return sources

    def _load_records(self) -> List[DatasetRecord]:
        all_records: List[DatasetRecord] = []

        for source in self._iter_sources():
            source_records = load_dataset_records(source.manifest_path)
            for record in source_records:
                if record.split != self.split:
                    continue
                record.weight = float(source.weight)
                record.image_path = str(
                    Path(source.image_root).joinpath(record.image_path).resolve()
                )
                all_records.append(record)

        if not all_records:
            raise RuntimeError(
                f"No records loaded for split='{self.split}'. "
                "Check manifest paths and data availability."
            )

        return all_records

    def _build_sampling_index(self) -> List[int]:
        if not self.oversample_by_weight or self.split != "train":
            return list(range(len(self.records)))

        index_list: List[int] = []
        for idx, record in enumerate(self.records):
            repeats = max(1, int(round(record.weight)))
            index_list.extend([idx] * repeats)

        random.shuffle(index_list)
        return index_list

    def __len__(self) -> int:
        return len(self.indexes)

    def _select_text(self, record: DatasetRecord) -> Optional[str]:
        if self.text_language == "ru":
            return record.caption_ru or record.search_text_ru or record.caption_en
        return record.caption_en or record.search_text_en or record.caption_ru

    def __getitem__(self, index: int) -> Dict[str, Any]:
        record = self.records[self.indexes[index]]
        image = _safe_open_image(record.image_path)

        if self.image_transform is not None:
            image_tensor = self.image_transform(image)
        else:
            image_tensor = image

        target_text = self._select_text(record)

        sample = {
            "sample_id": record.sample_id,
            "image": image_tensor,
            "image_path": record.image_path,
            "source_name": record.source_name,
            "source_type": record.source_type,
            "target_text": target_text,
            "ocr_text": record.ocr_text if record.has_ocr_text else None,
            "metadata": {
                "image_type": record.metadata.image_type,
                "style": record.metadata.style,
                "period": record.metadata.period,
                "emotion": record.metadata.emotion,
                "tags": record.metadata.tags,
                "printed_text": record.metadata.printed_text,
                "objects": record.metadata.objects,
                "cultural_context": record.metadata.cultural_context,
                "location_hint": record.metadata.location_hint,
            },
            "extra": record.extra,
        }
        return sample


def collate_multisource_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    images = [item["image"] for item in batch]
    if torch.is_tensor(images[0]):
        images = torch.stack(images, dim=0)

    return {
        "sample_id": [item["sample_id"] for item in batch],
        "image": images,
        "image_path": [item["image_path"] for item in batch],
        "source_name": [item["source_name"] for item in batch],
        "source_type": [item["source_type"] for item in batch],
        "target_text": [item["target_text"] for item in batch],
        "ocr_text": [item["ocr_text"] for item in batch],
        "metadata": [item["metadata"] for item in batch],
        "extra": [item["extra"] for item in batch],
    }
