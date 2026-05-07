from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class ModelConfig:
    vision_encoder_name: str = "openai/clip-vit-base-patch32"
    caption_model_name: str = "Salesforce/blip-image-captioning-base"
    translator_model_name: str = "Helsinki-NLP/opus-mt-en-ru"
    sentence_embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    ocr_langs: List[str] = field(default_factory=lambda: ["ru", "en"])
    device: str = "auto"


@dataclass
class RetrievalConfig:
    enabled: bool = True
    embedding_backend: str = "clip"
    top_k: int = 5
    use_faiss: bool = True
    index_path: str = "artifacts/retrieval/faiss.index"
    metadata_path: str = "artifacts/retrieval/metadata.jsonl"


@dataclass
class OCRConfig:
    enabled: bool = True
    backend: str = "easyocr"
    min_confidence: float = 0.3
    merge_lines: bool = True


@dataclass
class DatasetSourceConfig:
    name: str
    manifest_path: str
    image_root: str
    source_type: str
    language: str = "ru"
    weight: float = 1.0
    enabled: bool = True


@dataclass
class TrainingConfig:
    seed: int = 42
    batch_size: int = 8
    num_workers: int = 4
    learning_rate: float = 2e-5
    weight_decay: float = 1e-4
    max_epochs: int = 5
    warmup_ratio: float = 0.1
    gradient_accumulation_steps: int = 2
    max_text_length: int = 128
    image_size: int = 384
    mixed_precision: bool = True

    use_multitask: bool = True
    loss_caption_weight: float = 1.0
    loss_style_weight: float = 0.5
    loss_period_weight: float = 0.5
    loss_emotion_weight: float = 0.3
    loss_tags_weight: float = 0.5
    loss_ocr_weight: float = 0.2


@dataclass
class TaxonomyConfig:
    image_types: List[str] = field(default_factory=lambda: [
        "postcard",
        "poster",
        "illustration",
        "photo",
        "mixed_graphic",
    ])
    styles: List[str] = field(default_factory=lambda: [
        "vintage",
        "retro",
        "engraving",
        "watercolor",
        "painting",
        "propaganda",
        "decorative",
        "photographic",
        "minimalist",
        "unknown",
    ])
    periods: List[str] = field(default_factory=lambda: [
        "pre_1900",
        "1900_1917",
        "1918_1945",
        "1946_1960",
        "1961_1980",
        "1981_2000",
        "post_2000",
        "unknown",
    ])
    emotions: List[str] = field(default_factory=lambda: [
        "joy",
        "nostalgia",
        "solemnity",
        "romance",
        "melancholy",
        "patriotism",
        "festivity",
        "neutral",
    ])


@dataclass
class ProjectConfig:
    project_name: str = "tsu-image-description-research"
    root_dir: str = "."
    data_dir: str = "data"
    artifacts_dir: str = "artifacts"
    cache_dir: str = ".cache"

    models: ModelConfig = field(default_factory=ModelConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    ocr: OCRConfig = field(default_factory=OCRConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    taxonomy: TaxonomyConfig = field(default_factory=TaxonomyConfig)

    dataset_sources: List[DatasetSourceConfig] = field(default_factory=lambda: [
        DatasetSourceConfig(
            name="local_gold",
            manifest_path="data/manifests/local_gold.jsonl",
            image_root="data/images",
            source_type="local_curated",
            language="ru",
            weight=3.0,
        ),
        DatasetSourceConfig(
            name="europeana",
            manifest_path="data/manifests/europeana.jsonl",
            image_root="data/external/europeana/images",
            source_type="cultural_heritage",
            language="en",
            weight=1.5,
        ),
        DatasetSourceConfig(
            name="wikiart",
            manifest_path="data/manifests/wikiart.jsonl",
            image_root="data/external/wikiart/images",
            source_type="art_style",
            language="en",
            weight=1.0,
        ),
        DatasetSourceConfig(
            name="artemis",
            manifest_path="data/manifests/artemis.jsonl",
            image_root="data/external/artemis/images",
            source_type="emotion_art",
            language="en",
            weight=1.0,
        ),
    ])


def get_default_config() -> ProjectConfig:
    return ProjectConfig()


def resolve_path(root_dir: str, value: str) -> Path:
    return Path(root_dir).joinpath(value).resolve()
