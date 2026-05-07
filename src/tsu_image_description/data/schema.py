from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class StructuredMetadata:
    image_type: Optional[str] = None
    style: Optional[str] = None
    period: Optional[str] = None
    emotion: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    printed_text: Optional[str] = None
    objects: List[str] = field(default_factory=list)
    cultural_context: List[str] = field(default_factory=list)
    location_hint: Optional[str] = None


@dataclass
class DatasetRecord:
    sample_id: str
    image_path: str
    source_name: str
    source_type: str

    caption_ru: Optional[str] = None
    caption_en: Optional[str] = None
    search_text_ru: Optional[str] = None
    search_text_en: Optional[str] = None

    metadata: StructuredMetadata = field(default_factory=StructuredMetadata)

    has_ocr_text: bool = False
    ocr_text: Optional[str] = None

    split: str = "train"
    weight: float = 1.0

    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["metadata"] = asdict(self.metadata)
        return payload

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DatasetRecord":
        metadata = StructuredMetadata(**data.get("metadata", {}))
        return cls(
            sample_id=data["sample_id"],
            image_path=data["image_path"],
            source_name=data["source_name"],
            source_type=data["source_type"],
            caption_ru=data.get("caption_ru"),
            caption_en=data.get("caption_en"),
            search_text_ru=data.get("search_text_ru"),
            search_text_en=data.get("search_text_en"),
            metadata=metadata,
            has_ocr_text=data.get("has_ocr_text", False),
            ocr_text=data.get("ocr_text"),
            split=data.get("split", "train"),
            weight=float(data.get("weight", 1.0)),
            extra=data.get("extra", {}),
        )
