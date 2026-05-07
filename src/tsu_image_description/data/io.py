from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List

from tsu_image_description.data.schema import DatasetRecord


def load_jsonl(path: str | Path) -> List[dict]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSONL file not found: {path}")

    items = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}") from exc
    return items


def save_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_dataset_records(path: str | Path) -> List[DatasetRecord]:
    rows = load_jsonl(path)
    return [DatasetRecord.from_dict(row) for row in rows]


def save_dataset_records(path: str | Path, records: Iterable[DatasetRecord]) -> None:
    save_jsonl(path, (record.to_dict() for record in records))
