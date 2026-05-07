# file: src/inspect_nypl_manifest.py

from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    manifest_path = Path("data/external/nypl/nypl_manifest.jsonl")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    rows = []
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))

    print(f"Rows: {len(rows)}")
    for row in rows[:5]:
        print("-" * 80)
        print("sample_id:", row["sample_id"])
        print("image:", row["image"])
        print("text:", row["text"][:250])


if __name__ == "__main__":
    main()
