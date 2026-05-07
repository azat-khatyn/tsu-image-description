# file: src/diagnose_nypl_apiuri_structure.py

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


TARGET_KEYS = {
    "abstract",
    "title",
    "titleinfo",
    "genre",
    "subject",
    "topic",
    "geographic",
    "temporal",
    "dateissued",
    "datecreated",
    "date",
    "note",
    "name",
    "namepart",
    "form",
}


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def walk(obj: Any, path: str = "$") -> Iterable[Tuple[str, Any]]:
    yield path, obj

    if isinstance(obj, dict):
        for k, v in obj.items():
            child_path = f"{path}.{k}"
            yield from walk(v, child_path)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            child_path = f"{path}[{i}]"
            yield from walk(v, child_path)


def flatten_text_values(obj: Any) -> List[str]:
    values: List[str] = []

    if obj is None:
        return values
    if isinstance(obj, str):
        text = " ".join(obj.split())
        if text:
            values.append(text)
    elif isinstance(obj, (int, float, bool)):
        values.append(str(obj))
    elif isinstance(obj, list):
        for item in obj:
            values.extend(flatten_text_values(item))
    elif isinstance(obj, dict):
        for v in obj.values():
            values.extend(flatten_text_values(v))

    # dedupe preserving order
    seen = set()
    out = []
    for v in values:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def collect_target_paths(data: Any) -> List[Tuple[str, Any]]:
    found = []
    for path, value in walk(data):
        key = path.split(".")[-1]
        key = key.split("[")[0].lower()
        if key in TARGET_KEYS:
            found.append((path, value))
    return found


def summarize_paths(raw_paths: List[Path], max_files: int = 20) -> None:
    files = raw_paths[:max_files]

    path_counter = Counter()
    key_examples = defaultdict(list)

    print(f"Inspecting {len(files)} raw API responses...\n")

    for raw_path in files:
        print("=" * 100)
        print(raw_path.name)
        print("=" * 100)

        try:
            text = raw_path.read_text(encoding="utf-8")
            data = json.loads(text)
        except Exception as e:
            print(f"[ERROR] Could not parse JSON: {e}\n")
            continue

        found = collect_target_paths(data)

        if not found:
            print("No target keys found.\n")
            continue

        for path, value in found:
            path_counter[path] += 1

            values = flatten_text_values(value)
            preview = values[:3]
            print(f"{path}")
            print(f"  preview: {preview}")

            key = path.split(".")[-1].split("[")[0].lower()
            if len(key_examples[key]) < 5:
                key_examples[key].append((path, preview))

        print()

    print("\n" + "#" * 100)
    print("COMMON PATHS")
    print("#" * 100)
    for path, count in path_counter.most_common(50):
        print(f"{count:>3}  {path}")

    print("\n" + "#" * 100)
    print("KEY EXAMPLES")
    print("#" * 100)
    for key, examples in sorted(key_examples.items()):
        print(f"\n[{key}]")
        for path, preview in examples:
            print(f"  {path}")
            print(f"    {preview}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect raw NYPL apiUri JSON structure and locate useful metadata paths."
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="data/external/nypl/nypl_manifest_enriched.jsonl",
        help="Path to enriched NYPL manifest with mods_raw_path fields.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="How many raw files to inspect.",
    )

    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    rows = load_jsonl(manifest_path)

    raw_paths: List[Path] = []
    for row in rows:
        p = row.get("mods_raw_path")
        if p:
            raw_paths.append(Path(p))

    if not raw_paths:
        raise RuntimeError(
            "No mods_raw_path found in manifest. Re-run enrich script with --save-raw."
        )

    summarize_paths(raw_paths, max_files=args.limit)


if __name__ == "__main__":
    main()
