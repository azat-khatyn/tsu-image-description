# file: src/analyze_nypl_manifest.py

from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def nonempty_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        value = " ".join(value.split()).strip()
        return value if value else None
    return None


def text_len_words(text: str) -> int:
    return len(text.split())


def text_len_chars(text: str) -> int:
    return len(text)


def safe_len_list(value: Any) -> int:
    if isinstance(value, list):
        return len(value)
    return 0


def summarize_text_field(rows: List[Dict[str, Any]], field: str) -> Dict[str, Any]:
    texts = [nonempty_text(r.get(field)) for r in rows]
    texts = [t for t in texts if t]

    if not texts:
        return {
            "field": field,
            "count": 0,
            "share": 0.0,
            "avg_words": 0.0,
            "median_words": 0.0,
            "avg_chars": 0.0,
            "median_chars": 0.0,
        }

    words = [text_len_words(t) for t in texts]
    chars = [text_len_chars(t) for t in texts]

    return {
        "field": field,
        "count": len(texts),
        "share": round(len(texts) / len(rows), 4) if rows else 0.0,
        "avg_words": round(sum(words) / len(words), 2),
        "median_words": statistics.median(words),
        "avg_chars": round(sum(chars) / len(chars), 2),
        "median_chars": statistics.median(chars),
    }


def top_counter(rows: List[Dict[str, Any]], key: str, top_k: int = 15) -> List[tuple[str, int]]:
    counter = Counter()
    for row in rows:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            counter[value.strip()] += 1
    return counter.most_common(top_k)


def top_list_values(rows: List[Dict[str, Any]], key: str, top_k: int = 20) -> List[tuple[str, int]]:
    counter = Counter()
    for row in rows:
        value = row.get(key)
        if isinstance(value, list):
            for item in value:
                if isinstance(item, str) and item.strip():
                    counter[item.strip()] += 1
    return counter.most_common(top_k)


def collect_examples(
    rows: List[Dict[str, Any]],
    text_priority: List[str],
    num_examples: int = 10,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    candidates = []
    for row in rows:
        chosen_text = None
        chosen_field = None
        for field in text_priority:
            text = nonempty_text(row.get(field))
            if text:
                chosen_text = text
                chosen_field = field
                break

        if chosen_text:
            candidates.append({
                "sample_id": row.get("sample_id"),
                "source_type": row.get("source_type"),
                "query": row.get("query"),
                "field": row.get("field"),
                "title": row.get("title"),
                "text_field": chosen_field,
                "text": chosen_text,
            })

    random.Random(seed).shuffle(candidates)
    return candidates[:num_examples]


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze NYPL manifest quality and coverage.")
    parser.add_argument(
        "--manifest",
        type=str,
        default="data/external/nypl_expanded/nypl_manifest_expanded.jsonl",
        help="Path to manifest jsonl",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/external/nypl_expanded/manifest_analysis.json",
        help="Where to save analysis JSON",
    )
    parser.add_argument(
        "--examples",
        type=int,
        default=12,
        help="How many random examples to print/save",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    rows = load_jsonl(manifest_path)

    if not rows:
        raise RuntimeError(f"No rows found in {manifest_path}")

    summary = {
        "num_rows": len(rows),
        "num_with_title": sum(1 for r in rows if nonempty_text(r.get("title"))),
        "num_with_abstract": sum(1 for r in rows if nonempty_text(r.get("abstract"))),
        "num_with_best_text": sum(1 for r in rows if nonempty_text(r.get("best_text"))),
        "num_with_image_links": sum(1 for r in rows if safe_len_list(r.get("image_links")) > 0),
        "num_with_item_details_image_links": sum(
            1 for r in rows if safe_len_list(r.get("item_details_image_links")) > 0
        ),
        "num_with_chosen_image_url": sum(1 for r in rows if nonempty_text(r.get("chosen_image_url"))),
        "source_type_top": top_counter(rows, "source_type"),
        "query_top": top_counter(rows, "query"),
        "field_top": top_counter(rows, "field"),
        "collection_title_top": top_counter(rows, "collection_title"),
        "type_of_resource_top": top_list_values(rows, "type_of_resource"),
        "genres_mods_top": top_list_values(rows, "genres_mods"),
        "subjects_mods_top": top_list_values(rows, "subjects_mods"),
        "text_fields": [
            summarize_text_field(rows, "title"),
            summarize_text_field(rows, "abstract"),
            summarize_text_field(rows, "best_text"),
        ],
        "examples": collect_examples(
            rows,
            text_priority=["best_text", "abstract", "title"],
            num_examples=args.examples,
        ),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== SUMMARY ===")
    print(json.dumps({k: v for k, v in summary.items() if k != "examples"}, ensure_ascii=False, indent=2))

    print("\n=== EXAMPLES ===")
    for ex in summary["examples"]:
        print("-" * 100)
        print(f"sample_id: {ex['sample_id']}")
        print(f"source_type: {ex['source_type']}")
        print(f"query/field: {ex['query']} / {ex['field']}")
        print(f"title: {ex['title']}")
        print(f"text_field: {ex['text_field']}")
        print(f"text: {ex['text'][:400]}")

    print(f"\nSaved analysis to: {output_path}")


if __name__ == "__main__":
    main()
