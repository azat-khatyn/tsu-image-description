from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import requests


API_BASE = "https://api.repo.nypl.org/api/v2"


def build_headers(token: str) -> Dict[str, str]:
    return {"Authorization": f'Token token="{token}"'}


def safe_get(obj: Any, *keys, default=None):
    cur = obj
    for k in keys:
        if isinstance(cur, dict):
            cur = cur.get(k)
        else:
            return default
        if cur is None:
            return default
    return cur


def ensure_list(x):
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [x]


def flatten_strings(x) -> List[str]:
    result = []
    if x is None:
        return result
    if isinstance(x, str):
        s = x.strip()
        if s:
            result.append(s)
    elif isinstance(x, list):
        for item in x:
            result.extend(flatten_strings(item))
    elif isinstance(x, dict):
        for v in x.values():
            result.extend(flatten_strings(v))
    return result


def extract_results(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Пытаемся достать список результатов из search response.
    Структура NYPL может отличаться, поэтому ищем гибко.
    """
    candidates = [
        safe_get(payload, "response", "docs"),
        safe_get(payload, "nyplAPI", "response", "result"),
        safe_get(payload, "nyplAPI", "response", "capture"),
        safe_get(payload, "result"),
        safe_get(payload, "results"),
    ]
    for c in candidates:
        if isinstance(c, list):
            return c
    return []


def extract_uuid(record: Dict[str, Any]) -> Optional[str]:
    for key in ["uuid", "UUID"]:
        if key in record and isinstance(record[key], str):
            return record[key]
    return None


def extract_title(record: Dict[str, Any]) -> Optional[str]:
    titles = []
    for key in ["title", "search_text"]:
        titles.extend(flatten_strings(record.get(key)))
    if titles:
        return titles[0]
    return None


def extract_image_links(record: Dict[str, Any]) -> List[str]:
    links = []
    for key in ["imageLinks", "imageLink", "imageURL", "imageUrl", "highResLink"]:
        value = record.get(key)
        if isinstance(value, str):
            links.append(value)
        elif isinstance(value, list):
            links.extend([v for v in value if isinstance(v, str)])
    return list(dict.fromkeys(links))


def extract_item_link(record: Dict[str, Any]) -> Optional[str]:
    for key in ["itemLink", "url", "permalink"]:
        val = record.get(key)
        if isinstance(val, str):
            return val
    return None


def normalize_record(
    record: Dict[str, Any],
    query: str,
    field: Optional[str],
    target_type: str,
) -> Optional[Dict[str, Any]]:
    uuid = extract_uuid(record)
    if not uuid:
        return None

    title = extract_title(record)
    image_links = extract_image_links(record)
    item_link = extract_item_link(record)

    # Сохраняем сырые поля, которые могут пригодиться позже
    subjects = flatten_strings(record.get("subject"))
    genres = flatten_strings(record.get("genre"))
    type_of_resource = flatten_strings(record.get("typeOfResource"))
    date = flatten_strings(record.get("date"))

    return {
        "sample_id": uuid,
        "source": "nypl",
        "target_type": target_type,       # postcard/poster
        "query": query,
        "field": field,
        "title": title,
        "image_links": image_links,
        "item_link": item_link,
        "subjects": subjects,
        "genres": genres,
        "type_of_resource": type_of_resource,
        "date": date,
        "raw_record": record,
    }


def query_nypl(
    token: str,
    query: str,
    field: Optional[str],
    page: int,
    per_page: int = 100,
    public_domain_only: bool = True,
    type_filter: str = "still image",
) -> Dict[str, Any]:
    params = {
        "q": query,
        "page": page,
        "per_page": per_page,
        "publicDomainOnly": str(public_domain_only).lower(),
        "filter[]": f"typeOfResource:{type_filter}",
    }
    if field:
        params["field"] = field

    response = requests.get(
        f"{API_BASE}/items/search",
        headers=build_headers(token),
        params=params,
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


def save_json(path: Path, obj: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def save_jsonl(path: Path, rows: List[Dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_query_plan(mode: str) -> List[Tuple[str, Optional[str], str]]:
    """
    Возвращает список (query, field, target_type)
    """
    plans = []

    if mode in ("postcard", "both"):
        plans.extend([
            ("postcard", "genre", "postcard"),
            ("postcard", "topic", "postcard"),
            ("postcard", None, "postcard"),
        ])

    if mode in ("poster", "both"):
        plans.extend([
            ("poster", "genre", "poster"),
            ("poster", "topic", "poster"),
            ("poster", None, "poster"),
            ("posters", "topic", "poster"),  # иногда бывает во множественном числе
        ])

    return plans


def main():
    parser = argparse.ArgumentParser(
        description="Fetch NYPL metadata for postcards / posters"
    )
    parser.add_argument(
        "--token",
        type=str,
        default=os.getenv("NYPL_TOKEN"),
        help="NYPL API token (or set NYPL_TOKEN env var)"
    )
    parser.add_argument(
        "--mode",
        choices=["postcard", "poster", "both"],
        default="both",
        help="What to collect"
    )
    parser.add_argument(
        "--pages",
        type=int,
        default=3,
        help="How many pages per query to fetch"
    )
    parser.add_argument(
        "--per-page",
        type=int,
        default=100,
        help="Results per page (keep moderate)"
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.2,
        help="Sleep between requests"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/external/nypl",
        help="Where to save results"
    )

    args = parser.parse_args()

    if not args.token:
        raise ValueError(
            "NYPL token not provided. Use --token or set NYPL_TOKEN environment variable."
        )

    out_dir = Path(args.output_dir)
    raw_dir = out_dir / "raw_search"
    raw_dir.mkdir(parents=True, exist_ok=True)

    all_rows: List[Dict[str, Any]] = []
    seen_ids: Set[str] = set()

    query_plan = build_query_plan(args.mode)

    for query, field, target_type in query_plan:
        for page in range(1, args.pages + 1):
            print(f"Fetching query={query!r}, field={field!r}, page={page}")

            try:
                payload = query_nypl(
                    token=args.token,
                    query=query,
                    field=field,
                    page=page,
                    per_page=args.per_page,
                )
            except Exception as e:
                print(f"  Request failed: {e}")
                continue

            # Сохраняем сырой ответ
            field_name = field if field else "all"
            raw_path = raw_dir / f"{target_type}_{query}_{field_name}_page{page}.json"
            save_json(raw_path, payload)

            results = extract_results(payload)
            print(f"  Retrieved {len(results)} results")

            for rec in results:
                row = normalize_record(
                    record=rec,
                    query=query,
                    field=field,
                    target_type=target_type,
                )
                if row is None:
                    continue

                sample_id = row["sample_id"]
                if sample_id in seen_ids:
                    continue

                seen_ids.add(sample_id)
                all_rows.append(row)

            time.sleep(args.sleep)

    # Сохраняем полный JSONL
    manifest_path = out_dir / "nypl_manifest.jsonl"
    save_jsonl(manifest_path, all_rows)

    # Сохраняем краткую статистику
    stats = {
        "num_records": len(all_rows),
        "num_postcards": sum(1 for r in all_rows if r["target_type"] == "postcard"),
        "num_posters": sum(1 for r in all_rows if r["target_type"] == "poster"),
        "num_with_images": sum(1 for r in all_rows if r["image_links"]),
        "num_with_title": sum(1 for r in all_rows if r["title"]),
    }
    save_json(out_dir / "stats.json", stats)

    print("\nDone.")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"Saved manifest: {manifest_path}")


if __name__ == "__main__":
    main()