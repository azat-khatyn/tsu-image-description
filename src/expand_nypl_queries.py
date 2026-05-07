# file: src/expand_nypl_queries.py

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import requests


API_BASE = "https://api.repo.nypl.org/api/v2"


def build_headers(token: str) -> Dict[str, str]:
    return {"Authorization": f'Token token="{token}"'}


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def save_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def flatten_strings(x: Any) -> List[str]:
    result: List[str] = []
    if x is None:
        return result
    if isinstance(x, str):
        s = " ".join(x.split())
        if s:
            result.append(s)
    elif isinstance(x, list):
        for item in x:
            result.extend(flatten_strings(item))
    elif isinstance(x, dict):
        for v in x.values():
            result.extend(flatten_strings(v))
    return result


def recursive_find_records(obj: Any) -> List[Dict[str, Any]]:
    """
    Ищет dict-подобные записи с uuid.
    """
    found = []

    if isinstance(obj, dict):
        if "uuid" in obj and isinstance(obj["uuid"], str):
            found.append(obj)

        for v in obj.values():
            found.extend(recursive_find_records(v))

    elif isinstance(obj, list):
        for item in obj:
            found.extend(recursive_find_records(item))

    # dedupe by object id not needed here
    return found


def recursive_find_collections(obj: Any) -> List[Dict[str, Any]]:
    found = []
    if isinstance(obj, dict):
        keys = {k.lower() for k in obj.keys()}
        if "uuid" in obj and ("numItems" in obj or "numitems" in keys):
            found.append(obj)
        for v in obj.values():
            found.extend(recursive_find_collections(v))
    elif isinstance(obj, list):
        for item in obj:
            found.extend(recursive_find_collections(item))
    return found


def extract_title(record: Dict[str, Any]) -> Optional[str]:
    titles = flatten_strings(record.get("title"))
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
        elif isinstance(value, dict):
            links.extend(flatten_strings(value))
    # dedupe
    seen = set()
    out = []
    for link in links:
        if link not in seen:
            seen.add(link)
            out.append(link)
    return out


def normalize_item_record(
    record: Dict[str, Any],
    source_type: str,
    query: str,
    field: Optional[str],
    collection_uuid: Optional[str] = None,
    collection_title: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    uuid = record.get("uuid")
    if not isinstance(uuid, str):
        return None

    normalized = {
        "sample_id": uuid,
        "source": "nypl",
        "source_type": source_type,              # search / collection
        "query": query,
        "field": field,
        "collection_uuid": collection_uuid,
        "collection_title": collection_title,
        "title": extract_title(record),
        "image_links": extract_image_links(record),
        "item_link": record.get("itemLink") if isinstance(record.get("itemLink"), str) else None,
        "image_id": record.get("imageID") if isinstance(record.get("imageID"), str) else None,
        "type_of_resource": flatten_strings(record.get("typeOfResource")),
        "raw_record": record,
    }
    return normalized


def nypl_get(
    token: str,
    endpoint: str,
    params: Optional[Dict[str, Any]] = None,
    timeout: int = 60,
) -> Dict[str, Any]:
    url = f"{API_BASE}{endpoint}"
    response = requests.get(
        url,
        headers=build_headers(token),
        params=params or {},
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def search_items(
    token: str,
    query: str,
    field: Optional[str],
    page: int,
    per_page: int,
    public_domain_only: bool = True,
) -> Dict[str, Any]:
    params: Dict[str, Any] = {
        "q": query,
        "page": page,
        "per_page": per_page,
        "publicDomainOnly": str(public_domain_only).lower(),
        "filter[]": "typeOfResource:still image",
    }
    if field:
        params["field"] = field
    return nypl_get(token, "/items/search", params=params)


def search_collections(
    token: str,
    query: str,
    page: int,
    per_page: int,
    search_recursive: bool = True,
) -> Dict[str, Any]:
    params: Dict[str, Any] = {
        "q": query,
        "page": page,
        "per_page": per_page,
        "search_recursive": str(search_recursive).lower(),
    }
    return nypl_get(token, "/collections", params=params)


def collection_all_items(token: str, uuid: str) -> Dict[str, Any]:
    return nypl_get(token, f"/items/collection/all/{uuid}")


def build_query_plan(mode: str) -> List[Tuple[str, Optional[str], str]]:
    """
    Returns tuples of (query, field, target_type)
    """
    plan: List[Tuple[str, Optional[str], str]] = []

    postcard_terms = [
        "postcard",
        "postcards",
        "picture postcard",
        "greeting card",
        "holiday card",
    ]
    poster_terms = [
        "poster",
        "posters",
        "broadside",
    ]

    fields = ["genre", "topic", "title", None]

    if mode in ("postcard", "both"):
        for term in postcard_terms:
            for field in fields:
                plan.append((term, field, "postcard"))

    if mode in ("poster", "both"):
        for term in poster_terms:
            for field in fields:
                plan.append((term, field, "poster"))

    return plan


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Expand NYPL metadata collection using broader search queries and optional collection traversal."
    )
    parser.add_argument(
        "--token",
        type=str,
        default=os.getenv("NYPL_TOKEN"),
        help="NYPL API token (or set NYPL_TOKEN environment variable).",
    )
    parser.add_argument(
        "--mode",
        choices=["postcard", "poster", "both"],
        default="both",
        help="Target material type.",
    )
    parser.add_argument(
        "--pages",
        type=int,
        default=5,
        help="How many pages per item search query to fetch.",
    )
    parser.add_argument(
        "--collection-pages",
        type=int,
        default=3,
        help="How many pages per collection search query to fetch.",
    )
    parser.add_argument(
        "--per-page",
        type=int,
        default=100,
        help="per_page for paginated endpoints.",
    )
    parser.add_argument(
        "--include-collections",
        action="store_true",
        help="Also search collections and traverse them via /items/collection/all/:uuid.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.15,
        help="Sleep between requests.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/external/nypl_expanded",
        help="Where to save results.",
    )

    args = parser.parse_args()

    if not args.token:
        raise ValueError("Pass --token or set NYPL_TOKEN.")

    out_dir = Path(args.output_dir)
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    query_plan = build_query_plan(args.mode)

    seen_items: Set[str] = set()
    seen_collections: Set[str] = set()
    rows: List[Dict[str, Any]] = []

    # 1) Wider item search
    for query, field, target_type in query_plan:
        for page in range(1, args.pages + 1):
            print(f"[ITEM SEARCH] q={query!r} field={field!r} page={page}")

            try:
                payload = search_items(
                    token=args.token,
                    query=query,
                    field=field,
                    page=page,
                    per_page=args.per_page,
                    public_domain_only=True,
                )
            except Exception as e:
                print(f"  failed: {e}")
                continue

            raw_name = f"items__{target_type}__{query.replace(' ', '_')}__{field or 'all'}__page{page}.json"
            save_json(raw_dir / raw_name, payload)

            records = recursive_find_records(payload)
            print(f"  raw candidate records: {len(records)}")

            added = 0
            for record in records:
                norm = normalize_item_record(
                    record=record,
                    source_type="search",
                    query=query,
                    field=field,
                )
                if norm is None:
                    continue

                sample_id = norm["sample_id"]
                if sample_id in seen_items:
                    continue

                seen_items.add(sample_id)
                rows.append(norm)
                added += 1

            print(f"  added unique records: {added}")
            time.sleep(args.sleep)

    # 2) Optional collection search + expansion
    if args.include_collections:
        collection_terms = []
        if args.mode in ("postcard", "both"):
            collection_terms.extend(["postcard", "postcards", "picture postcard"])
        if args.mode in ("poster", "both"):
            collection_terms.extend(["poster", "posters", "broadside"])

        for query in collection_terms:
            for page in range(1, args.collection_pages + 1):
                print(f"[COLLECTION SEARCH] q={query!r} page={page}")

                try:
                    payload = search_collections(
                        token=args.token,
                        query=query,
                        page=page,
                        per_page=args.per_page,
                        search_recursive=True,
                    )
                except Exception as e:
                    print(f"  failed: {e}")
                    continue

                raw_name = f"collections__{query.replace(' ', '_')}__page{page}.json"
                save_json(raw_dir / raw_name, payload)

                collections = recursive_find_collections(payload)
                print(f"  candidate collections: {len(collections)}")

                for coll in collections:
                    coll_uuid = coll.get("uuid")
                    if not isinstance(coll_uuid, str):
                        continue
                    if coll_uuid in seen_collections:
                        continue

                    seen_collections.add(coll_uuid)
                    coll_title = extract_title(coll)

                    print(f"    traversing collection {coll_uuid} :: {coll_title}")

                    try:
                        coll_items_payload = collection_all_items(args.token, coll_uuid)
                    except Exception as e:
                        print(f"      collection traversal failed: {e}")
                        continue

                    raw_name = f"collection_all__{coll_uuid}.json"
                    save_json(raw_dir / raw_name, coll_items_payload)

                    coll_records = recursive_find_records(coll_items_payload)
                    print(f"      raw item records: {len(coll_records)}")

                    added = 0
                    for record in coll_records:
                        norm = normalize_item_record(
                            record=record,
                            source_type="collection",
                            query=query,
                            field=None,
                            collection_uuid=coll_uuid,
                            collection_title=coll_title,
                        )
                        if norm is None:
                            continue

                        sample_id = norm["sample_id"]
                        if sample_id in seen_items:
                            continue

                        seen_items.add(sample_id)
                        rows.append(norm)
                        added += 1

                    print(f"      added unique records: {added}")
                    time.sleep(args.sleep)

    manifest_path = out_dir / "nypl_manifest_expanded.jsonl"
    save_jsonl(manifest_path, rows)

    stats = {
        "num_records": len(rows),
        "num_with_images": sum(1 for r in rows if r.get("image_links")),
        "num_from_search": sum(1 for r in rows if r.get("source_type") == "search"),
        "num_from_collections": sum(1 for r in rows if r.get("source_type") == "collection"),
        "num_unique_queries": len(set((r["query"], r["field"]) for r in rows)),
        "num_unique_collections": len({r["collection_uuid"] for r in rows if r.get("collection_uuid")}),
    }
    save_json(out_dir / "stats.json", stats)

    print("\nDone.")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"Saved manifest: {manifest_path}")


if __name__ == "__main__":
    main()
