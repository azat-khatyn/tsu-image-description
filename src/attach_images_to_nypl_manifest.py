# file: src/attach_images_to_nypl_manifest.py

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests


API_BASE = "https://api.repo.nypl.org/api/v2"


def build_headers(token: str) -> Dict[str, str]:
    return {"Authorization": f'Token token="{token}"'}


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def save_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def nonempty_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        value = " ".join(value.split()).strip()
        return value if value else None
    return None


def item_details(token: str, uuid: str) -> Dict[str, Any]:
    response = requests.get(
        f"{API_BASE}/items/item_details/{uuid}",
        headers=build_headers(token),
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


def recursive_find_image_candidates(obj: Any, path: str = "$") -> List[Tuple[str, str]]:
    """
    Рекурсивно ищет image-like URLs в JSON item_details.
    Возвращает пары (json_path, url).
    """
    found: List[Tuple[str, str]] = []

    if isinstance(obj, dict):
        for k, v in obj.items():
            k_lower = k.lower()
            child_path = f"{path}.{k}"

            if isinstance(v, str):
                if v.startswith("http") and (
                    "imagelink" in k_lower
                    or "imageurl" in k_lower
                    or "highreslink" in k_lower
                    or "thumbnail" in k_lower
                    or "images.nypl.org" in v.lower()
                ):
                    found.append((child_path, v))
            else:
                found.extend(recursive_find_image_candidates(v, child_path))

    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            found.extend(recursive_find_image_candidates(item, f"{path}[{i}]"))

    seen = set()
    unique: List[Tuple[str, str]] = []
    for p, u in found:
        if u not in seen:
            seen.add(u)
            unique.append((p, u))

    return unique


def rank_image_url(url: str, prefer: str = "q") -> Tuple[int, int]:
    """
    Чем меньше rank, тем лучше.
    NYPL производные:
      w ~ 760px
      q ~ 1600px
      v ~ 2560px
      g ~ original jpeg-ish derivative
    """
    u = url.lower()

    preference_order = {
        "q": ["?t=q", "&t=q", "/q", "t=q"],
        "w": ["?t=w", "&t=w", "/w", "t=w"],
        "v": ["?t=v", "&t=v", "/v", "t=v"],
        "g": ["?t=g", "&t=g", "/g", "t=g"],
        "highres": ["highres", ".tif", ".tiff"],
    }

    order = [prefer] + [x for x in ["q", "w", "v", "g", "highres"] if x != prefer]

    for idx, bucket in enumerate(order):
        patterns = preference_order[bucket]
        if any(p in u for p in patterns):
            return idx, len(url)

    if ".jpg" in u or ".jpeg" in u or ".png" in u:
        return len(order) + 1, len(url)

    return len(order) + 2, len(url)


def choose_best_image_url(candidates: List[Tuple[str, str]], prefer: str = "q") -> Optional[str]:
    if not candidates:
        return None
    ranked = sorted(candidates, key=lambda x: rank_image_url(x[1], prefer=prefer))
    return ranked[0][1]


def download_image(url: str, out_path: Path) -> bool:
    try:
        response = requests.get(url, timeout=120)
        response.raise_for_status()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(response.content)
        return True
    except Exception:
        if out_path.exists():
            out_path.unlink(missing_ok=True)
        return False


def build_train_ready_record(row: Dict[str, Any], image_value: Optional[str]) -> Optional[Dict[str, Any]]:
    text = (
        nonempty_text(row.get("best_text"))
        or nonempty_text(row.get("abstract"))
        or nonempty_text(row.get("mods_title"))
        or nonempty_text(row.get("title"))
    )
    if not text:
        return None

    image_value = image_value or nonempty_text(row.get("chosen_image_url"))
    if not image_value:
        return None

    return {
        "sample_id": row.get("sample_id"),
        "image": image_value,
        "text": text,
        "source": "nypl",
        "source_type": row.get("source_type"),
        "query": row.get("query"),
        "field": row.get("field"),
        "collection_uuid": row.get("collection_uuid"),
        "collection_title": row.get("collection_title"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Attach real image URLs to NYPL manifest via item_details; optionally download images."
    )
    parser.add_argument(
        "--token",
        type=str,
        default=os.getenv("NYPL_TOKEN"),
        help="NYPL API token (or set NYPL_TOKEN env var)",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="data/external/nypl_expanded/nypl_manifest_expanded.jsonl",
        help="Input manifest",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/external/nypl_expanded/nypl_manifest_with_images.jsonl",
        help="Output manifest with attached image URLs",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process first N rows",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.1,
        help="Sleep between API calls",
    )
    parser.add_argument(
        "--prefer",
        choices=["q", "w", "v", "g", "highres"],
        default="q",
        help="Preferred NYPL image derivative",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download images locally",
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        default="data/external/nypl_expanded/images",
        help="Directory for downloaded images",
    )
    parser.add_argument(
        "--save-raw-details",
        action="store_true",
        help="Save raw item_details JSON",
    )
    args = parser.parse_args()

    if not args.token:
        raise ValueError("NYPL token not provided. Use --token or set NYPL_TOKEN.")

    manifest_path = Path(args.manifest)
    rows = load_jsonl(manifest_path)

    if args.limit is not None:
        rows = rows[:args.limit]

    output_path = Path(args.output)
    images_dir = Path(args.images_dir)
    raw_details_dir = output_path.parent / "raw_item_details"

    if args.save_raw_details:
        raw_details_dir.mkdir(parents=True, exist_ok=True)

    enriched_rows: List[Dict[str, Any]] = []
    train_ready_rows: List[Dict[str, Any]] = []

    num_with_candidates = 0
    num_with_chosen = 0
    num_downloaded = 0
    num_errors = 0

    for i, row in enumerate(rows, start=1):
        uuid = row.get("sample_id")
        print(f"[{i}/{len(rows)}] Processing {uuid}")

        enriched = dict(row)

        try:
            payload = item_details(args.token, uuid)

            if args.save_raw_details:
                save_json(raw_details_dir / f"{uuid}.json", payload)

            candidates = recursive_find_image_candidates(payload)
            urls = [u for _, u in candidates]
            chosen = choose_best_image_url(candidates, prefer=args.prefer)

            enriched["item_details_image_links"] = urls
            enriched["num_item_details_image_links"] = len(urls)
            enriched["chosen_image_url"] = chosen

            if urls:
                num_with_candidates += 1
            if chosen:
                num_with_chosen += 1

            downloaded_path: Optional[str] = None
            if args.download and chosen:
                ext = ".jpg"
                low = chosen.lower()
                if ".png" in low:
                    ext = ".png"
                elif ".tif" in low or ".tiff" in low:
                    ext = ".tif"

                out_path = images_dir / f"{uuid}{ext}"
                ok = download_image(chosen, out_path)
                if ok:
                    downloaded_path = str(out_path.resolve())
                    enriched["downloaded_image_path"] = downloaded_path
                    num_downloaded += 1
                else:
                    enriched["download_error"] = True

            train_row = build_train_ready_record(enriched, image_value=downloaded_path)
            if train_row is not None:
                train_ready_rows.append(train_row)

        except Exception as e:
            enriched["image_attach_error"] = str(e)
            num_errors += 1

        enriched_rows.append(enriched)
        time.sleep(args.sleep)

    save_jsonl(output_path, enriched_rows)

    train_ready_path = output_path.parent / "nypl_train_ready.jsonl"
    save_jsonl(train_ready_path, train_ready_rows)

    stats = {
        "num_rows": len(enriched_rows),
        "num_with_item_details_image_candidates": num_with_candidates,
        "num_with_chosen_image_url": num_with_chosen,
        "num_downloaded": num_downloaded,
        "num_train_ready_rows": len(train_ready_rows),
        "num_errors": num_errors,
        "share_with_chosen_image_url": round(num_with_chosen / len(enriched_rows), 4) if enriched_rows else 0.0,
    }
    save_json(output_path.parent / "image_attach_stats.json", stats)

    print("\nDone.")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"Saved manifest with images: {output_path}")
    print(f"Saved train-ready jsonl: {train_ready_path}")


if __name__ == "__main__":
    main()
