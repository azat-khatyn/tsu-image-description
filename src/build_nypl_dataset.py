# file: src/build_nypl_dataset.py

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests
from PIL import Image
from requests.structures import CaseInsensitiveDict
from tqdm import tqdm


API_BASE = "https://api.repo.nypl.org/api/v2"


def make_headers(token: str) -> CaseInsensitiveDict:
    headers = CaseInsensitiveDict()
    headers["Authorization"] = f'Token token="{token}"'
    return headers


def safe_get(d: Any, path: List[str], default=None):
    cur = d
    for key in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key)
        if cur is None:
            return default
    return cur


def flatten_strings(value: Any) -> List[str]:
    out: List[str] = []
    if value is None:
        return out
    if isinstance(value, str):
        s = value.strip()
        if s:
            out.append(s)
        return out
    if isinstance(value, list):
        for item in value:
            out.extend(flatten_strings(item))
    elif isinstance(value, dict):
        for v in value.values():
            out.extend(flatten_strings(v))
    return out


def recursive_find_urls(obj: Any) -> List[str]:
    urls: List[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, str) and v.startswith("http"):
                if any(tag in k.lower() for tag in ["image", "url", "link", "preview"]):
                    urls.append(v)
            else:
                urls.extend(recursive_find_urls(v))
    elif isinstance(obj, list):
        for item in obj:
            urls.extend(recursive_find_urls(item))
    return urls


def recursive_find_caption_candidates(obj: Any) -> List[str]:
    wanted_keys = {
        "title",
        "description",
        "abstract",
        "note",
        "summary",
        "label",
        "name",
    }
    vals: List[str] = []

    if isinstance(obj, dict):
        for k, v in obj.items():
            if k.lower() in wanted_keys:
                vals.extend(flatten_strings(v))
            vals.extend(recursive_find_caption_candidates(v))
    elif isinstance(obj, list):
        for item in obj:
            vals.extend(recursive_find_caption_candidates(item))
    return vals


def pick_best_caption(payload: Dict[str, Any]) -> Optional[str]:
    candidates = recursive_find_caption_candidates(payload)

    # Фильтрация мусора и очень коротких значений
    cleaned = []
    for c in candidates:
        c = re.sub(r"\s+", " ", c).strip(" -;\n\t")
        if len(c) >= 12:
            cleaned.append(c)

    if not cleaned:
        return None

    # Предпочитаем более описательные строки
    cleaned = sorted(cleaned, key=lambda x: (-len(x), x))
    return cleaned[0]


def pick_image_url(payload: Dict[str, Any]) -> Optional[str]:
    urls = recursive_find_urls(payload)
    if not urls:
        return None

    # Предпочтение JPEG/PNG, избегаем явно служебных урлов
    ranked = sorted(
        set(urls),
        key=lambda u: (
            0 if any(ext in u.lower() for ext in [".jpg", ".jpeg", ".png"]) else 1,
            len(u),
        ),
    )
    return ranked[0]


def search_items(
    token: str,
    query: str,
    page: int = 1,
    per_page: int = 50,
    public_domain_only: bool = True,
) -> Dict[str, Any]:
    params = {
        "q": query,
        "publicDomainOnly": str(public_domain_only).lower(),
        "page": page,
        "per_page": per_page,
    }
    resp = requests.get(
        f"{API_BASE}/items/search",
        headers=make_headers(token),
        params=params,
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()


def item_details(token: str, uuid: str) -> Dict[str, Any]:
    resp = requests.get(
        f"{API_BASE}/items/item_details/{uuid}",
        headers=make_headers(token),
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()


def recursive_find_uuids(obj: Any) -> List[str]:
    uuids: List[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k.lower() == "uuid" and isinstance(v, str):
                uuids.append(v)
            else:
                uuids.extend(recursive_find_uuids(v))
    elif isinstance(obj, list):
        for item in obj:
            uuids.extend(recursive_find_uuids(item))
    return uuids


def extract_search_result_uuids(payload: Dict[str, Any]) -> List[str]:
    uuids = recursive_find_uuids(payload)
    # Убираем дубликаты, сохраняя порядок
    seen = set()
    ordered = []
    for u in uuids:
        if u not in seen:
            seen.add(u)
            ordered.append(u)
    return ordered


def download_image(url: str, out_path: Path) -> bool:
    try:
        resp = requests.get(url, timeout=90)
        resp.raise_for_status()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(resp.content)

        # быстрая валидация
        with Image.open(out_path) as img:
            img.verify()
        return True
    except Exception:
        if out_path.exists():
            out_path.unlink(missing_ok=True)
        return False


def build_dataset(
    token: str,
    query: str,
    max_items: int,
    out_dir: Path,
    sleep_sec: float = 0.15,
) -> Tuple[int, int]:
    images_dir = out_dir / "images"
    manifest_path = out_dir / "nypl_manifest.jsonl"
    raw_dir = out_dir / "raw_json"
    raw_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    scanned = 0
    page = 1
    per_page = 50

    with manifest_path.open("w", encoding="utf-8") as manifest_f:
        while saved < max_items:
            search_payload = search_items(
                token=token,
                query=query,
                page=page,
                per_page=per_page,
                public_domain_only=True,
            )

            page_uuids = extract_search_result_uuids(search_payload)
            if not page_uuids:
                break

            for uuid in tqdm(page_uuids, desc=f"page {page}"):
                if saved >= max_items:
                    break

                scanned += 1

                try:
                    details = item_details(token, uuid)
                except Exception:
                    continue

                (raw_dir / f"{uuid}.json").write_text(
                    json.dumps(details, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

                image_url = pick_image_url(details)
                caption = pick_best_caption(details)

                if not image_url or not caption:
                    time.sleep(sleep_sec)
                    continue

                image_name = f"{uuid}.jpg"
                image_path = images_dir / image_name

                ok = download_image(image_url, image_path)
                if not ok:
                    time.sleep(sleep_sec)
                    continue

                record = {
                    "sample_id": uuid,
                    "image": str(image_path.resolve()),
                    "text": caption,
                    "source": "nypl",
                    "query": query,
                    "image_url": image_url,
                    "raw_json": str((raw_dir / f"{uuid}.json").resolve()),
                }
                manifest_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                saved += 1
                time.sleep(sleep_sec)

            page += 1

    return saved, scanned


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", type=str, default=os.getenv("NYPL_TOKEN"))
    parser.add_argument("--query", type=str, default="postcard")
    parser.add_argument("--max-items", type=int, default=200)
    parser.add_argument("--out-dir", type=str, default="data/external/nypl")
    args = parser.parse_args()

    if not args.token:
        raise ValueError(
            "NYPL token not provided. Pass --token or set NYPL_TOKEN environment variable."
        )

    out_dir = Path(args.out_dir)
    saved, scanned = build_dataset(
        token=args.token,
        query=args.query,
        max_items=args.max_items,
        out_dir=out_dir,
    )
    print(f"Done. Saved {saved} usable items out of {scanned} scanned.")


if __name__ == "__main__":
    main()

