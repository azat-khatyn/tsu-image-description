# file: src/enrich_nypl_from_apiuri.py

from __future__ import annotations

import argparse
import json
import os
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests


MODS_NS = {
    "mods": "http://www.loc.gov/mods/v3",
}


def build_headers(token: str) -> Dict[str, str]:
    return {"Authorization": f'Token token="{token}"'}


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def clean_text(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    s = " ".join(str(s).split())
    return s if s else None


def unique_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def extract_texts(root: ET.Element, xpath: str) -> List[str]:
    result = []
    for el in root.findall(xpath, MODS_NS):
        if el.text:
            text = clean_text(el.text)
            if text:
                result.append(text)
    return unique_keep_order(result)


def extract_nested_title(root: ET.Element) -> List[str]:
    titles = []

    for title_info in root.findall(".//mods:titleInfo", MODS_NS):
        title_el = title_info.find("mods:title", MODS_NS)
        subtitle_el = title_info.find("mods:subTitle", MODS_NS)

        title_text = clean_text(title_el.text) if title_el is not None and title_el.text else None
        subtitle_text = clean_text(subtitle_el.text) if subtitle_el is not None and subtitle_el.text else None

        if title_text and subtitle_text:
            titles.append(f"{title_text}: {subtitle_text}")
        elif title_text:
            titles.append(title_text)

    return unique_keep_order(titles)


def extract_subjects_xml(root: ET.Element) -> List[str]:
    subjects = []

    for subj in root.findall(".//mods:subject", MODS_NS):
        parts = []

        for child_xpath in [
            "mods:topic",
            "mods:geographic",
            "mods:temporal",
            "mods:name/mods:namePart",
            "mods:occupation",
        ]:
            for el in subj.findall(child_xpath, MODS_NS):
                if el.text:
                    text = clean_text(el.text)
                    if text:
                        parts.append(text)

        if parts:
            subjects.append(" | ".join(parts))

    return unique_keep_order(subjects)


def extract_notes_xml(root: ET.Element) -> List[str]:
    notes = []

    for note in root.findall(".//mods:note", MODS_NS):
        if note.text:
            text = clean_text(note.text)
            if text:
                note_type = note.attrib.get("type")
                if note_type:
                    notes.append(f"[{note_type}] {text}")
                else:
                    notes.append(text)

    return unique_keep_order(notes)


def extract_origin_dates_xml(root: ET.Element) -> List[str]:
    dates = []

    for xpath in [
        ".//mods:originInfo/mods:dateIssued",
        ".//mods:originInfo/mods:dateCreated",
        ".//mods:originInfo/mods:copyrightDate",
        ".//mods:part/mods:date",
    ]:
        dates.extend(extract_texts(root, xpath))

    return unique_keep_order(dates)


def parse_mods_xml(xml_text: str) -> Dict[str, Any]:
    root = ET.fromstring(xml_text)

    titles = extract_nested_title(root)
    abstracts = extract_texts(root, ".//mods:abstract")
    genres = extract_texts(root, ".//mods:genre")
    subjects = extract_subjects_xml(root)
    notes = extract_notes_xml(root)
    origin_dates = extract_origin_dates_xml(root)
    forms = extract_texts(root, ".//mods:physicalDescription/mods:form")

    return {
        "mods_title": titles[0] if titles else None,
        "mods_titles_all": titles,
        "abstract": abstracts[0] if abstracts else None,
        "abstracts_all": abstracts,
        "genres_mods": genres,
        "subjects_mods": subjects,
        "notes_mods": notes,
        "forms_mods": forms,
        "origin_dates_mods": origin_dates,
        "mods_parse_mode": "xml",
    }


def collect_values_by_key(obj: Any, target_keys: set[str]) -> List[str]:
    values: List[str] = []

    if isinstance(obj, dict):
        for k, v in obj.items():
            if k.lower() in target_keys:
                values.extend(flatten_text_values(v))
            values.extend(collect_values_by_key(v, target_keys))
    elif isinstance(obj, list):
        for item in obj:
            values.extend(collect_values_by_key(item, target_keys))

    return unique_keep_order(values)


def flatten_text_values(obj: Any) -> List[str]:
    values: List[str] = []

    if obj is None:
        return values

    if isinstance(obj, str):
        text = clean_text(obj)
        if text:
            values.append(text)
    elif isinstance(obj, (int, float)):
        values.append(str(obj))
    elif isinstance(obj, list):
        for item in obj:
            values.extend(flatten_text_values(item))
    elif isinstance(obj, dict):
        for v in obj.values():
            values.extend(flatten_text_values(v))

    return unique_keep_order(values)


def parse_mods_json(data: Dict[str, Any]) -> Dict[str, Any]:
    titles = collect_values_by_key(data, {"title", "titleinfo"})
    abstracts = collect_values_by_key(data, {"abstract", "description", "summary"})
    genres = collect_values_by_key(data, {"genre"})
    subjects = collect_values_by_key(data, {"subject", "topic", "geographic", "temporal"})
    notes = collect_values_by_key(data, {"note"})
    forms = collect_values_by_key(data, {"form"})
    origin_dates = collect_values_by_key(
        data,
        {"dateissued", "datecreated", "copyrightdate", "date"},
    )

    return {
        "mods_title": titles[0] if titles else None,
        "mods_titles_all": titles,
        "abstract": abstracts[0] if abstracts else None,
        "abstracts_all": abstracts,
        "genres_mods": genres,
        "subjects_mods": subjects,
        "notes_mods": notes,
        "forms_mods": forms,
        "origin_dates_mods": origin_dates,
        "mods_parse_mode": "json",
    }


def fetch_api_uri(api_uri: str, token: str, timeout: int = 60) -> Tuple[str, str, int]:
    response = requests.get(
        api_uri,
        headers=build_headers(token),
        timeout=timeout,
    )
    response.raise_for_status()
    content_type = response.headers.get("Content-Type", "")
    return response.text, content_type, response.status_code


def parse_api_response(text: str, content_type: str) -> Dict[str, Any]:
    stripped = text.lstrip()

    # Сначала пробуем XML
    if stripped.startswith("<"):
        return parse_mods_xml(text)

    # Потом JSON
    if stripped.startswith("{") or stripped.startswith("["):
        data = json.loads(text)
        if isinstance(data, dict):
            return parse_mods_json(data)
        return parse_mods_json({"root": data})

    # Иногда content-type подсказывает JSON/XML даже если body начинается странно
    ctype = content_type.lower()
    if "json" in ctype:
        data = json.loads(text)
        if isinstance(data, dict):
            return parse_mods_json(data)
        return parse_mods_json({"root": data})

    if "xml" in ctype:
        return parse_mods_xml(text)

    preview = stripped[:200].replace("\n", " ")
    raise ValueError(f"Unknown response format. Content-Type={content_type!r}, preview={preview!r}")


def enrich_rows(
    rows: List[Dict[str, Any]],
    token: str,
    sleep_sec: float = 0.15,
    limit: Optional[int] = None,
    save_raw_dir: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    enriched_rows = []

    if limit is not None:
        rows = rows[:limit]

    for i, row in enumerate(rows, start=1):
        raw_record = row.get("raw_record", {})
        api_uri = raw_record.get("apiUri")
        sample_id = row.get("sample_id")

        print(f"[{i}/{len(rows)}] Processing {sample_id}")

        enriched = dict(row)

        if not api_uri:
            enriched["mods_error"] = "apiUri missing"
            enriched_rows.append(enriched)
            continue

        try:
            body_text, content_type, status_code = fetch_api_uri(api_uri, token=token)

            enriched["api_uri_content_type"] = content_type
            enriched["api_uri_status_code"] = status_code

            if save_raw_dir is not None:
                save_raw_dir.mkdir(parents=True, exist_ok=True)

                # сохраняем как txt, потому что заранее не знаем, xml это или json или html
                raw_path = save_raw_dir / f"{sample_id}.txt"
                raw_path.write_text(body_text, encoding="utf-8")
                enriched["mods_raw_path"] = str(raw_path)

            parsed = parse_api_response(body_text, content_type)
            enriched.update(parsed)

            # Предпочитаем abstract, потом mods_title, потом title из исходного манифеста
            best_text = parsed.get("abstract") or parsed.get("mods_title") or row.get("title")
            enriched["best_text"] = best_text

        except Exception as e:
            enriched["mods_error"] = str(e)

            # сохраняем preview для быстрой диагностики
            try:
                preview = body_text[:300]
            except Exception:
                preview = None

            if preview:
                enriched["mods_error_preview"] = preview

        enriched_rows.append(enriched)
        time.sleep(sleep_sec)

    return enriched_rows


def main():
    parser = argparse.ArgumentParser(
        description="Enrich NYPL manifest with metadata from apiUri (XML/JSON tolerant)."
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
        default="data/external/nypl/nypl_manifest.jsonl",
        help="Path to NYPL manifest jsonl",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/external/nypl/nypl_manifest_enriched.jsonl",
        help="Path to save enriched manifest",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only first N rows",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.15,
        help="Sleep between requests",
    )
    parser.add_argument(
        "--save-raw",
        action="store_true",
        help="Save raw apiUri responses",
    )

    args = parser.parse_args()

    if not args.token:
        raise ValueError(
            "NYPL token not provided. Use --token or set NYPL_TOKEN environment variable."
        )

    manifest_path = Path(args.manifest)
    output_path = Path(args.output)

    rows = load_jsonl(manifest_path)

    raw_dir = None
    if args.save_raw:
        raw_dir = output_path.parent / "raw_apiuri"

    enriched_rows = enrich_rows(
        rows=rows,
        token=args.token,
        sleep_sec=args.sleep,
        limit=args.limit,
        save_raw_dir=raw_dir,
    )

    save_jsonl(output_path, enriched_rows)

    num_rows = len(enriched_rows)
    num_with_abstract = sum(1 for r in enriched_rows if r.get("abstract"))
    num_with_best_text = sum(1 for r in enriched_rows if r.get("best_text"))
    num_with_subjects = sum(1 for r in enriched_rows if r.get("subjects_mods"))
    num_with_genres = sum(1 for r in enriched_rows if r.get("genres_mods"))
    num_errors = sum(1 for r in enriched_rows if r.get("mods_error"))
    parse_modes = {}
    for row in enriched_rows:
        mode = row.get("mods_parse_mode")
        if mode:
            parse_modes[mode] = parse_modes.get(mode, 0) + 1

    stats = {
        "num_rows": num_rows,
        "num_with_abstract": num_with_abstract,
        "num_with_best_text": num_with_best_text,
        "num_with_subjects": num_with_subjects,
        "num_with_genres": num_with_genres,
        "num_errors": num_errors,
        "share_with_abstract": round(num_with_abstract / num_rows, 4) if num_rows else 0.0,
        "share_with_best_text": round(num_with_best_text / num_rows, 4) if num_rows else 0.0,
        "parse_modes": parse_modes,
    }

    stats_path = output_path.parent / "stats_mods_enriched.json"
    save_json(stats_path, stats)

    print("\nDone.")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"Saved enriched manifest: {output_path}")
    print(f"Saved stats: {stats_path}")


if __name__ == "__main__":
    main()
