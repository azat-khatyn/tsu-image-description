"""build_testset.py — собрать каноничный манифест тестового набора.

Единый источник истины: data/eval/testset.jsonl. Одна строка на изображение:
    {"image_path", "source", "role", "reference_ru", "technique", "has_text"}

role:
  - reference_based : есть экспертный эталон (НЭБ колл. 529) -> ref-based метрики
  - reference_free  : эталона нет -> CLIPScore / retrieval / эмбеддинги
  - demo            : малый набор для качественной демонстрации версий пайплайна

Страты собираются глобом фактических папок + слиянием с экспертной разметкой,
поэтому манифест всегда отражает то, что реально лежит на диске. Картинки
gitignored; трекается сам манифест.

Использование:
    python scripts/build_testset.py
"""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _exists(p):
    return Path(p).is_file()


def neb_wwii(refs_path):
    """НЭБ колл. 529 «Ленинград ВОВ» — экспертная RUSMARC-разметка."""
    rows = []
    for line in open(refs_path, encoding="utf-8"):
        r = json.loads(line)
        ip = r["image_path"]
        if not ip.endswith(".jpg"):
            ip += ".jpg"
        if not _exists(ip):
            continue
        rows.append({
            "image_path": ip, "source": "neb_wwii", "role": "reference_based",
            "reference_ru": r.get("reference_short_ru"),
            "technique": r.get("technique"), "has_text": None,
        })
    return rows


def glob_dir(folder, source, role):
    rows = []
    paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        paths += Path(folder).glob(ext)
    for p in sorted(paths):
        rows.append({
            "image_path": str(p), "source": source, "role": role,
            "reference_ru": None, "technique": None, "has_text": None,
        })
    return rows


def semantic_demo(path):
    rows = []
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        ip = r["image_path"]
        if not _exists(ip):
            continue
        ref = r.get("reference_short_ru")
        rows.append({
            "image_path": ip, "source": "semantic_demo", "role": "demo",
            "reference_ru": ref if ref and ref != "None" else None,
            "technique": None, "has_text": None,
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/eval/testset.jsonl")
    args = ap.parse_args()

    rows = []
    rows += neb_wwii("data/eval/references_neb_n224.jsonl")
    rows += glob_dir("data/neb_search_postcards/images", "neb_diverse", "reference_free")
    rows += glob_dir("data/eval/images NYPL based", "nypl_curated", "reference_free")
    rows += semantic_demo("data/eval/semantic_testset.jsonl")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    from collections import Counter
    by_src = Counter(r["source"] for r in rows)
    by_role = Counter(r["role"] for r in rows)
    print(f"[testset] {len(rows)} изображений -> {out}")
    print("  по источникам:", dict(by_src))
    print("  по ролям:", dict(by_role))


if __name__ == "__main__":
    main()
