"""build_analysis_manifest.py — манифест для анализа эмбеддингов.

Derive от каноничного тест-набора (data/eval/testset.jsonl) + SemArt как внешний
доменный референс (только для фигуры доменного сдвига). SemArt в тест-набор и
метрики НЕ входит.

Выход под схему embed_export: {image_path, source, theme, technique, text_ru}.

Использование:
    python scripts/build_analysis_manifest.py --semart-per-type 30 --seed 42
"""

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SEMART_CSV = ROOT / "data" / "semart" / "SemArt" / "semart_train.csv"
SEMART_IMAGES = ROOT / "data" / "semart" / "SemArt" / "Images"


def testset_rows(path):
    """Каноничный тест-набор -> схема embed_export (text_ru из reference_ru)."""
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        yield {
            "image_path": r["image_path"], "source": r["source"],
            "theme": None, "technique": r.get("technique"),
            "text_ru": r.get("reference_ru"),
        }


def semart_rows(per_type, seed):
    """Стратифицированная выборка SemArt по TYPE (внешний доменный референс)."""
    by_type = defaultdict(list)
    with open(SEMART_CSV, encoding="latin-1") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            t = (row.get("TYPE") or "").strip()
            img = SEMART_IMAGES / row["IMAGE_FILE"]
            if t and img.is_file():
                by_type[t].append((row["IMAGE_FILE"], row.get("TECHNIQUE", "")))
    rng = random.Random(seed)
    for t, items in sorted(by_type.items()):
        for fname, tech in rng.sample(items, min(per_type, len(items))):
            yield {
                "image_path": str(SEMART_IMAGES / fname), "source": "semart",
                "theme": t, "technique": (tech or None), "text_ru": None,
            }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--testset", default="data/eval/testset.jsonl")
    ap.add_argument("--semart-per-type", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/eval/embeddings/analysis_manifest.jsonl")
    args = ap.parse_args()

    rows = list(testset_rows(args.testset)) + list(semart_rows(args.semart_per_type, args.seed))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    by_src = Counter(r["source"] for r in rows)
    print(f"[manifest] {len(rows)} записей -> {out}")
    for s, n in sorted(by_src.items()):
        print(f"   {s}: {n}")


if __name__ == "__main__":
    main()
