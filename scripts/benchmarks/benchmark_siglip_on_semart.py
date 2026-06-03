"""benchmark_siglip_on_semart.py - измеряет точность MetadataExtractor
на размеченном SemArt (19 244 картин с каноническими метками TYPE/TECHNIQUE).

SemArt служит ground-truth для нашего zero-shot классификатора:
сравниваем top-1 prediction по осям theme/style с метаданными SemArt.

Маппинг SemArt → archival_v2:
  TYPE → ось themes
    religious   → "a religious subject"
    portrait    → "a portrait"
    landscape   → "a landscape"
    genre       → "a genre scene"
    still-life  → "a still life"
    (mythological/historical/interior/other/study - пропускается, нет однозначного маппинга)
  TECHNIQUE → ось styles
    "Oil on ..." → "an oil painting"
    "Watercolour" → "a watercolor painting"
    "Etching"     → "an etching"
    "Engraving"   → "an engraving"
    "Pencil/Charcoal/..." → "a pencil drawing"
    (fresco / tempera / panel — пропускаем, нет в нашей таксономии)

 99% сопоставленной техники = "oil painting", бенчмарк по style
информативен только как 1-vs-rest recall. Основной сигнал: theme accuracy на 5 классах.

Выход: JSON с
  - accuracy по каждой оси, macro precision/recall, confusion matrix
  - per-item prediction для разбора ошибок

Использование:
    PYTHONPATH=src python scripts/benchmark_siglip_on_semart.py \\
      --split val \\
      --taxonomy archival_v2 \\
      --output data/semart/benchmark_archival_v2_val.json

    # Опционально: полный train-сплит (~3-5 часов на M1)
    PYTHONPATH=src python scripts/benchmark_siglip_on_semart.py \\
      --split train --max-items 5000 \\
      --output data/semart/benchmark_archival_v2_train5k.json
"""

import argparse
import csv
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SEMART_DIR = ROOT / "data" / "semart" / "SemArt"
IMAGES_DIR = SEMART_DIR / "Images"


# ----------------------------------------------------------------------------
# Маппинг SemArt → taxonomy
# ----------------------------------------------------------------------------

THEME_MAP_ARCHIVAL_V2 = {
    "religious":  "a religious subject",
    "portrait":   "a portrait",
    "landscape":  "a landscape",
    "genre":      "a genre scene",
    "still-life": "a still life",
}

THEME_MAP_LEGACY_V1 = {
    "religious": "religious scene",
    "landscape": "nature scene",
    # прочие SemArt TYPE не имеют однозначного маппинга на themes из legacy_v1
}


def normalize_technique(tech: str) -> str:
    """Убирает размеры и приводит к нижнему регистру."""
    return (tech or "").strip().lower().split(",")[0].strip()


def map_style_archival_v2(tech_normalized: str):
    """Сопоставляет нормализованный SemArt TECHNIQUE → метка style из archival_v2.

    Возвращает None, если чистого маппинга нет (fresco, tempera, panel без явного медиума).
    """
    t = tech_normalized
    if "oil on" in t or t == "oil":
        return "an oil painting"
    if "tempera" in t or "fresco" in t:
        return None
    if "watercol" in t or "gouache" in t:
        return "a watercolor painting"
    if "etching" in t:
        return "an etching"
    if "engraving" in t:
        return "an engraving"
    if "lithograph" in t or "chromolith" in t:
        return "a chromolithograph"
    if "pencil" in t or "charcoal" in t or "pen and ink" in t or "crayon" in t:
        return "a pencil drawing"
    return None


def map_style_legacy_v1(tech_normalized: str):
    t = tech_normalized
    if "oil on" in t or t == "oil" or "tempera" in t:
        return "painting"
    if "fresco" in t:
        return "painting"
    if "etching" in t or "engraving" in t:
        return "engraving"
    if "pencil" in t or "charcoal" in t or "crayon" in t:
        return "drawing"
    return None


# ----------------------------------------------------------------------------
# Бенчмарк
# ----------------------------------------------------------------------------

def build_eval_set(csv_path: Path, taxonomy: str, max_items=None):
    """Читает SemArt CSV → список словарей {image_path, gt_theme, gt_style}."""
    rows = []
    with open(csv_path, encoding="latin-1") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for r in reader:
            img_file = r.get("IMAGE_FILE", "").strip()
            if not img_file:
                continue
            img_path = IMAGES_DIR / img_file
            if not img_path.exists():
                continue

            semart_type = (r.get("TYPE") or "").strip().lower()
            semart_tech = normalize_technique(r.get("TECHNIQUE", ""))

            if taxonomy == "archival_v2":
                gt_theme = THEME_MAP_ARCHIVAL_V2.get(semart_type)
                gt_style = map_style_archival_v2(semart_tech)
            else:  # legacy_v1
                gt_theme = THEME_MAP_LEGACY_V1.get(semart_type)
                gt_style = map_style_legacy_v1(semart_tech)

            # пропускаем строки, где не маппится НИ ОДНА ось
            if gt_theme is None and gt_style is None:
                continue

            rows.append({
                "image_path": str(img_path),
                "image_file": img_file,
                "semart_type": semart_type,
                "semart_technique": semart_tech,
                "gt_theme": gt_theme,
                "gt_style": gt_style,
                "title": (r.get("TITLE") or "").strip(),
                "author": (r.get("AUTHOR") or "").strip(),
                "timeframe": (r.get("TIMEFRAME") or "").strip(),
            })
            if max_items and len(rows) >= max_items:
                break
    return rows


def aggregate_axis(predictions, gt_key, pred_key, classes):
    """Считает accuracy + P/R по классам + confusion matrix для одной оси."""
    relevant = [p for p in predictions if p[gt_key] is not None]
    n = len(relevant)
    if n == 0:
        return {"n": 0}

    correct = sum(1 for p in relevant if p[pred_key] == p[gt_key])
    accuracy = correct / n

    # precision / recall по классам
    tp = Counter()
    fp = Counter()
    fn = Counter()
    for p in relevant:
        gt = p[gt_key]
        pr = p[pred_key]
        if pr == gt:
            tp[gt] += 1
        else:
            fp[pr] += 1
            fn[gt] += 1

    per_class = {}
    for c in classes:
        gt_count = sum(1 for p in relevant if p[gt_key] == c)
        precision = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) > 0 else 0.0
        recall = tp[c] / gt_count if gt_count > 0 else 0.0
        per_class[c] = {
            "support": gt_count,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(2 * precision * recall / (precision + recall), 4)
                if (precision + recall) > 0 else 0.0,
        }

    macro_p = sum(per_class[c]["precision"] for c in classes) / len(classes)
    macro_r = sum(per_class[c]["recall"] for c in classes) / len(classes)
    macro_f = sum(per_class[c]["f1"] for c in classes) / len(classes)

    # confusion matrix
    conf = defaultdict(lambda: defaultdict(int))
    for p in relevant:
        conf[p[gt_key]][p[pred_key]] += 1
    conf_clean = {gt: dict(preds) for gt, preds in conf.items()}

    return {
        "n": n,
        "accuracy": round(accuracy, 4),
        "macro_precision": round(macro_p, 4),
        "macro_recall": round(macro_r, 4),
        "macro_f1": round(macro_f, 4),
        "per_class": per_class,
        "confusion": conf_clean,
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=["train", "val"], default="val",
                   help="SemArt split (val=1069 paintings, train=19244)")
    p.add_argument("--taxonomy", choices=["archival_v2", "legacy_v1"], default="archival_v2")
    p.add_argument("--max-items", type=int, default=None, help="Cap evaluation set size")
    p.add_argument("--output", required=True)
    p.add_argument("--log-every", type=int, default=50)
    return p.parse_args()


def main():
    args = parse_args()
    csv_name = f"semart_{args.split}.csv"
    csv_path = SEMART_DIR / csv_name
    print(f"[1/3] Loading {csv_name} (taxonomy={args.taxonomy})")
    eval_set = build_eval_set(csv_path, args.taxonomy, max_items=args.max_items)
    print(f"      Evaluable rows: {len(eval_set)}")

    # краткая статистика по распределению меток
    type_dist = Counter(r["gt_theme"] for r in eval_set if r["gt_theme"])
    style_dist = Counter(r["gt_style"] for r in eval_set if r["gt_style"])
    print(f"      Theme gt distribution: {dict(type_dist)}")
    print(f"      Style gt distribution: {dict(style_dist)}")
    print()

    print(f"[2/3] Loading MetadataExtractor (taxonomy={args.taxonomy})")
    sys.path.insert(0, str(ROOT / "src"))
    from tsu_image_description.metadata_extractor import MetadataExtractor
    extractor = MetadataExtractor(taxonomy_version=args.taxonomy)
    print()

    print(f"[3/3] Running predictions on {len(eval_set)} images")
    t_start = time.time()
    predictions = []
    for i, row in enumerate(eval_set):
        try:
            result = extractor.extract(row["image_path"])
            pred_theme = result["theme"]["label"]
            pred_style = result["style"]["label"]
            theme_score = result["theme"]["score"]
            style_score = result["style"]["score"]
            theme_margin = result["theme"]["margin"]
            style_margin = result["style"]["margin"]
        except Exception as e:
            print(f"  [{i+1}] ERROR on {row['image_file']}: {e}")
            pred_theme = pred_style = None
            theme_score = style_score = theme_margin = style_margin = 0.0

        predictions.append({
            **row,
            "pred_theme": pred_theme,
            "pred_style": pred_style,
            "theme_score": theme_score,
            "style_score": style_score,
            "theme_margin": theme_margin,
            "style_margin": style_margin,
        })

        if (i + 1) % args.log_every == 0:
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed
            eta = (len(eval_set) - (i + 1)) / rate
            theme_acc = sum(1 for p in predictions
                            if p["gt_theme"] and p["pred_theme"] == p["gt_theme"]) \
                       / max(1, sum(1 for p in predictions if p["gt_theme"]))
            print(f"  [{i+1}/{len(eval_set)}] {rate:.2f} img/sec  ETA {eta/60:.1f} min  "
                  f"running theme_acc={theme_acc:.3f}")

    total_time = time.time() - t_start
    print(f"\n      Done in {total_time/60:.1f} min ({len(eval_set)/total_time:.2f} img/sec)")
    print()

    # агрегируем
    theme_classes = list(set(p["gt_theme"] for p in predictions if p["gt_theme"]))
    style_classes = list(set(p["gt_style"] for p in predictions if p["gt_style"]))

    theme_metrics = aggregate_axis(predictions, "gt_theme", "pred_theme", theme_classes)
    style_metrics = aggregate_axis(predictions, "gt_style", "pred_style", style_classes)

    print("=== THEME (axis) ===")
    print(json.dumps({k: v for k, v in theme_metrics.items() if k != "confusion"},
                     indent=2, ensure_ascii=False))
    print("\n=== STYLE (axis) ===")
    print(json.dumps({k: v for k, v in style_metrics.items() if k != "confusion"},
                     indent=2, ensure_ascii=False))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "taxonomy_version": args.taxonomy,
            "split": args.split,
            "n_evaluated": len(eval_set),
            "elapsed_sec": round(total_time, 1),
            "theme_metrics": theme_metrics,
            "style_metrics": style_metrics,
            "per_item": predictions,
        }, f, indent=2, ensure_ascii=False)
    print(f"\n[INFO] Saved to {out_path}")


if __name__ == "__main__":
    main()
