"""siglip_theme_probe.py - supervised linear классификатор на замороженных SigLIP features.

Альтернатива zero-shot prompt'ам: учим линейный классификатор поверх замороженных
SigLIP image embeddings, используя SemArt TYPE как разметку.

Сравнение делается с zero-shot baseline из benchmark_siglip_on_semart.py

Pipeline:
  1. (кэш) извлекаем SigLIP image features для всего SemArt train+val
  2. оставляем строки с TYPE (5 классов)
  3. обучаем sklearn LogisticRegression (class_weight='balanced')
  4. оцениваем на val: accuracy, P/R, F1, confusion matrix
  5. сравниваем с zero-shot baseline

Кэш: data/semart/siglip_features_{train,val}.npz
Результаты: data/semart/probe_theme_archival_v2.json

Использование:
    PYTHONPATH=src python scripts/siglip_theme_probe.py \\
      --output data/semart/probe_theme_archival_v2.json
"""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

from tsu_image_description.models import get_device

ROOT = Path(__file__).resolve().parent.parent
SEMART_DIR = ROOT / "data" / "semart" / "SemArt"
IMAGES_DIR = SEMART_DIR / "Images"
CACHE_DIR = ROOT / "data" / "semart"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

THEME_CLASSES = ["religious", "portrait", "landscape", "genre", "still-life"]


def load_rows(csv_path: Path):
    """Читает SemArt CSV → список словарей только со строками, где TYPE маппится."""
    out = []
    with open(csv_path, encoding="latin-1") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for r in reader:
            img_file = r.get("IMAGE_FILE", "").strip()
            if not img_file:
                continue
            img_path = IMAGES_DIR / img_file
            if not img_path.exists():
                continue
            t = (r.get("TYPE") or "").strip().lower()
            if t not in THEME_CLASSES:
                continue
            out.append({
                "image_path": str(img_path),
                "image_file": img_file,
                "type": t,
                "title": (r.get("TITLE") or "").strip(),
                "author": (r.get("AUTHOR") or "").strip(),
                "timeframe": (r.get("TIMEFRAME") or "").strip(),
            })
    return out


def extract_features(rows, model, processor, device, batch_size=16, log_every=200):
    """Прогоняет SigLIP vision encoder по списку строк, возвращает numpy-массив (N, d)."""
    feats = []
    t0 = time.time()
    n = len(rows)
    for i in range(0, n, batch_size):
        batch = rows[i:i + batch_size]
        images = [Image.open(r["image_path"]).convert("RGB") for r in batch]
        inputs = processor(images=images, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.get_image_features(**inputs)
        # в одних версиях transformers get_image_features возвращает полный
        # BaseModelOutputWithPooling, в других — спроецированный тензор
        if isinstance(out, torch.Tensor):
            tensor = out
        elif hasattr(out, "pooler_output") and out.pooler_output is not None:
            tensor = out.pooler_output
        elif hasattr(out, "last_hidden_state"):
            # как крайний случай — mean-pool по patch-токенам
            tensor = out.last_hidden_state.mean(dim=1)
        else:
            raise RuntimeError(f"Unexpected get_image_features return type: {type(out)}")
        feats.append(tensor.detach().cpu().numpy())

        done = i + len(batch)
        if done % log_every < batch_size:
            elapsed = time.time() - t0
            rate = done / elapsed
            eta = (n - done) / rate
            print(f"  [{done}/{n}] {rate:.1f} img/sec  ETA {eta/60:.1f} min")

    return np.concatenate(feats, axis=0)


def build_or_load_cache(split: str, model, processor, device, force_recompute=False):
    """Возвращает (features, rows). Кэширует features на диск."""
    cache = CACHE_DIR / f"siglip_features_{split}.npz"
    csv_path = SEMART_DIR / f"semart_{split}.csv"

    rows = load_rows(csv_path)
    print(f"  {split}: {len(rows)} rows with mappable TYPE")

    if cache.exists() and not force_recompute:
        data = np.load(cache, allow_pickle=True)
        feats = data["features"]
        cached_files = list(data["image_files"])
        # проверяем, что кэш совпадает с текущим порядком строк
        if cached_files == [r["image_file"] for r in rows]:
            print(f"  → loaded {feats.shape} from cache")
            return feats, rows
        print(f"  → cache mismatch (rows changed), recomputing")

    print(f"  Extracting SigLIP features for {len(rows)} images …")
    feats = extract_features(rows, model, processor, device)
    np.savez_compressed(
        cache,
        features=feats,
        image_files=np.array([r["image_file"] for r in rows]),
    )
    print(f"  → cached {feats.shape} to {cache.name}")
    return feats, rows


def evaluate(X, y_true, clf, class_order):
    """Возвращает словарь с accuracy, P/R/F1 по классам и confusion matrix."""
    from collections import Counter
    y_pred = clf.predict(X)
    n = len(y_true)
    accuracy = float((y_pred == y_true).mean())

    per_class = {}
    for c in class_order:
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        support = int((y_true == c).sum())
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / support if support > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        per_class[c] = {
            "support": support,
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f1, 4),
        }

    macro_p = sum(per_class[c]["precision"] for c in class_order) / len(class_order)
    macro_r = sum(per_class[c]["recall"] for c in class_order) / len(class_order)
    macro_f = sum(per_class[c]["f1"] for c in class_order) / len(class_order)

    # confusion matrix
    conf = {gt: {pr: 0 for pr in class_order} for gt in class_order}
    for gt, pr in zip(y_true, y_pred):
        conf[gt][pr] += 1

    return {
        "n": n,
        "accuracy": round(accuracy, 4),
        "macro_precision": round(macro_p, 4),
        "macro_recall": round(macro_r, 4),
        "macro_f1": round(macro_f, 4),
        "per_class": per_class,
        "confusion": conf,
    }, y_pred


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-name", default="google/siglip-base-patch16-224")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--output", default="data/semart/probe_theme_archival_v2.json")
    p.add_argument("--force-extract", action="store_true",
                   help="Re-extract features even if cache exists")
    p.add_argument("--C", type=float, default=1.0, help="LogReg inverse regularization")
    p.add_argument("--class-weight", choices=["balanced", "none"], default="balanced")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(get_device())
    print(f"[1/4] Loading SigLIP ({args.model_name}) on {device}")
    processor = AutoProcessor.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name).to(device).eval()
    print()

    print("[2/4] Building features for train + val splits")
    X_train, rows_train = build_or_load_cache("train", model, processor, device,
                                               force_recompute=args.force_extract)
    X_val, rows_val = build_or_load_cache("val", model, processor, device,
                                           force_recompute=args.force_extract)
    print()

    y_train = np.array([r["type"] for r in rows_train])
    y_val = np.array([r["type"] for r in rows_val])

    from collections import Counter
    print(f"  Train class distribution: {dict(Counter(y_train))}")
    print(f"  Val   class distribution: {dict(Counter(y_val))}")
    print()

    print(f"[3/4] Training LogisticRegression (C={args.C}, class_weight={args.class_weight})")
    from sklearn.linear_model import LogisticRegression
    cw = "balanced" if args.class_weight == "balanced" else None
    # аргумент multi_class убран в sklearn 1.5+ (multinomial по умолчанию для >2 классов)
    clf = LogisticRegression(
        C=args.C,
        class_weight=cw,
        max_iter=2000,
        n_jobs=-1,
    )
    t0 = time.time()
    clf.fit(X_train, y_train)
    print(f"  fit time: {time.time()-t0:.1f}s")
    train_acc = clf.score(X_train, y_train)
    print(f"  train accuracy: {train_acc:.4f}")
    print()

    print("[4/4] Evaluating on val")
    metrics, y_pred = evaluate(X_val, y_val, clf, THEME_CLASSES)
    print(json.dumps({k: v for k, v in metrics.items() if k != "confusion"},
                     indent=2, ensure_ascii=False))
    print("\nConfusion matrix (GT \\ Pred):")
    print(f'{"":15}' + "".join(f"{c[:11]:>12}" for c in THEME_CLASSES))
    for gt in THEME_CLASSES:
        row = metrics["confusion"][gt]
        print(f'{gt:15}' + "".join(f"{row[pr]:>12}" for pr in THEME_CLASSES))

    # сравниваем с zero-shot baseline (читаем вывод бенчмарка, если есть)
    baseline_path = CACHE_DIR / "benchmark_archival_v2_val.json"
    comparison = None
    if baseline_path.exists():
        with open(baseline_path) as f:
            baseline = json.load(f)
        b_theme = baseline["theme_metrics"]
        comparison = {
            "zero_shot_accuracy": b_theme["accuracy"],
            "probe_accuracy": metrics["accuracy"],
            "delta_pp": round((metrics["accuracy"] - b_theme["accuracy"]) * 100, 1),
            "zero_shot_macro_f1": b_theme["macro_f1"],
            "probe_macro_f1": metrics["macro_f1"],
            "macro_f1_delta_pp": round((metrics["macro_f1"] - b_theme["macro_f1"]) * 100, 1),
        }
        print("\n=== COMPARISON vs zero-shot baseline ===")
        print(json.dumps(comparison, indent=2))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    per_item = []
    for r, gt, pr in zip(rows_val, y_val, y_pred):
        per_item.append({
            "image_file": r["image_file"],
            "gt": str(gt), "pred": str(pr),
            "correct": bool(gt == pr),
        })
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "model_name": args.model_name,
            "n_train": len(rows_train),
            "n_val": len(rows_val),
            "C": args.C,
            "class_weight": args.class_weight,
            "train_accuracy": round(float(train_acc), 4),
            "val_metrics": metrics,
            "comparison_to_zero_shot": comparison,
            "per_item": per_item,
        }, f, indent=2, ensure_ascii=False)
    print(f"\n[INFO] Saved to {out_path}")


if __name__ == "__main__":
    main()
