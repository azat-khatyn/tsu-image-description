"""probe_on_postcards.py — применяет обученный на SemArt linear probe к postcards.

Проверка переноса между доменами: обучен на ренессансной/барочной живописи
(5 классов: religious/portrait/landscape/genre/still-life). Открытки относятся к
другому визуальному домену (печатная графика начала XX века) и другому
распределению сюжетов (праздники, городские виды, военные сцены - ничего из этого
в SemArt не было).

Этот скрипт:
  1. извлекает SigLIP features для postcards (n=60 + n=14 semtest).
  2. переобучает из кэшированных SemArt features.
  3. применяет предсказанный theme + confidence для каждой открытки.
  4. также прогоняет zero-shot SigLIP по themes из archival_v2 (8 классов) для сравнения.
  5. сохраняет per-item JSON с обоими предсказаниями для ручной проверки.

Выход: data/semart/probe_postcards_n74.json
"""

import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from transformers import AutoModel, AutoProcessor

ROOT = Path(__file__).resolve().parent.parent

# ---------- данные для обучения probe (кэшированные SemArt features) ----------
PROBE_CACHE_TRAIN = ROOT / "data" / "semart" / "siglip_features_train.npz"
PROBE_VAL_CSV = ROOT / "data" / "semart" / "SemArt" / "semart_train.csv"
THEME_CLASSES = ["religious", "portrait", "landscape", "genre", "still-life"]

# ---------- источники postcards ----------
EVAL_N60_METRICS = ROOT / "data" / "eval" / "results" / "final" / "metrics_E06b_archival_v2_n60.json"
EVAL_SEMTEST_METRICS = ROOT / "data" / "eval" / "results" / "final" / "metrics_E00_semtest.json"

# ---------- themes из archival_v2 для zero-shot сравнения ----------
ARCHIVAL_THEMES = [
    "a landscape",
    "an urban view",
    "a portrait",
    "a genre scene",
    "a still life",
    "a religious subject",
    "a military subject",
    "a holiday scene",
]


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_postcard_images():
    """Возвращает список словарей {image_path, source, set, reference_ru, image_file}."""
    items = []
    seen = set()
    for set_name, metrics_path in [("n60", EVAL_N60_METRICS), ("semtest", EVAL_SEMTEST_METRICS)]:
        if not metrics_path.exists():
            print(f"  WARN: {metrics_path} not found, skipping {set_name}")
            continue
        with open(metrics_path) as f:
            d = json.load(f)
        for it in d.get("per_item", []):
            ip = it["image_path"]
            abs_path = ROOT / ip
            if not abs_path.exists():
                continue
            key = (str(abs_path), set_name)
            if key in seen:
                continue
            seen.add(key)
            items.append({
                "image_path": str(abs_path),
                "image_file": Path(ip).name,
                "set": set_name,
                "source": it.get("source", "unknown"),
                "reference_ru": it.get("reference_ru") or "",
            })
    return items


def extract_features_and_zeroshot(items, model, processor, device,
                                   archival_themes):
    """Один forward pass на изображение: image features + zero-shot scores по themes."""
    feats_list = []
    zs_preds = []
    # токенизируем текст один раз
    text_inputs = processor(text=archival_themes, return_tensors="pt",
                            padding="max_length").to(device)
    with torch.no_grad():
        text_features = model.get_text_features(**text_inputs)
        if hasattr(text_features, "pooler_output"):
            text_features = text_features.pooler_output
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    t0 = time.time()
    for i, it in enumerate(items):
        img = Image.open(it["image_path"]).convert("RGB")
        inputs = processor(images=[img], return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.get_image_features(**inputs)
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            img_feat = out.pooler_output
        elif isinstance(out, torch.Tensor):
            img_feat = out
        else:
            img_feat = out.last_hidden_state.mean(dim=1)

        # zero-shot через cosine + softmax (как logits_per_image в forward SigLIP)
        img_feat_norm = img_feat / img_feat.norm(dim=-1, keepdim=True)
        logits = (img_feat_norm @ text_features.T).squeeze(0)
        probs = torch.softmax(logits, dim=0).detach().cpu().numpy()
        top_idx = int(np.argmax(probs))
        zs_preds.append({
            "label": archival_themes[top_idx],
            "score": float(probs[top_idx]),
            "all_probs": {t: float(p) for t, p in zip(archival_themes, probs)},
        })

        feats_list.append(img_feat.detach().cpu().numpy())

    feats = np.concatenate(feats_list, axis=0)
    print(f"  → {len(items)} images, {time.time()-t0:.1f}s")
    return feats, zs_preds


def train_probe_from_cache():
    """Переобучает probe из кэшированных SemArt train features."""
    data = np.load(PROBE_CACHE_TRAIN, allow_pickle=True)
    X_train = data["features"]
    image_files = list(data["image_files"])

    # метки восстанавливаем из semart_train.csv
    label_by_file = {}
    with open(PROBE_VAL_CSV, encoding="latin-1") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for r in reader:
            t = (r.get("TYPE") or "").strip().lower()
            if t in THEME_CLASSES:
                label_by_file[r["IMAGE_FILE"].strip()] = t

    y_train = np.array([label_by_file[f] for f in image_files])

    clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)
    clf.fit(X_train, y_train)
    train_acc = clf.score(X_train, y_train)
    print(f"  probe re-trained: train_acc={train_acc:.4f}, classes={list(clf.classes_)}")
    return clf


def main():
    device = get_device()
    print(f"[1/5] Loading SigLIP on {device}")
    processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")
    model = AutoModel.from_pretrained("google/siglip-base-patch16-224").to(device).eval()
    print()

    print("[2/5] Re-training probe from cached SemArt features")
    clf = train_probe_from_cache()
    print()

    print("[3/5] Loading postcard images")
    items = load_postcard_images()
    by_set = {}
    for it in items:
        by_set.setdefault(it["set"], []).append(it)
    print(f"  Loaded {len(items)} unique postcards: " +
          ", ".join(f"{k}={len(v)}" for k, v in by_set.items()))
    print()

    print("[4/5] Extracting features + zero-shot predictions for postcards")
    X_post, zs_preds = extract_features_and_zeroshot(
        items, model, processor, device, ARCHIVAL_THEMES,
    )
    print()

    print("[5/5] Applying probe + analysing predictions")
    probe_probs = clf.predict_proba(X_post)
    probe_class_order = list(clf.classes_)
    probe_preds = []
    for probs in probe_probs:
        ranked = sorted(zip(probe_class_order, probs), key=lambda x: -x[1])
        probe_preds.append({
            "label": ranked[0][0],
            "score": float(ranked[0][1]),
            "margin": float(ranked[0][1] - ranked[1][1]),
            "all_probs": {c: float(p) for c, p in ranked},
        })

    # сборка per-item
    per_item = []
    for it, zp, pp in zip(items, zs_preds, probe_preds):
        per_item.append({
            **it,
            "probe_theme": pp,
            "zero_shot_theme": zp,
        })

    # анализ распределений
    from collections import Counter
    probe_dist = Counter(p["probe_theme"]["label"] for p in per_item)
    zs_dist = Counter(p["zero_shot_theme"]["label"] for p in per_item)

    print("\n=== Probe (5 SemArt classes) — predictions on postcards ===")
    for c, n in probe_dist.most_common():
        print(f"  {n:4d}  {c}")

    print("\n=== Zero-shot SigLIP (8 archival_v2 classes) — predictions on postcards ===")
    for c, n in zs_dist.most_common():
        print(f"  {n:4d}  {c}")

    # confidence probe (средний margin)
    margins = [p["probe_theme"]["margin"] for p in per_item]
    mean_margin = sum(margins) / len(margins)
    low_conf = sum(1 for m in margins if m < 0.1)
    print(f"\nProbe confidence: mean margin = {mean_margin:.3f}, low-conf (margin<0.1): {low_conf}/{len(per_item)}")

    # разбивка по set
    print("\n=== Probe predictions by set ===")
    for set_name, set_items in by_set.items():
        set_probe = Counter(it["probe_theme"]["label"]
                            for it in per_item if it["set"] == set_name)
        print(f"  {set_name} (n={len(set_items)}): {dict(set_probe)}")

    # сохраняем
    out_path = ROOT / "data" / "semart" / "probe_postcards_n74.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "n_postcards": len(items),
            "probe_classes": probe_class_order,
            "archival_themes": ARCHIVAL_THEMES,
            "probe_distribution": dict(probe_dist),
            "zero_shot_distribution": dict(zs_dist),
            "probe_mean_margin": mean_margin,
            "per_item": per_item,
        }, f, indent=2, ensure_ascii=False)
    print(f"\n[INFO] Saved to {out_path}")


if __name__ == "__main__":
    main()
