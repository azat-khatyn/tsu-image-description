"""siglip_technique_probe_neb.py — domain-matched technique probe на НЭБ.

Опция B из обсуждения: linear probe на frozen SigLIP features, обученный на
НЭБ-данных (Leningrad WWII collection, 322 postcards) с supervisor-меткой =
канонизированной TECHNIQUE из RUSMARC field 215 $c.

3 класса (после нормализации, см. CANONICAL_CLASSES):
  - lithograph  (литогр. + хромолитогр., n≈148)
  - autotypia   (автотип., n≈98)
  - zincography (цинкогр., n≈19)

Архитектурное обоснование выбора именно TECHNIQUE (а не theme):
  - Техника — визуальный признак, мало зависящий от тематики (цвет vs ч/б,
    тип штриха, наличие растровой структуры). Поэтому overfit-риск на узкий
    Soviet-WWII domain здесь существенно ниже, чем для theme probe.
  - 3 класса с разумным data balance после балансирующих весов.

Pipeline:
  1. Extract SigLIP features для 322 НЭБ-изображений (CPU, чтобы не конфликтовать
     с одновременно работающим E12 pipeline на MPS).
  2. Map technique strings → 3 канонических класса.
  3. 80/20 train/val split с stratify по классу.
  4. Train sklearn LogisticRegression (class_weight='balanced').
  5. Evaluate on val: accuracy, per-class P/R/F1, confusion.
  6. CROSS-DOMAIN: apply probe to n=60 (RGB+NYPL) postcards, посчитать
     distribution и compare к zero-shot SigLIP style-axis. Если probe
     уверенно говорит «lithograph» на 100% NYPL-photos — overfit подтверждён.

Outputs:
  - data/neb_leningrad_wwii/probe_technique.json — metrics + per-item predictions
  - logs/probe_technique_neb.log
"""

import argparse
import json
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from transformers import AutoModel, AutoProcessor

ROOT = Path(__file__).resolve().parent.parent
NEB_MANIFEST = ROOT / "data" / "neb_leningrad_wwii" / "manifest.json"
N60_METRICS = ROOT / "data" / "eval" / "results" / "final" / "metrics_E06b_archival_v2_n60.json"


# Канонизация — какие строки technique в какой класс
def canonicalize_technique(tech: str) -> str | None:
    t = (tech or "").strip().lower()
    if not t:
        return None
    # priority order
    if "автотип" in t:
        return "autotypia"
    if "цинкогр" in t and "литогр" not in t:
        return "zincography"
    if "литогр" in t or "хромолитогр" in t:
        return "lithograph"
    return None  # рис., портр., ил., фототип. — skip


CLASS_ORDER = ["lithograph", "autotypia", "zincography"]


def load_neb_items():
    with open(NEB_MANIFEST) as f:
        m = json.load(f)
    out = []
    for it in m["items"]:
        img = it.get("image_path")
        if not img or not (ROOT / img).exists():
            continue
        cls = canonicalize_technique(it.get("technique") or "")
        if cls is None:
            continue
        out.append({
            "image_path": str(ROOT / img),
            "image_file": Path(img).name,
            "technique_raw": it.get("technique"),
            "technique_class": cls,
            "year": it.get("year"),
            "title": it.get("title", ""),
        })
    return out


def load_n60_items():
    if not N60_METRICS.exists():
        return []
    with open(N60_METRICS) as f:
        d = json.load(f)
    out = []
    for it in d.get("per_item", []):
        ip = it["image_path"]
        abs_p = ROOT / ip
        if not abs_p.exists():
            continue
        out.append({
            "image_path": str(abs_p),
            "image_file": Path(ip).name,
            "source": it.get("source", "unknown"),
        })
    return out


def get_device(prefer="cpu"):
    if prefer == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def extract_features(items, model, processor, device, batch_size=16, log_every=64):
    feats = []
    t0 = time.time()
    n = len(items)
    for i in range(0, n, batch_size):
        batch = items[i:i + batch_size]
        images = [Image.open(it["image_path"]).convert("RGB") for it in batch]
        inputs = processor(images=images, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.get_image_features(**inputs)
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            tensor = out.pooler_output
        elif isinstance(out, torch.Tensor):
            tensor = out
        else:
            tensor = out.last_hidden_state.mean(dim=1)
        feats.append(tensor.detach().cpu().numpy())
        done = i + len(batch)
        if done % log_every < batch_size:
            rate = done / (time.time() - t0)
            eta = (n - done) / rate
            print(f"  [{done}/{n}] {rate:.1f} img/sec  ETA {eta:.1f}s")
    return np.concatenate(feats, axis=0)


def evaluate_split(X_val, y_val, clf, class_order):
    y_pred = clf.predict(X_val)
    n = len(y_val)
    acc = float((y_pred == y_val).mean())
    per_class = {}
    for c in class_order:
        tp = int(((y_pred == c) & (y_val == c)).sum())
        fp = int(((y_pred == c) & (y_val != c)).sum())
        fn = int(((y_pred != c) & (y_val == c)).sum())
        sup = int((y_val == c).sum())
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / sup if sup > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        per_class[c] = {"support": sup, "precision": round(p, 4),
                        "recall": round(r, 4), "f1": round(f1, 4)}
    macro_f1 = sum(per_class[c]["f1"] for c in class_order) / len(class_order)
    # confusion
    conf = {gt: {pr: 0 for pr in class_order} for gt in class_order}
    for gt, pr in zip(y_val, y_pred):
        conf[gt][pr] += 1
    return {
        "n": n,
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "per_class": per_class,
        "confusion": conf,
    }, y_pred


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--device", choices=["cpu", "mps"], default="cpu",
                   help="SigLIP device. CPU avoids contention with concurrent MPS jobs.")
    p.add_argument("--output", default="data/neb_leningrad_wwii/probe_technique.json")
    p.add_argument("--C", type=float, default=1.0)
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    device = get_device(args.device)
    print(f"[1/5] Loading SigLIP on {device}")
    processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")
    model = AutoModel.from_pretrained("google/siglip-base-patch16-224").to(device).eval()
    print()

    print("[2/5] Loading НЭБ items with canonical TECHNIQUE label")
    neb_items = load_neb_items()
    dist = Counter(it["technique_class"] for it in neb_items)
    print(f"  Usable items: {len(neb_items)} (filtered out 'other' techniques)")
    print(f"  Distribution: {dict(dist)}")
    print()

    print("[3/5] Extracting SigLIP features for НЭБ")
    X = extract_features(neb_items, model, processor, device)
    y = np.array([it["technique_class"] for it in neb_items])
    print(f"  Feature matrix: {X.shape}")
    print()

    # Stratified split
    X_train, X_val, y_train, y_val, idx_train, idx_val = train_test_split(
        X, y, np.arange(len(neb_items)),
        test_size=args.val_fraction, stratify=y, random_state=args.seed,
    )
    print(f"[4/5] Train/val split: {len(X_train)}/{len(X_val)} "
          f"(stratified by class)")

    clf = LogisticRegression(C=args.C, class_weight="balanced", max_iter=2000)
    clf.fit(X_train, y_train)
    train_acc = clf.score(X_train, y_train)
    print(f"  fit ok, train_acc={train_acc:.4f}")

    val_metrics, y_pred_val = evaluate_split(X_val, y_val, clf, CLASS_ORDER)
    print(f"  val: accuracy={val_metrics['accuracy']:.4f} "
          f"macro_f1={val_metrics['macro_f1']:.4f}")
    for c in CLASS_ORDER:
        d = val_metrics["per_class"][c]
        print(f"    {c:>12}: sup={d['support']} P={d['precision']:.3f} "
              f"R={d['recall']:.3f} F1={d['f1']:.3f}")
    print("  Confusion (GT \\ Pred):")
    print(f"  {'':>12} " + "".join(f"{c[:11]:>13}" for c in CLASS_ORDER))
    for gt in CLASS_ORDER:
        row = val_metrics["confusion"][gt]
        print(f"  {gt:>12} " + "".join(f"{row[pr]:>13}" for pr in CLASS_ORDER))
    print()

    # Cross-domain check on n=60
    print("[5/5] Cross-domain check on n=60 (RGB+NYPL, no ground-truth TECHNIQUE)")
    n60_items = load_n60_items()
    print(f"  Loaded {len(n60_items)} n=60 items")
    if n60_items:
        X_n60 = extract_features(n60_items, model, processor, device)
        y_pred_n60 = clf.predict(X_n60)
        n60_probs = clf.predict_proba(X_n60)
        # Distribution
        cross_dist = Counter(y_pred_n60.tolist())
        print(f"  Probe predictions on n=60: {dict(cross_dist)}")
        # Per-source breakdown
        by_source = {}
        for it, pr in zip(n60_items, y_pred_n60):
            by_source.setdefault(it["source"], []).append(pr)
        cross_by_source = {}
        for src, preds in by_source.items():
            d = Counter(preds)
            cross_by_source[src] = {"n": len(preds), "distribution": dict(d)}
            print(f"    {src}: n={len(preds)}, "
                  f"{', '.join(f'{c}={n}' for c, n in d.most_common())}")
        # Confidence distribution
        max_probs = n60_probs.max(axis=1)
        print(f"  Mean confidence (max prob): {max_probs.mean():.3f}")
        print(f"  Low-confidence (<0.5): {(max_probs < 0.5).sum()}/{len(max_probs)}")
    else:
        X_n60 = None
        y_pred_n60 = None
        cross_dist = {}
        cross_by_source = {}

    # Save
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    per_item_val = []
    for i in idx_val:
        it = neb_items[int(i)]
        gt = y[int(i)]
        # find prediction
        pred_idx = list(idx_val).index(int(i))
        pred = y_pred_val[pred_idx]
        per_item_val.append({
            "image_file": it["image_file"],
            "technique_raw": it["technique_raw"],
            "gt": str(gt),
            "pred": str(pred),
            "correct": bool(gt == pred),
        })

    per_item_cross = []
    if n60_items:
        for it, pr, probs in zip(n60_items, y_pred_n60, n60_probs):
            per_item_cross.append({
                "image_file": it["image_file"],
                "source": it["source"],
                "pred": str(pr),
                "max_prob": float(probs.max()),
            })

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "device": str(device),
            "n_train": len(X_train),
            "n_val": len(X_val),
            "n_cross_n60": len(n60_items),
            "class_distribution_neb": dict(dist),
            "train_accuracy": round(float(train_acc), 4),
            "val_metrics": val_metrics,
            "cross_domain_n60": {
                "distribution": dict(cross_dist),
                "by_source": cross_by_source,
            },
            "per_item_val": per_item_val,
            "per_item_cross_n60": per_item_cross,
        }, f, indent=2, ensure_ascii=False)
    print(f"\n[INFO] Saved to {out_path}")


if __name__ == "__main__":
    main()
