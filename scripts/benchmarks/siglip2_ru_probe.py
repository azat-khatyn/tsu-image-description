"""siglip2_ru_probe.py — Шаг 0: read-only A/B проба SigLIP 2 + русские промпты.

Сравнивает текущий продакшн (SigLIP 1 + англ. промпты) с кандидатом
(SigLIP 2 + русские промпты) на двух осях:

  1. ТЕХНИКА (style) на размеченном НЭБ-наборе (есть GT: литография/автотипия/
     цинкография) — главный тест технических терминов на русском.
  2. ТЕМА (theme) на открытках n60+semtest (без GT) — confident-rate, распределение
     и согласие меток между конфигурациями.

Конфигурации:
  A  siglip1 + EN   (продакшн-базлайн)
  B  siglip2 + RU   (кандидат)
  C  siglip2 + EN   (контроль: отделяет эффект модели от языка)

Скоринг повторяет продакшн (_classify_with_scores/_pack_field):
softmax по кандидатам от logits_per_image; confident = score>=threshold И
(best-second)>=margin. Пороги взяты из MetadataExtractor (style/theme).

Только чтение: исходники не меняются, веса докачиваются в кэш HF.
Результат → data/eval/siglip2_ru_probe.json.
"""

import json
import time
from collections import Counter
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

ROOT = Path(__file__).resolve().parents[2]

SIGLIP1 = "google/siglip-base-patch16-224"
SIGLIP2 = "google/siglip2-base-patch16-224"

# пороги продакшна (MetadataExtractor)
STYLE_THRESHOLD, STYLE_MARGIN = 0.22, 0.03
THEME_THRESHOLD, THEME_MARGIN = 0.18, 0.03

# --- ось ТЕХНИКА: 3 класса с GT в НЭБ ---
TECH_CLASSES = ["lithograph", "autotypia", "zincography"]
TECH_PROMPTS = {
    "en": {"lithograph": "a lithograph", "autotypia": "a halftone print", "zincography": "a zincograph"},
    "ru": {"lithograph": "литография", "autotypia": "автотипия", "zincography": "цинкография"},
}

# --- ось ТЕМА: 8 классов archival_v2 (en = стабильный id) ---
THEME_IDS = [
    "a landscape", "an urban view", "a portrait", "a genre scene",
    "a still life", "a religious subject", "a military subject", "a holiday scene",
]
THEME_PROMPTS = {
    "en": {t: t for t in THEME_IDS},
    "ru": dict(zip(THEME_IDS, [
        "пейзаж", "городской вид", "портрет", "жанровая сцена",
        "натюрморт", "религиозный сюжет", "военный сюжет", "праздничный сюжет",
    ])),
}

CONFIGS = {
    "A_siglip1_en": (SIGLIP1, "en"),
    "B_siglip2_ru": (SIGLIP2, "ru"),
    "C_siglip2_en": (SIGLIP2, "en"),
}


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def canon_technique(tech: str):
    t = (tech or "").strip().lower()
    if "автотип" in t:
        return "autotypia"
    if "цинкогр" in t and "литогр" not in t:
        return "zincography"
    if "литогр" in t or "хромолитогр" in t:
        return "lithograph"
    return None


def load_neb_items():
    m = json.load(open(ROOT / "data" / "neb_leningrad_wwii" / "manifest.json"))
    out = []
    for it in m["items"]:
        cls = canon_technique(it.get("technique") or "")
        ip = it.get("image_path")
        if cls and ip and (ROOT / ip).exists():
            out.append({"image_path": str(ROOT / ip), "image_file": Path(ip).name, "gt": cls})
    return out


def load_postcards():
    out, seen = [], set()
    for name, p in [("n60", "data/eval/results/final/metrics_E06b_archival_v2_n60.json"),
                    ("semtest", "data/eval/results/final/metrics_E00_semtest.json")]:
        d = json.load(open(ROOT / p))
        for it in d.get("per_item", []):
            ip = it["image_path"]
            abs_p = ROOT / ip
            if not abs_p.exists() or str(abs_p) in seen:
                continue
            seen.add(str(abs_p))
            out.append({"image_path": str(abs_p), "image_file": Path(ip).name,
                        "set": name, "source": it.get("source", "?")})
    return out


def classify(model, processor, image, prompts, device):
    """Возвращает {prompt: prob} — softmax по кандидатам, как в продакшне."""
    inputs = processor(text=prompts, images=image, padding="max_length",
                       return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**inputs).logits_per_image
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().tolist()
    return {p: float(s) for p, s in zip(prompts, probs)}


def pack(scores, threshold, margin):
    ordered = sorted(scores.items(), key=lambda x: -x[1])
    best_label, best = ordered[0]
    second = ordered[1][1] if len(ordered) > 1 else 0.0
    return best_label, best, bool(best >= threshold and (best - second) >= margin)


def run_config(model_name, lang, neb_items, postcards, device):
    print(f"\n=== load {model_name} ({lang}) ===")
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()

    # --- технология (GT) ---
    tech_map = TECH_PROMPTS[lang]                      # id -> prompt
    tech_prompts = [tech_map[c] for c in TECH_CLASSES]
    prompt_to_id = {tech_map[c]: c for c in TECH_CLASSES}
    tech_items, correct, confident_n = [], 0, 0
    t0 = time.time()
    for it in neb_items:
        img = Image.open(it["image_path"]).convert("RGB")
        scores = classify(model, processor, img, tech_prompts, device)
        best_prompt, score, conf = pack(scores, STYLE_THRESHOLD, STYLE_MARGIN)
        pred = prompt_to_id[best_prompt]
        correct += int(pred == it["gt"])
        confident_n += int(conf)
        tech_items.append({"image_file": it["image_file"], "gt": it["gt"],
                           "pred": pred, "score": round(score, 4), "confident": conf})
    tech_acc = correct / len(neb_items) if neb_items else 0.0
    # per-class recall
    per_class = {}
    for c in TECH_CLASSES:
        sub = [t for t in tech_items if t["gt"] == c]
        per_class[c] = {"support": len(sub),
                        "recall": round(sum(t["pred"] == c for t in sub) / len(sub), 4) if sub else 0.0}
    print(f"  technique: acc={tech_acc:.3f} confident={confident_n}/{len(neb_items)} "
          f"({time.time()-t0:.0f}s)")

    # --- тема (без GT) ---
    theme_map = THEME_PROMPTS[lang]                    # id -> prompt
    theme_prompts = [theme_map[t] for t in THEME_IDS]
    prompt_to_theme = {theme_map[t]: t for t in THEME_IDS}
    theme_items, theme_conf = [], 0
    for it in postcards:
        img = Image.open(it["image_path"]).convert("RGB")
        scores = classify(model, processor, img, theme_prompts, device)
        best_prompt, score, conf = pack(scores, THEME_THRESHOLD, THEME_MARGIN)
        theme_id = prompt_to_theme[best_prompt]
        theme_conf += int(conf)
        theme_items.append({"image_file": it["image_file"], "set": it["set"],
                            "theme_id": theme_id, "score": round(score, 4), "confident": conf})
    print(f"  theme: confident={theme_conf}/{len(postcards)} "
          f"dist={dict(Counter(t['theme_id'] for t in theme_items))}")

    del model
    if device.type == "mps":
        torch.mps.empty_cache()

    return {
        "model_name": model_name, "lang": lang,
        "technique": {"accuracy": round(tech_acc, 4),
                      "confident_rate": round(confident_n / len(neb_items), 4) if neb_items else 0.0,
                      "per_class": per_class, "per_item": tech_items},
        "theme": {"confident_rate": round(theme_conf / len(postcards), 4) if postcards else 0.0,
                  "distribution": dict(Counter(t["theme_id"] for t in theme_items)),
                  "per_item": theme_items},
    }


def main():
    device = get_device()
    print(f"device={device}")
    neb_items = load_neb_items()
    postcards = load_postcards()
    print(f"NEB technique-GT: {len(neb_items)} | postcards: {len(postcards)}")

    results = {}
    for name, (model_name, lang) in CONFIGS.items():
        results[name] = run_config(model_name, lang, neb_items, postcards, device)

    # согласие тем: B(siglip2+ru) vs A(siglip1+en) и C(siglip2+en) vs A
    def theme_by_file(cfg):
        return {t["image_file"]: t["theme_id"] for t in results[cfg]["theme"]["per_item"]}
    a = theme_by_file("A_siglip1_en")
    agree = {}
    for cfg in ("B_siglip2_ru", "C_siglip2_en"):
        b = theme_by_file(cfg)
        same = sum(1 for f in a if a[f] == b.get(f))
        agree[cfg] = {"agree_with_A": same, "n": len(a),
                      "agree_rate": round(same / len(a), 4) if a else 0.0}

    summary = {
        "technique_accuracy": {k: results[k]["technique"]["accuracy"] for k in CONFIGS},
        "technique_confident_rate": {k: results[k]["technique"]["confident_rate"] for k in CONFIGS},
        "theme_confident_rate": {k: results[k]["theme"]["confident_rate"] for k in CONFIGS},
        "theme_agreement_vs_A": agree,
    }
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    out = ROOT / "data" / "eval" / "siglip2_ru_probe.json"
    json.dump({"summary": summary, "configs": {k: list(v) for k, v in CONFIGS.items()},
               "results": results}, open(out, "w"), indent=2, ensure_ascii=False)
    print(f"\n[INFO] saved {out}")


if __name__ == "__main__":
    main()
