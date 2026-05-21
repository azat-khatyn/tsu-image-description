"""compute_sds.py — Semantic Density Score (SDS) для архивных описаний.

SDS — четвёртый reference-free сигнал из раздела 2.2 (П3) метода.
Измеряет долю из 6 семантических осей, покрытых описанием:
  a1 — тип материала
  a2 — визуальный сюжет
  a3 — художественный стиль
  a4 — эпоха
  a5 — культурно-исторический контекст
  a6 — эмоциональный тон

Полная рубрика и определение — `docs/llm_judge_rubric.md` (раздел SDS).

Два режима:
  --mode keyword  — offline, по словарю ключевых слов (default; не требует API)
  --mode llm      — Claude Sonnet, по axis-rubric из docs/llm_judge_rubric.md
                    (требует ANTHROPIC_API_KEY)

Вход: metrics_triad_*.json из scripts/evaluate.py (поле per_item с archive_ru).
Выход: JSON с SDS per-item, агрегатами и per-axis coverage rate.

Использование:
    python scripts/compute_sds.py \\
        --input data/eval/results/final/metrics_proposed_retrieval_n220.json \\
        --output data/eval/results/final/sds_proposed_retrieval_n220.json \\
        --mode keyword
"""

import argparse
import base64
import json
import os
import re
import time
from pathlib import Path

# ----------------------------------------------------------------------------
# Keyword-based detection (offline fallback)
# ----------------------------------------------------------------------------

AXIS_KEYWORDS = {
    "type_of_material": [
        r"\bоткрыт",
        r"\bплакат",
        r"\bиллюстрац",
        r"\bфотограф",
        r"\bгравюр",
        r"\bпоздравительн",
        r"\bкарточк",
    ],
    "artistic_style": [
        # legacy_v1 colloquial terms (retained for backward compat)
        r"\bвинтаж",
        r"\bретро",
        r"\bдекоративн",
        # core techniques (present in both taxonomies)
        r"\bгравюр",
        r"\bживопис",
        r"\bрисун",
        r"\bчёрно-бел",
        r"\bчерно-бел",
        r"\bцветн[аыо][йея]\s+фотограф",
        # archival_v2 technical terms (E06b)
        r"\bлитограф",            # литография
        r"\bхромолитограф",       # хромолитография
        r"\bофорт",                # офорт
        r"\bакварел",             # акварель
        r"\bгуаш",                 # гуашь
        r"\bмасл[яёо]н",          # масляная (живопись)
        r"\bкарандашн",           # карандашный (рисунок)
    ],
    "epoch": [
        r"\bвек\b",
        r"\bвека\b",
        r"\bстолети",
        r"\b18\d{2}\b",
        r"\b19\d{2}\b",
        r"\bдореволюционн",
        r"\bдовоенн",
        r"\bначал[оае] xx",
        r"\bконец xix",
    ],
    "cultural_context": [
        r"\bрождеств",
        r"\bпасха",
        r"\bпасхальн",
        r"\bновогодн",
        r"\bновый год",
        r"\bрелигиозн",
        r"\bхрам",
        r"\bцерков",
        r"\bпраздничн",
        r"\bгородск",
        r"\bсельск",
        r"\bвоенн",
        r"\bсемейн",
    ],
    "mood": [
        r"\bрадостн",
        r"\bторжеств",
        r"\bпраздничн",
        r"\bромантическ",
        r"\bностальгическ",
        r"\bспокойн",
        r"\bсерьёзн",
        r"\bсерьезн",
    ],
}

AXES = [
    "type_of_material",
    "visual_subject",
    "artistic_style",
    "epoch",
    "cultural_context",
    "mood",
]


def detect_visual_subject(text: str) -> int:
    """Heuristic for axis a2 (visual_subject) under keyword mode.

    Visual subject is hard to detect by keyword list — it's the presence of
    object-level concreteness. Proxy: caption has >= 4 words and contains at
    least one likely noun (i.e. a word > 3 chars that's not a function word).
    """
    if not text:
        return 0
    words = re.findall(r"[А-Яа-яЁё]+", text)
    if len(words) < 4:
        return 0
    function_words = {
        "это", "есть", "была", "был", "были", "будет",
        "только", "также", "которые", "который", "которая",
        "очень", "более", "менее", "вряд", "просто",
    }
    content_words = [w for w in words if len(w) > 3 and w.lower() not in function_words]
    return 1 if content_words else 0


def sds_keyword(description: str) -> dict:
    """Compute keyword-based SDS for one description.

    Returns dict with binary indicator per axis + total SDS.
    """
    text = (description or "").lower()
    indicators = {}
    for axis in AXES:
        if axis == "visual_subject":
            indicators[axis] = detect_visual_subject(description)
        else:
            patterns = AXIS_KEYWORDS[axis]
            indicators[axis] = int(any(re.search(p, text) for p in patterns))
    indicators["sds"] = sum(indicators[a] for a in AXES) / len(AXES)
    return indicators


# ----------------------------------------------------------------------------
# LLM-based detection (Claude Sonnet)
# ----------------------------------------------------------------------------

LLM_SYSTEM_PROMPT = """Ты — библиотечный каталогизатор. Тебе показывают изображение открытки и сгенерированное описание. Твоя задача — определить, покрывает ли описание каждую из 6 семантических осей, существенных для архивного поиска.

Для каждой оси верни 1, если описание явно содержит соответствующую информацию, и 0 — если ось не покрыта или покрыта только косвенно. Будь строг: косвенные намёки и общие фразы не засчитываются.

Оси:
1. type_of_material — явно назван тип архивного материала (открытка, плакат, иллюстрация, фотография, гравюра).
2. visual_subject — конкретно описано, что изображено (люди, объекты, действия, сцена). Абстрактные слова "изображение", "композиция" без объектной конкретики не засчитываются.
3. artistic_style — указана техника или художественная манера (винтаж, ретро, гравюра, живопись, рисунок, чёрно-белая фотография, декоративная иллюстрация).
4. epoch — есть хронологическая привязка (век, десятилетие, период).
5. cultural_context — указан культурный контекст сюжета (религиозный, праздничный, городская/сельская/военная/семейная сцена). Просто "сцена" без культурной привязки не засчитывается.
6. mood — явно охарактеризован эмоциональный тон (радостное, торжественное, ностальгическое и т.п.).

Верни строго JSON:
{
  "type_of_material": 0|1,
  "visual_subject": 0|1,
  "artistic_style": 0|1,
  "epoch": 0|1,
  "cultural_context": 0|1,
  "mood": 0|1,
  "justification": "<1 предложение>"
}"""


def load_anthropic_client():
    try:
        import anthropic
    except ImportError as exc:
        raise ImportError("anthropic not installed. Run: pip install anthropic") from exc
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise EnvironmentError(
            "ANTHROPIC_API_KEY not set. Either export it or use --mode keyword."
        )
    return anthropic.Anthropic()


def encode_image_b64(image_path):
    with open(image_path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8")


def media_type_for(path):
    suffix = Path(path).suffix.lower()
    return {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
    }.get(suffix, "image/jpeg")


def sds_llm_one(client, model, image_path, description):
    message = client.messages.create(
        model=model,
        max_tokens=256,
        system=LLM_SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type_for(image_path),
                            "data": encode_image_b64(image_path),
                        },
                    },
                    {"type": "text", "text": f"Описание:\n\n{description}"},
                ],
            }
        ],
    )
    text = message.content[0].text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        parsed = json.loads(text[start : end + 1])

    indicators = {a: int(parsed.get(a, 0)) for a in AXES}
    indicators["sds"] = sum(indicators[a] for a in AXES) / len(AXES)
    indicators["justification"] = parsed.get("justification", "")
    return indicators


# ----------------------------------------------------------------------------
# Aggregation
# ----------------------------------------------------------------------------

def aggregate_sds(items):
    n = len(items)
    if n == 0:
        return {"n": 0}
    mean_sds = sum(it["sds"]["sds"] for it in items) / n
    per_axis = {
        axis: sum(it["sds"][axis] for it in items) / n for axis in AXES
    }
    # Breakdown by source if available
    by_source = {}
    for it in items:
        src = it.get("source") or "unknown"
        by_source.setdefault(src, []).append(it["sds"]["sds"])
    by_source_mean = {
        src: {"n": len(vals), "mean_sds": sum(vals) / len(vals)}
        for src, vals in by_source.items()
    }
    return {
        "n": n,
        "mean_sds": mean_sds,
        "per_axis_coverage": per_axis,
        "by_source": by_source_mean,
    }


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="metrics_triad_*.json")
    p.add_argument("--output", required=True, help="output JSON with SDS per item + aggregates")
    p.add_argument(
        "--mode",
        choices=["keyword", "llm"],
        default="keyword",
        help="keyword — offline (default); llm — Claude (needs ANTHROPIC_API_KEY)",
    )
    p.add_argument("--model", default="claude-sonnet-4-6", help="LLM model (only used in --mode llm)")
    p.add_argument("--description-key", default="archive_ru")
    p.add_argument("--max-items", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.input) as f:
        payload = json.load(f)
    per_item = payload.get("per_item", [])
    if args.max_items:
        per_item = per_item[: args.max_items]
    print(f"[INFO] Processing {len(per_item)} items in mode '{args.mode}'")

    client = None
    if args.mode == "llm":
        client = load_anthropic_client()

    scored = []
    for i, it in enumerate(per_item):
        desc = it.get(args.description_key) or ""
        image_path = it.get("image_path", "")
        source = it.get("source", "unknown")

        if args.mode == "keyword":
            indicators = sds_keyword(desc)
        else:
            t0 = time.time()
            try:
                indicators = sds_llm_one(client, args.model, image_path, desc)
                dt = time.time() - t0
                print(
                    f"[{i+1}/{len(per_item)}] sds={indicators['sds']:.3f}  "
                    f"({dt:.1f}s)  {Path(image_path).name}"
                )
            except Exception as e:
                print(f"[{i+1}/{len(per_item)}] ERROR: {e}")
                indicators = {a: 0 for a in AXES}
                indicators["sds"] = 0.0
                indicators["error"] = str(e)

        scored.append({
            "image_path": image_path,
            "source": source,
            "description": desc,
            "sds": indicators,
        })

    aggregates = aggregate_sds(scored)
    print("\n=== AGGREGATES ===")
    print(json.dumps(aggregates, indent=2, ensure_ascii=False))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "mode": args.mode,
                "model": args.model if args.mode == "llm" else None,
                "description_key": args.description_key,
                "aggregates": aggregates,
                "per_item": scored,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"\n[INFO] Saved to {out_path}")


if __name__ == "__main__":
    main()
