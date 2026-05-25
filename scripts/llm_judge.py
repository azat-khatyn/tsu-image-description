"""llm_judge.py — LLM-as-judge evaluator.

Берёт результаты пайплайна (`per_item` из metrics_triad_*.json) и для каждой
открытки запрашивает у LLM оценку по фиксированной рубрике, заданной
константой `SYSTEM_PROMPT` ниже в этом файле.

ВАЖНО: рубрика и system prompt замораживаются ДО запуска основной серии
экспериментов. После freeze правки запрещены — иначе LLM-judge становится
circular metric.

Зависимости:
    pip install anthropic

Переменные окружения:
    ANTHROPIC_API_KEY=...
"""

import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path

# Default model is Claude Sonnet 4.6 — strong + cost-reasonable for ~200 calls.
# Can override via --model flag. Frozen for the experiment series after I-6 pilot.
DEFAULT_MODEL = "claude-sonnet-4-6"


# Рубрика заморожена и хранится здесь как единственный источник истины.
# Любые правки требуют новой серии прогонов для согласованности с прошлыми результатами.
SYSTEM_PROMPT = """Ты — библиотечный каталогизатор. Тебе показывают изображение открытки из архива и сгенерированное описание.
Твоя задача — оценить описание по 5 критериям, каждый по шкале 1–5.

Критерии:
1. Faithfulness — точность по отношению к изображению (5 — все детали верны; 1 — есть грубые ошибки/галлюцинации).
2. Completeness — полнота: упомянуты ли ключевые визуальные элементы (5 — покрыты сюжет/тип/стиль/детали; 1 — слишком общее).
3. Style — стилевая уместность для библиотечного каталога (5 — нейтральный описательный стиль; 1 — эмоционально/разговорно/с ошибками).
4. No-hallucinations — отсутствие выдуманных деталей (5 — нет утверждений вне изображения; 1 — очевидные выдумки).
5. Brevity — компактность при сохранении информативности (5 — 1–3 предложения, каждое информативно; 1 — односложно или с водой).

Верни ответ строго в JSON формате:
{
  "faithfulness": <int 1-5>,
  "completeness": <int 1-5>,
  "style": <int 1-5>,
  "no_hallucinations": <int 1-5>,
  "brevity": <int 1-5>,
  "justification": "<1-2 предложения о ключевых наблюдениях>"
}

Будь строг и последователен. Не делай скидок на то, что описание автоматически сгенерировано."""


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input",
        required=True,
        help="metrics_triad_*.json from evaluate.py (must contain per_item with archive_ru and image_path)",
    )
    p.add_argument(
        "--output",
        required=True,
        help="output JSON with per-item judge scores",
    )
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--max-items", type=int, default=None,
                   help="limit (useful for pilot)")
    p.add_argument("--description-key", default="archive_ru",
                   help="which generated text to judge (default: archive_ru)")
    return p.parse_args()


def load_anthropic_client():
    try:
        import anthropic
    except ImportError as exc:
        raise ImportError(
            "anthropic not installed. Run: pip install anthropic"
        ) from exc

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY not set. Set it before running:\n"
            "    export ANTHROPIC_API_KEY=sk-..."
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
        ".gif": "image/gif",
    }.get(suffix, "image/jpeg")


def judge_one(client, model, image_path, description):
    """Call the LLM judge for one (image, description) pair, return scores dict."""
    message = client.messages.create(
        model=model,
        max_tokens=512,
        system=SYSTEM_PROMPT,
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
                    {
                        "type": "text",
                        "text": f"Описание для оценки:\n\n{description}",
                    },
                ],
            }
        ],
    )

    text = message.content[0].text.strip()
    # Strip ```json ... ``` if present
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        scores = json.loads(text)
    except json.JSONDecodeError:
        # Salvage: find JSON-looking block
        start = text.find("{")
        end = text.rfind("}")
        scores = json.loads(text[start : end + 1])

    return scores, text


def aggregate(scored_items):
    keys = ["faithfulness", "completeness", "style", "no_hallucinations", "brevity"]
    out = {}
    for k in keys:
        vals = [it["scores"][k] for it in scored_items if it["scores"].get(k) is not None]
        out[k] = sum(vals) / len(vals) if vals else None

    # Composite = mean across all 5 dimensions for each item, then mean across items
    composite_per_item = []
    for it in scored_items:
        vals = [it["scores"][k] for k in keys if it["scores"].get(k) is not None]
        if vals:
            composite_per_item.append(sum(vals) / len(vals))
    out["composite"] = (
        sum(composite_per_item) / len(composite_per_item) if composite_per_item else None
    )
    return out


def main():
    args = parse_args()

    with open(args.input) as f:
        payload = json.load(f)
    items = payload.get("per_item", [])
    if args.max_items:
        items = items[: args.max_items]
    print(f"[INFO] {len(items)} items to judge with model {args.model}")

    client = load_anthropic_client()

    scored = []
    for i, it in enumerate(items):
        image_path = it["image_path"]
        desc = it.get(args.description_key) or ""
        if not desc:
            print(f"[{i+1}/{len(items)}] {Path(image_path).name}: empty description, skip")
            continue

        t0 = time.time()
        try:
            scores, raw = judge_one(client, args.model, image_path, desc)
        except Exception as e:
            print(f"[{i+1}/{len(items)}] ERROR on {image_path}: {e}")
            continue
        dt = time.time() - t0

        composite = sum(
            scores.get(k, 0)
            for k in ("faithfulness", "completeness", "style", "no_hallucinations", "brevity")
        ) / 5
        print(
            f"[{i+1}/{len(items)}] {Path(image_path).name}  "
            f"composite={composite:.2f}  ({dt:.1f}s)"
        )

        scored.append(
            {
                "image_path": image_path,
                "description": desc,
                "scores": scores,
                "raw_response": raw,
                "elapsed_sec": dt,
            }
        )

    aggregates = aggregate(scored)
    print("\n=== AGGREGATES ===")
    print(json.dumps(aggregates, indent=2))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": args.model,
                "description_key": args.description_key,
                "system_prompt_hash": hash(SYSTEM_PROMPT),
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
