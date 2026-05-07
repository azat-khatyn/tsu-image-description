from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


def load_captions(path: Path) -> List[Dict[str, str]]:
    data = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(data, list):
        raise ValueError("Captions file must contain a JSON list")

    rows: List[Dict[str, str]] = []
    for i, item in enumerate(data):
        if isinstance(item, str):
            rows.append({"name": f"caption_{i+1}", "text": item})
        elif isinstance(item, dict):
            text = item.get("text")
            if not text:
                raise ValueError(f"Caption #{i+1} has no 'text'")
            rows.append(
                {
                    "name": str(item.get("name", f"caption_{i+1}")),
                    "text": str(text),
                }
            )
        else:
            raise ValueError("Unsupported captions JSON format")

    return rows


def lcs_length(a: List[str], b: List[str]) -> int:
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n):
        for j in range(m):
            if a[i] == b[j]:
                dp[i + 1][j + 1] = dp[i][j] + 1
            else:
                dp[i + 1][j + 1] = max(dp[i][j + 1], dp[i + 1][j])
    return dp[n][m]


def rouge_l_f1(candidate: str, reference: str) -> float:
    cand_tokens = candidate.lower().split()
    ref_tokens = reference.lower().split()

    if not cand_tokens or not ref_tokens:
        return 0.0

    lcs = lcs_length(cand_tokens, ref_tokens)
    precision = lcs / len(cand_tokens)
    recall = lcs / len(ref_tokens)

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


def try_bertscore(candidates: List[str], references: List[str]) -> Optional[List[float]]:
    try:
        from bert_score import score as bert_score
    except Exception:
        return None

    _, _, f1 = bert_score(
        candidates,
        references,
        lang="en",
        verbose=False,
        rescale_with_baseline=False,
    )
    return [float(x) for x in f1]


def compute_clipscores(
    image: Image.Image,
    captions: List[str],
    model_name: str = "openai/clip-vit-base-patch32",
) -> List[float]:
    """
    Надёжный вариант через обычный forward CLIPModel.
    Берём logits_per_image и масштабируем их в CLIPScore-like формат.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name).to(device)
    model.eval()

    with torch.no_grad():
        inputs = processor(
            text=captions,
            images=[image] * len(captions),
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        outputs = model(**inputs)

        # logits_per_image: shape [batch_images, batch_texts]
        # так как images=[image] * len(captions), диагональ соответствует нашим парам
        logits_per_image = outputs.logits_per_image

        if logits_per_image.ndim != 2:
            raise RuntimeError(f"Unexpected logits_per_image shape: {tuple(logits_per_image.shape)}")

        if logits_per_image.shape[0] != logits_per_image.shape[1]:
            # fallback: если по какой-то причине форма не квадратная,
            # берём первую строку как similarity одного изображения ко всем текстам
            scores = logits_per_image[0].detach().cpu().tolist()
        else:
            scores = torch.diag(logits_per_image).detach().cpu().tolist()

    return [float(s) for s in scores]


def print_results(rows: List[Dict[str, Any]]) -> None:
    headers = ["name", "clipscore", "rouge_l", "bertscore", "text"]
    widths = {
        "name": 18,
        "clipscore": 12,
        "rouge_l": 10,
        "bertscore": 10,
        "text": 80,
    }

    line = " | ".join(h.ljust(widths[h]) for h in headers)
    print(line)
    print("-" * len(line))

    for row in rows:
        text_preview = row["text"][:77] + "..." if len(row["text"]) > 80 else row["text"]
        vals = {
            "name": str(row["name"])[: widths["name"]],
            "clipscore": f"{row['clipscore']:.4f}",
            "rouge_l": "-" if row["rouge_l"] is None else f"{row['rouge_l']:.4f}",
            "bertscore": "-" if row["bertscore"] is None else f"{row['bertscore']:.4f}",
            "text": text_preview,
        }
        print(" | ".join(vals[h].ljust(widths[h]) for h in headers))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate several caption candidates for one image."
    )
    parser.add_argument("--image", type=str, required=True, help="Path to image")
    parser.add_argument(
        "--captions",
        type=str,
        required=True,
        help="Path to JSON file with candidate captions",
    )
    parser.add_argument(
        "--reference",
        type=str,
        default=None,
        help="Optional gold/reference caption for text-based metrics",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="openai/clip-vit-base-patch32",
        help="CLIP model name for CLIPScore-like metric",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save results JSON",
    )

    args = parser.parse_args()

    image = Image.open(args.image).convert("RGB")
    caption_rows = load_captions(Path(args.captions))
    texts = [row["text"] for row in caption_rows]

    clip_scores = compute_clipscores(
        image=image,
        captions=texts,
        model_name=args.model_name,
    )

    rouge_scores: List[Optional[float]] = [None] * len(texts)
    bert_scores: List[Optional[float]] = [None] * len(texts)

    if args.reference:
        rouge_scores = [rouge_l_f1(text, args.reference) for text in texts]

        bert = try_bertscore(texts, [args.reference] * len(texts))
        if bert is not None:
            bert_scores = bert

    results = []
    for row, clip_s, rouge_s, bert_s in zip(caption_rows, clip_scores, rouge_scores, bert_scores):
        results.append(
            {
                "name": row["name"],
                "text": row["text"],
                "clipscore": clip_s,
                "rouge_l": rouge_s,
                "bertscore": bert_s,
            }
        )

    results = sorted(results, key=lambda x: x["clipscore"], reverse=True)

    print_results(results)

    if args.output:
        Path(args.output).write_text(
            json.dumps(results, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\nSaved results to: {args.output}")


if __name__ == "__main__":
    main()
