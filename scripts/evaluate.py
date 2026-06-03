"""evaluate.py — reference-free оценка архивных описаний.

Поддерживает:
  - бэкенды генерации caption_en: BLIP-1 / BLIP-2
  - набор RGB-20 с разметкой
  - пул NYPL-200
  - объединённый прогон ALL-220 (retrieval / устойчивость)
  - варианты CLIPScore
  - retrieval R@k (изображение <-> архивное описание)

Основная метрика:
  - CLIPScore_RU(изображение, archive_description_ru)

Рекомендуемые сценарии:
  - RGB-20 — качество отображения
  - NYPL-200 — устойчивость
  - ALL-220 — retrieval
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from tsu_image_description.pipeline import ArchiveDescriptionPipeline
from tsu_image_description.clip_scorer import CLIPScorer
from tsu_image_description.models import get_device


# ---------------------------------------------------------------------
# Аргументы CLI
# ---------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--finetuned",
        type=str,
        default=None,
        help="Path to a full finetuned BLIP checkpoint (BLIP-1 only).",
    )
    parser.add_argument(
        "--caption-backend",
        type=str,
        choices=["blip1", "blip2"],
        default="blip1",
        help="Captioning backend.",
    )
    parser.add_argument(
        "--mclip-model",
        type=str,
        default="M-CLIP/XLM-Roberta-Large-Vit-B-32",
        help="HuggingFace model name for M-CLIP RU text encoder.",
    )
    parser.add_argument(
        "--references",
        type=str,
        default="data/eval/references.jsonl",
        help="Path to annotated references JSONL.",
    )
    parser.add_argument(
        "--pool",
        type=str,
        default=None,
        help="Optional retrieval pool JSONL appended to --references.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/eval/results/metrics_triad_run.json",
        help="Path to output metrics JSON.",
    )
    parser.add_argument(
        "--drop-theme",
        action="store_true",
        help="Drop theme sentence from archive_description.",
    )
    parser.add_argument(
        "--drop-mood",
        action="store_true",
        help="Drop mood sentence from archive_description.",
    )
    parser.add_argument(
        "--template-mode",
        type=str,
        choices=["full", "minimal", "caption_only"],
        default="full",
        help="Template for archive_description.",
    )
    parser.add_argument(
        "--drop-generic-style",
        action="store_true",
        help="E05d: drop generic style labels (vintage/decorative/retro) from intro phrase. "
             "No-op when taxonomy_version=archival_v2 (no generic labels exist).",
    )
    parser.add_argument(
        "--taxonomy-version",
        type=str,
        default="archival_v2",
        choices=["legacy_v1", "archival_v2"],
        help="SigLIP taxonomy version. legacy_v1 reproduces pre-E06b experiments; "
             "archival_v2 uses Файнштейн / MARC 21 / Getty AAT archival vocabulary.",
    )
    parser.add_argument(
        "--use-llm-rewriter",
        action="store_true",
        help="E12: route archive description through a local LLM rewriter.",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default="Vikhrmodels/Vikhr-Nemo-12B-Instruct-R-21-09-24",
        help="LLM rewriter model path (only used with --use-llm-rewriter).",
    )
    parser.add_argument(
        "--llm-prompt-style",
        type=str,
        choices=["v1_archival", "v2_curator"],
        default="v1_archival",
        help="LLM rewriter prompt style (only used with --use-llm-rewriter). "
             "v1_archival = E12 baseline (preamble: 'Открытка-хромолитография. ...'). "
             "v2_curator = E13, calibrated to НЭБ field 327 (no material-type preamble).",
    )
    parser.add_argument(
        "--translator-model",
        type=str,
        default=None,
        help="Override EN→RU translator model.",
    )
    parser.add_argument(
        "--num-beams",
        type=int,
        default=1,
        help="Caption decoding beam count.",
    )
    parser.add_argument(
        "--length-penalty",
        type=float,
        default=1.0,
        help="Caption decoding length penalty.",
    )
    parser.add_argument(
        "--prompt-prefix",
        type=str,
        default=None,
        help="Optional English prompt prefix for captioning.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=50,
        help="Max new tokens for caption generation.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------
def load_references(path):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def mean(values):
    return float(np.mean(values)) if values else None


# ---------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------
def compute_retrieval(img_matrix, txt_matrix, ks=(1, 5, 10)):
    """R@k для retrieval изображение <-> archive_description."""
    n = img_matrix.shape[0]
    sim = img_matrix @ txt_matrix.T

    ranks_i2t = []
    ranks_t2i = []

    for i in range(n):
        order_i2t = np.argsort(-sim[i])
        ranks_i2t.append(int(np.where(order_i2t == i)[0][0]) + 1)

        order_t2i = np.argsort(-sim[:, i])
        ranks_t2i.append(int(np.where(order_t2i == i)[0][0]) + 1)

    def at_k(ranks, k):
        return float(np.mean([r <= k for r in ranks]))

    aggregates = {
        "n_pool": n,
        "i2t": {f"R@{k}": at_k(ranks_i2t, k) for k in ks},
        "t2i": {f"R@{k}": at_k(ranks_t2i, k) for k in ks},
        "mean_rank_i2t": float(np.mean(ranks_i2t)),
        "mean_rank_t2i": float(np.mean(ranks_t2i)),
    }

    per_item_ranks = [
        {"rank_i2t": ri, "rank_t2i": rt}
        for ri, rt in zip(ranks_i2t, ranks_t2i)
    ]

    return {"aggregates": aggregates, "ranks": per_item_ranks}


# ---------------------------------------------------------------------
# Сборка pipeline
# ---------------------------------------------------------------------
def build_pipeline(
    *,
    finetuned_path=None,
    builder_kwargs=None,
    translator_model=None,
    caption_kwargs=None,
    use_llm_rewriter=False,
    llm_rewriter_kwargs=None,
    taxonomy_version: str = "archival_v2",
):
    pipeline = ArchiveDescriptionPipeline(
        model_path=finetuned_path,
        caption_kwargs=caption_kwargs,
        translator_model=translator_model,
        use_llm_rewriter=use_llm_rewriter,
        llm_rewriter_kwargs=llm_rewriter_kwargs,
        builder_kwargs=builder_kwargs,
        taxonomy_version=taxonomy_version,
    )
    return pipeline


# ---------------------------------------------------------------------
# Точка входа
# ---------------------------------------------------------------------
def main():
    args = parse_args()
    device = get_device()

    refs = load_references(args.references)
    n_annotated = len(refs)

    if args.pool:
        pool_items = load_references(args.pool)
        refs = refs + pool_items
        print(
            f"[INFO] Loaded {n_annotated} annotated + {len(pool_items)} pool items "
            f"= {len(refs)} total"
        )
    else:
        print(f"[INFO] Loaded {n_annotated} annotated items")

    print(f"[INFO] Device: {device}")
    print(f"[INFO] Eval items: {len(refs)}")

    builder_kwargs = {
        "template_mode": args.template_mode,
        "include_theme": not args.drop_theme,
        "include_mood": not args.drop_mood,
        "drop_generic_style": args.drop_generic_style,
    }

    caption_kwargs = {
        "backend": args.caption_backend,
        "num_beams": args.num_beams,
        "length_penalty": args.length_penalty,
        "prompt_prefix": args.prompt_prefix,
        "max_new_tokens": args.max_new_tokens,
    }

    finetuned_path = getattr(args, "finetuned", None)

    print(
        f"\n[INFO] Loading pipeline (finetuned={finetuned_path}, "
        f"template_mode={builder_kwargs['template_mode']}, "
        f"caption_kwargs={caption_kwargs})..."
    )

    pipeline = build_pipeline(
        finetuned_path=finetuned_path,
        builder_kwargs=builder_kwargs,
        translator_model=args.translator_model,
        caption_kwargs=caption_kwargs,
        use_llm_rewriter=args.use_llm_rewriter,
        llm_rewriter_kwargs={
            "model_path": args.llm_model,
            "prompt_style": args.llm_prompt_style,
        } if args.use_llm_rewriter else None,
        taxonomy_version=args.taxonomy_version,
    )

    print(f"[INFO] Loading CLIPScorer (EN ViT-B-32 + RU {args.mclip_model})...")
    scorer = CLIPScorer(mclip_model=args.mclip_model, device=device)

    per_item = []
    img_embs = []
    archive_embs = []
    total_time = 0.0

    print("\n--- RUNNING EVALUATION ---\n")

    for i, item in enumerate(refs):
        image_path = item["image_path"]
        ref_ru = (item.get("reference_short_ru") or "").strip()

        t0 = time.time()
        result = pipeline.run(image_path)
        elapsed = time.time() - t0
        total_time += elapsed

        caption_en = result["caption"].get("en", "") or ""
        caption_ru = result["caption"].get("ru", "") or ""
        caption_ru_raw = result["caption"].get("ru_raw", "") or ""
        archive_ru = result.get("archive_description", "") or ""

        img_emb = scorer.encode_image(image_path)

        archive_emb = (
            scorer.encode_text(archive_ru, "ru")
            if archive_ru
            else None
        )

        scores = {
            "CLIPScore_EN_caption_en": (
                scorer.cosine(img_emb, scorer.encode_text(caption_en, "en"))
                if caption_en
                else None
            ),
            "CLIPScore_RU_caption_ru": (
                scorer.cosine(img_emb, scorer.encode_text(caption_ru, "ru"))
                if caption_ru
                else None
            ),
            "CLIPScore_RU_archive_ru": (
                scorer.cosine(img_emb, archive_emb) if archive_emb is not None else None
            ),
            "CLIPScore_RU_reference_ru": (
                scorer.cosine(img_emb, scorer.encode_text(ref_ru, "ru"))
                if ref_ru
                else None
            ),
        }

        img_embs.append(img_emb)
        archive_embs.append(archive_emb if archive_emb is not None else np.zeros_like(img_emb))

        print(f"[{i + 1}/{len(refs)}] {Path(image_path).name}")
        print(f"  caption_en:    {caption_en}")
        print(f"  caption_ru:    {caption_ru}")
        if caption_ru_raw and caption_ru_raw != caption_ru:
            print(f"  caption_ru_raw:{caption_ru_raw}")
        print(f"  archive:       {archive_ru[:140]}{'…' if len(archive_ru) > 140 else ''}")
        for k, v in scores.items():
            if v is not None:
                print(f"  {k:32s}: {v:.4f}")
        print(f"  latency: {elapsed:.2f}s\n")

        per_item.append(
            {
                "image_path": image_path,
                "source": item.get("source", "rgb"),
                "caption_en": caption_en,
                "caption_ru": caption_ru,
                "caption_ru_raw": caption_ru_raw,
                "archive_ru": archive_ru,
                "reference_ru": ref_ru,
                "scores": scores,
                "latency_sec": elapsed,
            }
        )

    def mean_of(key, source_filter=None):
        vals = [
            it["scores"][key]
            for it in per_item
            if it["scores"].get(key) is not None
            and (source_filter is None or it.get("source") == source_filter)
        ]
        return mean(vals)

    retrieval = compute_retrieval(np.stack(img_embs), np.stack(archive_embs))
    for it, ranks in zip(per_item, retrieval["ranks"]):
        it["retrieval"] = ranks

    sources = sorted(set(it.get("source", "rgb") for it in per_item))
    by_source = {}
    for src in sources:
        n_src = sum(1 for it in per_item if it.get("source") == src)
        by_source[src] = {
            "n": n_src,
            "CLIPScore_RU_archive_ru": mean_of("CLIPScore_RU_archive_ru", src),
            "CLIPScore_RU_caption_ru": mean_of("CLIPScore_RU_caption_ru", src),
            "CLIPScore_EN_caption_en": mean_of("CLIPScore_EN_caption_en", src),
        }

    summary = {
        "num_examples": len(refs),
        "primary_metric": "CLIPScore_RU_archive_ru",
        "config": {
            "finetuned": finetuned_path,
            "caption_backend": args.caption_backend,
            "translator_model": args.translator_model or "Helsinki-NLP/opus-mt-en-ru",
            "builder_kwargs": builder_kwargs,
            "caption_kwargs": caption_kwargs,
            "taxonomy_version": args.taxonomy_version,
            "use_llm_rewriter": args.use_llm_rewriter,
            "llm_model": args.llm_model if args.use_llm_rewriter else None,
            "llm_prompt_style": args.llm_prompt_style if args.use_llm_rewriter else None,
            "references_path": args.references,
            "pool_path": args.pool,
        },
        "scorers": {
            "EN": "open_clip ViT-B-32 (openai)",
            "RU": args.mclip_model,
        },
        "metrics": {
            "CLIPScore_EN_caption_en": mean_of("CLIPScore_EN_caption_en"),
            "CLIPScore_RU_caption_ru": mean_of("CLIPScore_RU_caption_ru"),
            "CLIPScore_RU_archive_ru": mean_of("CLIPScore_RU_archive_ru"),
            "CLIPScore_RU_reference_ru": mean_of("CLIPScore_RU_reference_ru"),
        },
        "by_source": by_source,
        "retrieval": retrieval["aggregates"],
        "latency": {
            "mean_sec": total_time / len(refs) if refs else None,
            "images_per_sec": len(refs) / total_time if total_time > 0 else None,
        },
    }

    print("\n=== FINAL RESULTS ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {"summary": summary, "per_item": per_item},
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"\n[INFO] Saved metrics to {out_path}")


if __name__ == "__main__":
    main()
