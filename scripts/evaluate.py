"""evaluate.py — triad evaluation (I-1 + I-2 from docs/experiments.md).

Computes CLIPScore variants per image:
  - CLIPScore_EN(image, caption_en)        — back-compat with baseline reports
  - CLIPScore_RU(image, caption_ru)        — translation isolation
  - CLIPScore_RU(image, archive_description_ru)  — PRIMARY (after I-2)
  - CLIPScore_RU(image, reference_short_ru)      — ceiling indicator (when ref present)

Image encoder: open_clip ViT-B-32 OpenAI (shared with M-CLIP image side).
RU text encoder: M-CLIP/XLM-Roberta-Large-Vit-B-32 by default.
EN text encoder: open_clip ViT-B-32 (existing).

Retrieval R@k (I-5) and bootstrap CI (I-4) are not yet wired in — placeholders.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import open_clip
import torch
from PIL import Image

sys.path.insert(0, "src")

from tsu_image_description.pipeline import ArchiveDescriptionPipeline


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--finetuned",
        type=str,
        default=None,
        help="Path to finetuned BLIP model (substitutes pipeline.caption_generator)",
    )
    parser.add_argument(
        "--mclip-model",
        type=str,
        default="M-CLIP/XLM-Roberta-Large-Vit-B-32",
        help="HuggingFace model name for M-CLIP RU text encoder",
    )
    parser.add_argument(
        "--references",
        type=str,
        default="data/eval/references.jsonl",
        help="Path to references JSONL (one item per line)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/eval/results/metrics_triad_v1.json",
        help="Path to output metrics JSON",
    )
    parser.add_argument(
        "--drop-theme",
        action="store_true",
        help="E-16: drop the 'Предположительно, это X' template sentence",
    )
    parser.add_argument(
        "--drop-mood",
        action="store_true",
        help="E-16: drop the 'Общее настроение …' template sentence",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


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
# CLIP loaders
# ---------------------------------------------------------------------
def load_clip_en(device):
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai"
    )
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    return model, preprocess, tokenizer


class MCLIPTextEncoder(torch.nn.Module):
    """Custom M-CLIP text encoder: XLM-R + linear projection to CLIP space.

    We avoid the `multilingual_clip` package because v1.0.10 conflicts with
    newer transformers (nested from_pretrained inside meta-device init).
    Replicating its tiny architecture (mean-pooled XLM-R + Linear) directly.
    """

    def __init__(self, transformer, linear):
        super().__init__()
        self.transformer = transformer
        self.LinearTransformation = linear


def load_mclip(model_name, device):
    """Load M-CLIP text encoder via direct weight loading.

    Image side is OpenAI ViT-B-32 (already loaded for EN), so we load only
    the multilingual text encoder here. M-CLIP repos contain a single
    state_dict with both `transformer.*` and `LinearTransformation.*` weights.
    """
    from huggingface_hub import hf_hub_download
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    # M-CLIP config has `model_type: "M-CLIP"` which AutoConfig can't resolve,
    # so read the raw JSON ourselves. Defaults match the XLM-R-Large variant.
    config_path = hf_hub_download(repo_id=model_name, filename="config.json")
    with open(config_path) as cf:
        cfg = json.load(cf)
    base_model_name = cfg.get("modelBase", "xlm-roberta-large")
    transformer_dim = cfg.get("transformerDimensions", 1024)
    num_dims = cfg.get("numDims", 512)

    try:
        weights_path = hf_hub_download(repo_id=model_name, filename="model.safetensors")
        from safetensors.torch import load_file
        state = load_file(weights_path)
    except Exception:
        weights_path = hf_hub_download(repo_id=model_name, filename="pytorch_model.bin")
        state = torch.load(weights_path, map_location="cpu")

    base_cfg = AutoConfig.from_pretrained(base_model_name)
    transformer = AutoModel.from_config(base_cfg)

    transformer_state = {
        k[len("transformer."):]: v
        for k, v in state.items()
        if k.startswith("transformer.")
    }
    missing, unexpected = transformer.load_state_dict(transformer_state, strict=False)
    if missing or unexpected:
        # Some XLM-R variants have small differences in head naming; warn but don't fail.
        print(f"[INFO] M-CLIP transformer load: {len(missing)} missing, {len(unexpected)} unexpected keys")

    linear = torch.nn.Linear(transformer_dim, num_dims)
    linear.load_state_dict(
        {
            "weight": state["LinearTransformation.weight"],
            "bias": state["LinearTransformation.bias"],
        }
    )

    text_model = MCLIPTextEncoder(transformer, linear)
    text_model = text_model.to(device).eval()

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    return text_model, tokenizer


# ---------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------
@torch.no_grad()
def encode_image(image_path, clip_model, preprocess, device):
    image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
    feat = clip_model.encode_image(image)
    feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.squeeze(0).cpu().numpy()


@torch.no_grad()
def encode_text_en(text, clip_model, tokenizer, device):
    tokens = tokenizer([text]).to(device)
    feat = clip_model.encode_text(tokens)
    feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.squeeze(0).cpu().numpy()


@torch.no_grad()
def encode_text_ru(text, mclip_model, mclip_tokenizer, device):
    """M-CLIP forward replicated with explicit device handling.

    The library's `MultilingualCLIP.forward(text, tokenizer)` doesn't move
    tokenized inputs to the model's device, which breaks on MPS/CUDA.
    We replicate the logic here with explicit `.to(device)`.
    """
    tok = mclip_tokenizer([text], padding=True, return_tensors="pt")
    tok = {k: v.to(device) for k, v in tok.items()}

    embs = mclip_model.transformer(**tok)[0]
    att = tok["attention_mask"]
    pooled = (embs * att.unsqueeze(2)).sum(dim=1) / att.sum(dim=1)[:, None]
    feat = mclip_model.LinearTransformation(pooled)

    feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.squeeze(0).cpu().numpy()


def cosine(a, b):
    return float(np.dot(a, b))


# ---------------------------------------------------------------------
# Retrieval (I-5)
# ---------------------------------------------------------------------
def compute_retrieval(img_matrix, txt_matrix, ks=(1, 5, 10)):
    """R@k for image ↔ archive_description retrieval over the eval pool.

    Reports both directions:
      - i2t: given image, rank correct description.
      - t2i: given description (search query), rank correct image.

    The latter is the closer proxy for the library archive use case.
    """
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
        {"rank_i2t": ri, "rank_t2i": rt} for ri, rt in zip(ranks_i2t, ranks_t2i)
    ]
    return {"aggregates": aggregates, "ranks": per_item_ranks}


# ---------------------------------------------------------------------
# Pipeline setup
# ---------------------------------------------------------------------
def build_pipeline(finetuned_path, device, builder_kwargs=None):
    pipeline = ArchiveDescriptionPipeline(
        model_path=finetuned_path,
        builder_kwargs=builder_kwargs,
    )

    if finetuned_path:
        from transformers import BlipProcessor, BlipForConditionalGeneration

        print(f"[INFO] Substituting caption_generator with {finetuned_path}")
        processor = BlipProcessor.from_pretrained(finetuned_path)
        model = BlipForConditionalGeneration.from_pretrained(finetuned_path).to(device)
        pipeline.caption_generator.model = model
        pipeline.caption_generator.processor = processor

    return pipeline


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    args = parse_args()
    device = get_device()

    refs = load_references(args.references)
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Eval items: {len(refs)}")

    # Pipeline
    builder_kwargs = {
        "include_theme": not args.drop_theme,
        "include_mood": not args.drop_mood,
    }
    print(
        f"\n[INFO] Loading pipeline (finetuned={args.finetuned}, "
        f"include_theme={builder_kwargs['include_theme']}, "
        f"include_mood={builder_kwargs['include_mood']})..."
    )
    pipeline = build_pipeline(args.finetuned, device, builder_kwargs=builder_kwargs)

    # Scorers
    print("[INFO] Loading EN CLIP (open_clip ViT-B-32 openai)...")
    clip_en, preprocess, tokenizer_en = load_clip_en(device)

    print(f"[INFO] Loading RU CLIP ({args.mclip_model})...")
    mclip_text, mclip_tokenizer = load_mclip(args.mclip_model, device)

    # Loop
    per_item = []
    # Parallel arrays for retrieval R@k (I-5). Excluded from JSON output.
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
        archive_ru = result.get("archive_description", "") or ""

        # Image embedding (shared between EN and RU CLIP, since image side is same)
        img_emb = encode_image(image_path, clip_en, preprocess, device)

        # Pre-compute the archive RU embedding once: used both for CLIPScore_RU(archive)
        # and for retrieval R@k below.
        archive_emb = (
            encode_text_ru(archive_ru, mclip_text, mclip_tokenizer, device)
            if archive_ru else None
        )

        scores = {
            "CLIPScore_EN_caption_en": (
                cosine(img_emb, encode_text_en(caption_en, clip_en, tokenizer_en, device))
                if caption_en else None
            ),
            "CLIPScore_RU_caption_ru": (
                cosine(img_emb, encode_text_ru(caption_ru, mclip_text, mclip_tokenizer, device))
                if caption_ru else None
            ),
            "CLIPScore_RU_archive_ru": (
                cosine(img_emb, archive_emb) if archive_emb is not None else None
            ),
            "CLIPScore_RU_reference_ru": (
                cosine(img_emb, encode_text_ru(ref_ru, mclip_text, mclip_tokenizer, device))
                if ref_ru else None
            ),
        }

        img_embs.append(img_emb)
        archive_embs.append(archive_emb if archive_emb is not None else np.zeros_like(img_emb))

        print(f"[{i + 1}/{len(refs)}] {Path(image_path).name}")
        print(f"  caption_en: {caption_en}")
        print(f"  caption_ru: {caption_ru}")
        print(f"  archive:    {archive_ru[:140]}{'…' if len(archive_ru) > 140 else ''}")
        for k, v in scores.items():
            if v is not None:
                print(f"  {k:32s}: {v:.4f}")
        print(f"  latency: {elapsed:.2f}s\n")

        per_item.append(
            {
                "image_path": image_path,
                "caption_en": caption_en,
                "caption_ru": caption_ru,
                "archive_ru": archive_ru,
                "reference_ru": ref_ru,
                "scores": scores,
                "latency_sec": elapsed,
            }
        )

    # Aggregate
    def mean_of(key):
        vals = [it["scores"][key] for it in per_item if it["scores"].get(key) is not None]
        return mean(vals)

    # Retrieval R@k (I-5) — image ↔ archive_description_ru over the eval pool.
    retrieval = compute_retrieval(np.stack(img_embs), np.stack(archive_embs))
    for it, ranks in zip(per_item, retrieval["ranks"]):
        it["retrieval"] = ranks

    summary = {
        "num_examples": len(refs),
        "primary_metric": "CLIPScore_RU_archive_ru",
        "config": {
            "finetuned": args.finetuned,
            "builder_kwargs": builder_kwargs,
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
        "retrieval": retrieval["aggregates"],
        "latency": {
            "mean_sec": total_time / len(refs) if refs else None,
            "images_per_sec": len(refs) / total_time if total_time > 0 else None,
        },
        "notes": {
            "I-1": "M-CLIP integrated; primary metric switched to CLIPScore_RU(archive).",
            "I-2": "archive_description_ru declared as primary; old metric kept as back-compat.",
            "I-4": "bootstrap CI implemented in scripts/eval_stats.py (not invoked here)",
            "I-5": "retrieval R@1 / R@5 over eval pool computed below",
        },
    }

    print("\n=== FINAL RESULTS (triad partial; I-1 + I-2) ===")
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
