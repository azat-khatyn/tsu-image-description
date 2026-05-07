import json
import time
from pathlib import Path
import argparse

import numpy as np
import torch
from PIL import Image
import open_clip

import sys
sys.path.insert(0, "src")

from tsu_image_description.pipeline import ArchiveDescriptionPipeline


# ---------------------------
# ARGS
# ---------------------------
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--finetuned",
        type=str,
        default=None,
        help="Path to finetuned BLIP model"
    )

    return parser.parse_args()


# ---------------------------
# HELPERS
# ---------------------------
def load_references(path: str):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items


def mean(values):
    return sum(values) / len(values) if values else 0.0


# ---------------------------
# CLIP
# ---------------------------
def encode_clip_image(image_path, model, preprocess, device):
    image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = model.encode_image(image)
        feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.squeeze(0).cpu().numpy()


def encode_clip_text(text, model, tokenizer, device):
    tokens = tokenizer([text]).to(device)
    with torch.no_grad():
        feat = model.encode_text(tokens)
        feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.squeeze(0).cpu().numpy()


def compute_clipscore(img_emb, txt_emb):
    return float(np.dot(img_emb, txt_emb))


# ---------------------------
# MAIN
# ---------------------------
def main():
    args = parse_args()

    references_path = "data/eval/references.jsonl"
    refs = load_references(references_path)

    # ---------------------------
    # PIPELINE
    # ---------------------------
    # pipeline = ArchiveDescriptionPipeline()

    pipeline = ArchiveDescriptionPipeline(
        model_path=args.finetuned
    )
    device = (
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    if args.finetuned:
        from transformers import BlipProcessor, BlipForConditionalGeneration

        print(f"\n[INFO] Using finetuned model: {args.finetuned}\n")

        processor = BlipProcessor.from_pretrained(args.finetuned)
        model = BlipForConditionalGeneration.from_pretrained(args.finetuned).to(device)

        # 🔥 ПРАВИЛЬНАЯ ПОДМЕНА
        pipeline.model = model
        pipeline.processor = processor

        print("[DEBUG] pipeline.model replaced with finetuned model\n")

    # ---------------------------
    # CLIP MODEL
    # ---------------------------
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32",
        pretrained="openai",
    )
    clip_model = clip_model.to(device).eval()
    tokenizer = open_clip.get_tokenizer("ViT-B-32")

    # ---------------------------
    # LOOP
    # ---------------------------
    clip_scores = []
    total_time = 0.0

    print("\n--- RUNNING EVALUATION ---\n")

    for item in refs:
        image_path = item["image_path"]

        t0 = time.time()
        result = pipeline.run(image_path)
        elapsed = time.time() - t0

        caption_en = result["caption"].get("en", "")

        # DEBUG
        print(f"{Path(image_path).name}")
        print("→", caption_en)

        # embeddings
        img_emb = encode_clip_image(image_path, clip_model, clip_preprocess, device)
        txt_emb = encode_clip_text(caption_en, clip_model, tokenizer, device)

        score = compute_clipscore(img_emb, txt_emb)

        print("CLIPScore:", round(score, 4))
        print()

        clip_scores.append(score)
        total_time += elapsed

    # ---------------------------
    # RESULTS
    # ---------------------------
    summary = {
        "num_examples": len(refs),
        "CLIPScore_mean": float(mean(clip_scores)),
        "Latency_mean_sec": total_time / len(refs),
        "Images_per_sec": len(refs) / total_time,
    }

    print("\n=== FINAL RESULTS ===")
    print(json.dumps(summary, indent=2))

    # save
    out_dir = Path("data/eval/results")
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "metrics_summary_v1.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
