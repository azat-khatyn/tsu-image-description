import os
import time
import json
import torch
import numpy as np
from PIL import Image

import open_clip
from transformers import BlipProcessor, BlipForConditionalGeneration
from peft import PeftModel

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


# =========================
# CONFIG
# =========================
IMAGE_DIR = "data/eval/"
LORA_PATH = "models/blip_caplift_v3_lora_epoch_1"

# =========================
# LOAD MODEL (LoRA)
# =========================
def load_model():
    print("[INFO] Loading base BLIP...")

    base_model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )

    print("[INFO] Loading LoRA adapter...")

    model = PeftModel.from_pretrained(base_model, LORA_PATH)
    model.to(DEVICE)
    model.eval()

    processor = BlipProcessor.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )

    return model, processor


# =========================
# LOAD CLIP (for score)
# =========================
def load_clip():
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32",
        pretrained="openai"
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-32")

    return model.to(DEVICE).eval(), preprocess, tokenizer


# =========================
# GENERATE CAPTION
# =========================
def generate_caption(model, processor, image_path):
    image = Image.open(image_path).convert("RGB")

    inputs = processor(images=image, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=50,
            num_beams=5,
            length_penalty=1.2
        )

    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption


# =========================
# CLIP SCORE
# =========================
def clip_score(model, preprocess, tokenizer, image_path, text):
    image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(DEVICE)
    text_tokens = tokenizer([text]).to(DEVICE)

    with torch.no_grad():
        img_feat = model.encode_image(image)
        txt_feat = model.encode_text(text_tokens)

        img_feat /= img_feat.norm(dim=-1, keepdim=True)
        txt_feat /= txt_feat.norm(dim=-1, keepdim=True)

    return float((img_feat @ txt_feat.T).item())


# =========================
# MAIN
# =========================
def main():
    print("\n--- RUNNING LoRA EVALUATION ---\n")

    model, processor = load_model()
    clip_model, preprocess, tokenizer = load_clip()

    image_files = [
        f for f in sorted(os.listdir(IMAGE_DIR))
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    scores = []
    times = []

    for img_name in image_files:
        img_path = os.path.join(IMAGE_DIR, img_name)

        start = time.time()

        caption = generate_caption(model, processor, img_path)
        score = clip_score(clip_model, preprocess, tokenizer, img_path, caption)

        elapsed = time.time() - start

        scores.append(score)
        times.append(elapsed)

        print(f"{img_name}")
        print(f"→ {caption}")
        print(f"CLIPScore: {round(score, 4)}\n")

    results = {
        "num_examples": len(scores),
        "CLIPScore_mean": float(np.mean(scores)),
        "Latency_mean_sec": float(np.mean(times)),
        "Images_per_sec": float(1.0 / np.mean(times))
    }

    print("\n=== FINAL RESULTS ===")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
