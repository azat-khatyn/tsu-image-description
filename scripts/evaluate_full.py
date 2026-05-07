import os
import json
import torch
import numpy as np
from PIL import Image

import open_clip
from transformers import BlipProcessor, BlipForConditionalGeneration
from peft import PeftModel

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
IMAGE_DIR = "data/eval/"


# =========================
# CONFIG: все модели
# =========================
MODELS = {
    "baseline": {
        "type": "base",
        "path": "Salesforce/blip-image-captioning-base"
    },
    "caplift_v1": {
        "type": "full",
        "path": "models/blip_nypl_epoch_1"
    },
    "caplift_v2": {
        "type": "full",
        "path": "models/blip_caplift_v2_epoch_1"
    },
    "lora_v3": {
        "type": "lora",
        "path": "models/blip_caplift_v3_lora_epoch_1"
    },
    "lora_v4_epoch_4": {
        "type": "lora",
        "path": "models/blip_caplift_v4_lora_epoch_4"
    }
}


# =========================
# LOAD MODEL
# =========================
def load_model(cfg):
    if cfg["type"] == "base":
        model = BlipForConditionalGeneration.from_pretrained(cfg["path"])

    elif cfg["type"] == "full":
        model = BlipForConditionalGeneration.from_pretrained(
            cfg["path"], local_files_only=True
        )

    elif cfg["type"] == "lora":
        base = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )
        model = PeftModel.from_pretrained(
            base, cfg["path"], local_files_only=True
        )

    model.to(DEVICE).eval()

    processor = BlipProcessor.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )

    return model, processor


# =========================
# LOAD CLIP
# =========================
def load_clip():
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32",
        pretrained="openai"
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    return model.to(DEVICE).eval(), preprocess, tokenizer


# =========================
# CAPTION
# =========================
def generate_caption(model, processor, image):
    inputs = processor(images=image, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=50,
            num_beams=5,
            length_penalty=1.2
        )

    return processor.decode(out[0], skip_special_tokens=True)


# =========================
# CLIP SCORE
# =========================
def clip_score(clip_model, preprocess, tokenizer, image, text):
    image = preprocess(image).unsqueeze(0).to(DEVICE)
    text_tokens = tokenizer([text]).to(DEVICE)

    with torch.no_grad():
        img_feat = clip_model.encode_image(image)
        txt_feat = clip_model.encode_text(text_tokens)

        img_feat /= img_feat.norm(dim=-1, keepdim=True)
        txt_feat /= txt_feat.norm(dim=-1, keepdim=True)

    return float((img_feat @ txt_feat.T).item()), img_feat, txt_feat


# =========================
# BLIP SCORE
# =========================
def blip_score(model, processor, image, text):
    inputs = processor(images=image, text=text, return_tensors="pt").to(DEVICE)
    labels = inputs["input_ids"]

    with torch.no_grad():
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss.item()

    return -loss


# =========================
# EVALUATE ONE MODEL
# =========================
def evaluate_model(name, cfg, clip_model, preprocess, tokenizer):
    print(f"\n=== Evaluating: {name} ===")

    model, processor = load_model(cfg)

    image_files = [
        f for f in sorted(os.listdir(IMAGE_DIR))
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    clip_scores = []
    blip_scores = []
    lengths = []

    image_features = []
    text_features = []

    for img_name in image_files:
        image = Image.open(os.path.join(IMAGE_DIR, img_name)).convert("RGB")

        caption = generate_caption(model, processor, image)

        cs, img_feat, txt_feat = clip_score(
            clip_model, preprocess, tokenizer, image, caption
        )
        bs = blip_score(model, processor, image, caption)

        clip_scores.append(cs)
        blip_scores.append(bs)
        lengths.append(len(caption.split()))

        image_features.append(img_feat)
        text_features.append(txt_feat)

    # Retrieval
    image_matrix = torch.cat(image_features, dim=0)
    text_matrix = torch.cat(text_features, dim=0)
    sim = image_matrix @ text_matrix.T

    correct = sum(torch.argmax(sim[i]).item() == i for i in range(len(image_files)))
    recall_at_1 = correct / len(image_files)

    return {
        "CLIPScore": float(np.mean(clip_scores)),
        "BLIPScore": float(np.mean(blip_scores)),
        "Recall@1": recall_at_1,
        "Length": float(np.mean(lengths))
    }


# =========================
# MAIN
# =========================
def main():
    clip_model, preprocess, tokenizer = load_clip()

    all_results = {}

    for name, cfg in MODELS.items():
        results = evaluate_model(name, cfg, clip_model, preprocess, tokenizer)
        all_results[name] = results

    print("\n=== FINAL COMPARISON ===")
    print(json.dumps(all_results, indent=2))


if __name__ == "__main__":
    main()
