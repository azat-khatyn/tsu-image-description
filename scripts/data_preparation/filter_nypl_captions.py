import json
import torch
from PIL import Image
import open_clip

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32",
    pretrained="openai"
)
model = model.to(DEVICE).eval()

tokenizer = open_clip.get_tokenizer("ViT-B-32")

INPUT = "data/nypl/splits/capfilt_raw.jsonl"
OUTPUT = "data/nypl/splits/capfilt_filtered_v2.jsonl"

THRESHOLD = 0.3  # можно варьировать


def clip_score(image_path, text):
    image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(DEVICE)
    text_tokens = tokenizer([text]).to(DEVICE)

    with torch.no_grad():
        img_feat = model.encode_image(image)
        txt_feat = model.encode_text(text_tokens)

        img_feat /= img_feat.norm(dim=-1, keepdim=True)
        txt_feat /= txt_feat.norm(dim=-1, keepdim=True)

    return float((img_feat @ txt_feat.T).item())


kept = 0
total = 0

with open(INPUT) as f_in, open(OUTPUT, "w") as f_out:
    for line in f_in:
        item = json.loads(line)
        total += 1

        caption = item.get("caption", "").strip()
        title = item.get("title", "").strip()

        # 1. фильтр по длине
        if len(caption.split()) < 4:
            continue

        try:
            score = clip_score(item["image"], caption)
        except:
            continue

        if score >= THRESHOLD:
            kept += 1

            # 2. гибридная подпись: caption и title
            if title:
                final_caption = f"{caption}. {title}"
            else:
                final_caption = caption

            out_item = {
                "image": item["image"],
                "caption": final_caption
            }

            f_out.write(json.dumps(out_item) + "\n")

print(f"Kept {kept}/{total}")
