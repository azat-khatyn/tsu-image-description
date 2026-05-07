import json
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
).to(DEVICE)

INPUT = "data/nypl/nypl_train_ready.jsonl"
OUTPUT = "data/nypl/capfilt_raw.jsonl"


def generate_caption(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=50
        )

    return processor.decode(out[0], skip_special_tokens=True)


with open(INPUT, "r") as f_in, open(OUTPUT, "w") as f_out:
    for line in f_in:
        item = json.loads(line)

        image_path = item["image"]

        try:
            caption = generate_caption(image_path)
        except Exception as e:
            continue

        new_item = {
            "image": image_path,
            "caption": caption,
            "title": item.get("text", "")
        }

        f_out.write(json.dumps(new_item) + "\n")
