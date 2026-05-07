import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import os

DEVICE = (
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

model = BlipForConditionalGeneration.from_pretrained("models/blip_nypl").to(DEVICE)
processor = BlipProcessor.from_pretrained("models/blip_nypl")

def generate_caption(image_path):
    image = Image.open(image_path).convert("RGB")

    inputs = processor(images=image, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=50)

    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption


TEST_DIR = "data/eval/images"

for img in os.listdir(TEST_DIR):
    path = os.path.join(TEST_DIR, img)

    caption = generate_caption(path)

    print(img)
    print("→", caption)
    print()
