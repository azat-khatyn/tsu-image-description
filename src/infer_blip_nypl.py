# file: src/infer_blip_nypl.py

from __future__ import annotations

import argparse

import torch
from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = BlipProcessor.from_pretrained(args.model_dir)
    model = BlipForConditionalGeneration.from_pretrained(args.model_dir).to(device)
    model.eval()

    image = Image.open(args.image).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        generated_ids = model.generate(
            pixel_values=inputs["pixel_values"],
            max_new_tokens=64,
            num_beams=4,
        )

    caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(caption)


if __name__ == "__main__":
    main()
