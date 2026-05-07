from PIL import Image
import torch
from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    MarianMTModel,
    MarianTokenizer,
)


def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# class CaptionGenerator:
#     def __init__(self, model_name: str = "Salesforce/blip-image-captioning-base"):
#         self.device = get_device()
#         self.processor = BlipProcessor.from_pretrained(model_name)
#         self.model = BlipForConditionalGeneration.from_pretrained(model_name).to(self.device)
#
#     def generate(self, image_path: str) -> str:
#         image = Image.open(image_path).convert("RGB")
#         inputs = self.processor(images=image, return_tensors="pt").to(self.device)
#         output = self.model.generate(**inputs, max_new_tokens=60)
#         caption = self.processor.decode(output[0], skip_special_tokens=True)
#         return caption.strip()

class CaptionGenerator:
    def __init__(self, model_path=None):
        self.model_path = model_path or "Salesforce/blip-image-captioning-base"

        self.device = (
            "mps" if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available()
            else "cpu"
        )

        print(f"[CaptionGenerator] Using model: {self.model_path}")

        self.processor = BlipProcessor.from_pretrained(self.model_path)
        self.model = BlipForConditionalGeneration.from_pretrained(self.model_path).to(self.device)

    def generate(self, image_path: str) -> str:
        image = Image.open(image_path).convert("RGB")

        inputs = self.processor(images=image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=50
            )

        caption = self.processor.decode(out[0], skip_special_tokens=True)
        return caption


class Translator:
    def __init__(self, model_name: str = "Helsinki-NLP/opus-mt-en-ru"):
        self.device = get_device()
        self.tokenizer = MarianTokenizer.from_pretrained(model_name)
        self.model = MarianMTModel.from_pretrained(model_name).to(self.device)

    def translate(self, text: str) -> str:
        inputs = self.tokenizer(text, return_tensors="pt", padding=True).to(self.device)
        translated = self.model.generate(**inputs)
        return self.tokenizer.decode(translated[0], skip_special_tokens=True)
