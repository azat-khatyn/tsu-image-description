from typing import Dict, List
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel


class CLIPMetadataExtractor:
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)

        self.image_types = [
            "a postcard",
            "a poster",
            "a photograph",
            "an illustration",
        ]

        self.styles = [
            "vintage illustration",
            "modern design",
            "black and white photo",
            "color photograph",
            "painting",
            "drawing",
        ]

        self.themes = [
            "holiday scene",
            "romantic scene",
            "children scene",
            "urban scene",
            "nature scene",
            "religious scene",
        ]

        self.moods = [
            "happy",
            "festive",
            "romantic",
            "nostalgic",
            "serious",
        ]

    def _classify_with_scores(self, image: Image.Image, candidates: List[str]) -> Dict[str, float]:
        inputs = self.processor(
            text=candidates,
            images=image,
            return_tensors="pt",
            padding=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits_per_image
            probs = logits.softmax(dim=1).squeeze(0).tolist()

        return {candidate: float(score) for candidate, score in zip(candidates, probs)}

    def _best_label(self, scores: Dict[str, float]) -> str:
        return max(scores, key=scores.get)

    def extract(self, image_path: str) -> Dict:
        image = Image.open(image_path).convert("RGB")

        image_type_scores = self._classify_with_scores(image, self.image_types)
        style_scores = self._classify_with_scores(image, self.styles)
        theme_scores = self._classify_with_scores(image, self.themes)
        mood_scores = self._classify_with_scores(image, self.moods)

        image_type = self._best_label(image_type_scores)
        style = self._best_label(style_scores)
        theme = self._best_label(theme_scores)
        mood = self._best_label(mood_scores)

        return {
            "image_type": image_type,
            "style": style,
            "theme": theme,
            "mood": mood,
            "tags": [image_type, style, theme, mood],
            "scores": {
                "image_type": image_type_scores,
                "style": style_scores,
                "theme": theme_scores,
                "mood": mood_scores,
            }
        }
