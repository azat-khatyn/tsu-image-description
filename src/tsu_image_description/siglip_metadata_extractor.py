from typing import Dict, List
from PIL import Image
import torch
from transformers import AutoModel, AutoProcessor

from .models import get_device


class SigLIPMetadataExtractor:
    def __init__(self, model_name: str = "google/siglip-base-patch16-224"):
        self.device = get_device()
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)

        self.image_types = [
            "a postcard",
            "a poster",
            "a greeting card",
            "an illustration",
            "a photograph",
        ]

        self.styles = [
            "vintage illustration",
            "retro design",
            "decorative illustration",
            "engraving",
            "drawing",
            "painting",
            "black and white photo",
            "color photograph",
        ]

        self.themes = [
            "holiday scene",
            "Easter holiday scene",
            "Christmas holiday scene",
            "New Year celebration",
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
            "calm",
            "serious",
        ]

        self.thresholds = {
            "image_type": 0.35,
            "style": 0.22,
            "theme": 0.18,
            "mood": 0.18,
        }

    def _classify_with_scores(self, image: Image.Image, candidates: List[str]) -> Dict[str, float]:
        inputs = self.processor(
            text=candidates,
            images=image,
            padding="max_length",
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits_per_image
            probs = torch.softmax(logits, dim=1).squeeze(0).detach().cpu().tolist()

        return {candidate: float(score) for candidate, score in zip(candidates, probs)}

    def _top_k(self, scores: Dict[str, float], k: int = 3) -> List[Dict[str, float]]:
        ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [{"label": label, "score": round(score, 4)} for label, score in ordered[:k]]

    def _pack_field(self, scores: Dict[str, float], threshold: float, k: int = 3) -> Dict:
        top = self._top_k(scores, k=k)
        best = top[0]
        return {
            "label": best["label"],
            "score": best["score"],
            "confident": best["score"] >= threshold,
            "alternatives": top[1:],
        }

    def extract(self, image_path: str) -> Dict:
        image = Image.open(image_path).convert("RGB")

        image_type_scores = self._classify_with_scores(image, self.image_types)
        style_scores = self._classify_with_scores(image, self.styles)
        theme_scores = self._classify_with_scores(image, self.themes)
        mood_scores = self._classify_with_scores(image, self.moods)

        image_type = self._pack_field(image_type_scores, self.thresholds["image_type"])
        style = self._pack_field(style_scores, self.thresholds["style"])
        theme = self._pack_field(theme_scores, self.thresholds["theme"])
        mood = self._pack_field(mood_scores, self.thresholds["mood"])

        tags = []
        for field in [image_type, style, theme, mood]:
            if field["confident"]:
                tags.append(field["label"])

        return {
            "image_type": image_type,
            "style": style,
            "theme": theme,
            "mood": mood,
            "tags": tags,
        }
