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
            "Easter scene",
            "New Year scene",
        ]

        self.moods = [
            "happy",
            "festive",
            "romantic",
            "nostalgic",
            "serious",
        ]

        self.thresholds = {
            "image_type": 0.35,
            "style": 0.30,
            "theme": 0.28,
            "mood": 0.28,
        }

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

    def _top_k(self, scores: Dict[str, float], k: int = 3) -> List[Dict[str, float]]:
        ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [
            {"label": label, "score": round(float(score), 4)}
            for label, score in ordered[:k]
        ]

    def _select_label(self, scores: Dict[str, float], threshold: float) -> Dict[str, object]:
        top_items = self._top_k(scores, k=3)
        best = top_items[0]

        if best["score"] < threshold:
            return {
                "label": None,
                "score": best["score"],
                "accepted": False,
                "top_k": top_items,
            }

        return {
            "label": best["label"],
            "score": best["score"],
            "accepted": True,
            "top_k": top_items,
        }

    def extract(self, image_path: str) -> Dict:
        image = Image.open(image_path).convert("RGB")

        image_type_scores = self._classify_with_scores(image, self.image_types)
        style_scores = self._classify_with_scores(image, self.styles)
        theme_scores = self._classify_with_scores(image, self.themes)
        mood_scores = self._classify_with_scores(image, self.moods)

        image_type = self._select_label(image_type_scores, self.thresholds["image_type"])
        style = self._select_label(style_scores, self.thresholds["style"])
        theme = self._select_label(theme_scores, self.thresholds["theme"])
        mood = self._select_label(mood_scores, self.thresholds["mood"])

        tags = []
        for item in [image_type, style, theme, mood]:
            if item["label"] is not None:
                tags.append(item["label"])

        return {
            "image_type": image_type,
            "style": style,
            "theme": theme,
            "mood": mood,
            "tags": tags,
        }
