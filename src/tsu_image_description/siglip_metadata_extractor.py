"""SigLIP zero-shot классификатор по таксономии.

Поддерживает несколько версий таксономии (для воспроизводимости старых
экспериментов). По умолчанию — "archival_v2": каталожная таксономия,
согласованная с MARC 21 п.655, Getty AAT (Print processes / Subjects)
и российской филокартической традицией (Файнштейн Э.Б. «В мире открытки», 1976).

Для воспроизводимости старых прогонов передавайте taxonomy_version="legacy_v1"
(вернёт разговорные метки стиля и поле mood).
"""

from typing import Dict, List
from PIL import Image
import torch
from transformers import AutoModel, AutoProcessor

from .models import get_device


TAXONOMIES = {
    # ====================================================================
    # legacy_v1 — первая итерация таксономии.
    # Разговорные метки стиля (vintage illustration, decorative illustration,
    # retro design) и субъективные mood-категории. Сохранена только для
    # воспроизводимости старых прогонов.
    # ====================================================================
    "legacy_v1": {
        "image_types": [
            "a postcard",
            "a poster",
            "a greeting card",
            "an illustration",
            "a photograph",
        ],
        "styles": [
            "vintage illustration",
            "retro design",
            "decorative illustration",
            "engraving",
            "drawing",
            "painting",
            "black and white photo",
            "color photograph",
        ],
        "themes": [
            "holiday scene",
            "Easter holiday scene",
            "Christmas holiday scene",
            "New Year celebration",
            "romantic scene",
            "children scene",
            "urban scene",
            "nature scene",
            "religious scene",
        ],
        "moods": [
            "happy", "festive", "romantic",
            "nostalgic", "calm", "serious",
        ],
    },

    # ====================================================================
    # archival_v2 — каталожная таксономия (основная).
    #
    # Источники: Файнштейн Э.Б. «В мире открытки» (1976) — типология открыток;
    # MARC 21 поле 655; Getty AAT (Print processes, Subjects); ГОСТ 7.69-2009.
    #
    # Отличия от legacy_v1: разговорные метки стиля заменены на конкретные
    # техники печати; темы переименованы на каталожный язык; поле mood удалено
    # (субъективные суждения не идут в архивную опись).
    # ====================================================================
    "archival_v2": {
        "image_types": [
            "a postcard",          # открытка
            "a greeting card",     # поздравительная карточка
            "an illustration",     # иллюстрация
            "a photograph",        # фотография
            "a poster",            # плакат
        ],
        "styles": [
            # Print processes (Getty AAT)
            "a chromolithograph",      # хромолитография
            "an engraving",            # гравюра
            "an etching",              # офорт
            # Painting / drawing media
            "a watercolor painting",   # акварель
            "an oil painting",         # масляная живопись
            "a pencil drawing",        # карандашный рисунок
            # Photography
            "a black and white photograph",  # чёрно-белая фотография
            "a color photograph",            # цветная фотография
        ],
        "themes": [
            # Traditional art subject categories
            "a landscape",           # пейзаж
            "an urban view",         # городской вид
            "a portrait",            # портрет
            "a genre scene",         # жанровая сцена
            "a still life",          # натюрморт
            "a religious subject",   # религиозный сюжет
            "a military subject",    # военный сюжет
            "a holiday scene",       # праздничная сцена (Рождество, Пасха, и пр.)
        ],
        # mood удалён — субъективные суждения не каталожны.
        "moods": [],
    },
}


class SigLIPMetadataExtractor:
    """SigLIP zero-shot классификатор с версионируемой таксономией.

    Args:
        model_name: идентификатор модели HF.
        taxonomy_version: одно из {"legacy_v1", "archival_v2"}. По умолчанию "archival_v2".
    """

    def __init__(
        self,
        model_name: str = "google/siglip-base-patch16-224",
        *,
        taxonomy_version: str = "archival_v2",
    ):
        if taxonomy_version not in TAXONOMIES:
            raise ValueError(
                f"Unknown taxonomy_version={taxonomy_version!r}. "
                f"Available: {sorted(TAXONOMIES.keys())}"
            )
        self.taxonomy_version = taxonomy_version
        self.device = get_device()
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)

        taxonomy = TAXONOMIES[taxonomy_version]
        self.image_types = taxonomy["image_types"]
        self.styles = taxonomy["styles"]
        self.themes = taxonomy["themes"]
        self.moods = taxonomy["moods"]

        self.thresholds = {
            "image_type": 0.35,
            "style": 0.22,
            "theme": 0.18,
            "mood": 0.18,
        }

        self.margins = {
            "image_type": 0.05,
            "style": 0.03,
            "theme": 0.03,
            "mood": 0.03,
        }

        print(
            f"[SigLIPMetadataExtractor] taxonomy_version={taxonomy_version} "
            f"({len(self.image_types)}/{len(self.styles)}/"
            f"{len(self.themes)}/{len(self.moods)} types/styles/themes/moods)"
        )

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

    def _pack_field(self, scores: Dict[str, float], threshold: float, margin: float, k: int = 3) -> Dict:
        top = self._top_k(scores, k=k)
        best = top[0]
        second_score = top[1]["score"] if len(top) > 1 else 0.0

        confident = (best["score"] >= threshold) and ((best["score"] - second_score) >= margin)

        return {
            "label": best["label"],
            "score": best["score"],
            "margin": round(best["score"] - second_score, 4),
            "confident": confident,
            "alternatives": top[1:],
        }

    def _empty_field(self) -> Dict:
        """Заглушка для таксономий без меток (например, mood в archival_v2)."""
        return {
            "label": None,
            "score": 0.0,
            "margin": 0.0,
            "confident": False,
            "alternatives": [],
        }

    def extract(self, image_path: str) -> Dict:
        image = Image.open(image_path).convert("RGB")

        image_type_scores = self._classify_with_scores(image, self.image_types)
        style_scores = self._classify_with_scores(image, self.styles)
        theme_scores = self._classify_with_scores(image, self.themes)

        image_type = self._pack_field(
            image_type_scores,
            self.thresholds["image_type"],
            self.margins["image_type"],
        )
        style = self._pack_field(
            style_scores,
            self.thresholds["style"],
            self.margins["style"],
        )
        theme = self._pack_field(
            theme_scores,
            self.thresholds["theme"],
            self.margins["theme"],
        )

        if self.moods:
            mood_scores = self._classify_with_scores(image, self.moods)
            mood = self._pack_field(
                mood_scores,
                self.thresholds["mood"],
                self.margins["mood"],
            )
        else:
            # archival_v2: mood удалён, пустое поле
            mood = self._empty_field()

        tags = []
        for field in [image_type, style, theme, mood]:
            if field.get("confident") and field.get("label"):
                tags.append(field["label"])

        return {
            "image_type": image_type,
            "style": style,
            "theme": theme,
            "mood": mood,
            "tags": tags,
            "taxonomy_version": self.taxonomy_version,
        }
