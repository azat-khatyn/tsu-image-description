from typing import Dict


class DescriptionBuilder:
    """Builds the Russian archive description from caption + metadata.

    template_mode:
      - "full"         : <type> в стиле <style>. На изображении: <caption>. + theme + mood
                          (theme/mood are gated by include_theme/include_mood)
      - "minimal"      : "На изображении: <caption>."
      - "caption_only" : raw caption_ru only — no template wrapping at all
    """

    VALID_MODES = {"full", "minimal", "caption_only"}

    def __init__(
        self,
        *,
        template_mode: str = "full",
        include_theme: bool = True,
        include_mood: bool = True,
    ):
        if template_mode not in self.VALID_MODES:
            raise ValueError(
                f"template_mode must be one of {self.VALID_MODES}, got {template_mode!r}"
            )
        self.template_mode = template_mode
        self.include_theme = include_theme
        self.include_mood = include_mood

    def build(self, result: Dict) -> Dict:
        caption_ru = result["caption"]["ru"]
        metadata = result["metadata"]
        inference = result["inference"]

        # === template_mode shortcuts ===
        if self.template_mode == "caption_only":
            search_text = self._search_text(metadata, inference)
            return {"archive_description": caption_ru, "search_text": search_text}

        if self.template_mode == "minimal":
            archive = f"На изображении: {caption_ru}."
            search_text = self._search_text(metadata, inference)
            return {"archive_description": archive, "search_text": search_text}

        # === Full template (default) ===

        image_type_field = metadata.get("image_type", {})
        style_field = metadata.get("style", {})
        tags = metadata.get("tags", [])

        theme = inference.get("theme")
        mood = inference.get("mood")

        image_type_map = {
            "a postcard": "открытка",
            "a poster": "плакат",
            "a greeting card": "поздравительная открытка",
            "an illustration": "иллюстрация",
            "a photograph": "фотография",
        }

        style_map = {
            "vintage illustration": "винтажная иллюстрация",
            "retro design": "ретро-дизайн",
            "decorative illustration": "декоративная иллюстрация",
            "engraving": "гравюра",
            "drawing": "рисунок",
            "painting": "живопись",
            "black and white photo": "черно-белая фотография",
            "color photograph": "цветная фотография",
        }

        theme_map = {
            "holiday scene": "праздничная сцена",
            "Easter holiday scene": "пасхальная сцена",
            "Christmas holiday scene": "рождественская сцена",
            "New Year celebration": "новогодняя сцена",
            "romantic scene": "романтическая сцена",
            "children scene": "детская сцена",
            "urban scene": "городская сцена",
            "nature scene": "сцена природы",
            "religious scene": "религиозная сцена",
        }

        mood_map = {
            "happy": "радостное",
            "festive": "праздничное",
            "romantic": "романтическое",
            "nostalgic": "ностальгическое",
            "calm": "спокойное",
            "serious": "серьёзное",
        }

        image_type_label = image_type_field.get("label")
        style_label = style_field.get("label")

        image_type_ru = image_type_map.get(image_type_label, "изображение")
        style_ru = style_map.get(style_label, style_label) if style_label else None
        theme_ru = theme_map.get(theme, theme) if theme else None
        mood_ru = mood_map.get(mood, mood) if mood else None

        parts = []

        if image_type_field.get("confident"):
            if style_field.get("confident") and style_ru:
                parts.append(f"{image_type_ru.capitalize()} в стиле {style_ru}.")
            else:
                parts.append(f"{image_type_ru.capitalize()}.")
        else:
            parts.append("Иллюстративное изображение.")

        parts.append(f"На изображении: {caption_ru}.")

        if self.include_theme and theme_ru:
            parts.append(f"Предположительно, это {theme_ru}.")
        if self.include_mood and mood_ru:
            parts.append(f"Общее настроение изображения можно охарактеризовать как {mood_ru}.")

        archive_description = " ".join(parts)
        search_text = self._search_text(metadata, inference)

        return {
            "archive_description": archive_description,
            "search_text": search_text,
        }

    @staticmethod
    def _search_text(metadata: Dict, inference: Dict) -> str:
        """Поисковый поток тегов — собирается одинаково для всех template_mode."""
        image_type_field = metadata.get("image_type", {})
        style_field = metadata.get("style", {})
        tags = metadata.get("tags", [])

        image_type_map = {
            "a postcard": "открытка",
            "a poster": "плакат",
            "a greeting card": "поздравительная открытка",
            "an illustration": "иллюстрация",
            "a photograph": "фотография",
        }
        style_map = {
            "vintage illustration": "винтажная иллюстрация",
            "retro design": "ретро-дизайн",
            "decorative illustration": "декоративная иллюстрация",
            "engraving": "гравюра",
            "drawing": "рисунок",
            "painting": "живопись",
            "black and white photo": "черно-белая фотография",
            "color photograph": "цветная фотография",
        }
        theme_map = {
            "holiday scene": "праздничная сцена",
            "Easter holiday scene": "пасхальная сцена",
            "Christmas holiday scene": "рождественская сцена",
            "New Year celebration": "новогодняя сцена",
            "romantic scene": "романтическая сцена",
            "children scene": "детская сцена",
            "urban scene": "городская сцена",
            "nature scene": "сцена природы",
            "religious scene": "религиозная сцена",
        }
        mood_map = {
            "happy": "радостное",
            "festive": "праздничное",
            "romantic": "романтическое",
            "nostalgic": "ностальгическое",
            "calm": "спокойное",
            "serious": "серьёзное",
        }

        image_type_ru = image_type_map.get(image_type_field.get("label"), "изображение")
        style_label = style_field.get("label")
        style_ru = style_map.get(style_label, style_label) if style_label else None
        theme_ru = theme_map.get(inference.get("theme"), inference.get("theme")) if inference.get("theme") else None
        mood_ru = mood_map.get(inference.get("mood"), inference.get("mood")) if inference.get("mood") else None

        search_terms = []
        if image_type_field.get("confident"):
            search_terms.append(image_type_ru)
        if style_field.get("confident") and style_ru:
            search_terms.append(style_ru)
        if theme_ru:
            search_terms.append(theme_ru)
        if mood_ru:
            search_terms.append(mood_ru)
        for tag in tags:
            if tag not in search_terms:
                search_terms.append(tag)
        return " ".join(search_terms)
