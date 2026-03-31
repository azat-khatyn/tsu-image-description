from typing import Dict


class DescriptionBuilder:
    def build(self, result: Dict) -> Dict:
        caption_ru = result["caption"]["ru"]
        metadata = result["metadata"]
        inference = result["inference"]

        image_type_field = metadata.get("image_type", {})
        style_field = metadata.get("style", {})
        tags = metadata.get("tags", [])

        theme = inference.get("theme")
        mood = inference.get("mood")

        image_type_map = {
            "a postcard": "открытка",
            "a poster": "плакат",
            "a photograph": "фотография",
            "an illustration": "иллюстрация",
        }

        style_map = {
            "vintage illustration": "винтажная иллюстрация",
            "modern design": "современный дизайн",
            "black and white photo": "черно-белая фотография",
            "color photograph": "цветная фотография",
            "painting": "живопись",
            "drawing": "рисунок",
        }

        theme_map = {
            "holiday scene": "праздничная сцена",
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
            if style_ru:
                parts.append(f"{image_type_ru.capitalize()} в стиле {style_ru}.")
            else:
                parts.append(f"{image_type_ru.capitalize()}.")
        else:
            parts.append("Иллюстративное изображение.")

        parts.append(f"На изображении: {caption_ru}.")

        if theme_ru:
            parts.append(f"Предположительно, это {theme_ru}.")
        if mood_ru:
            parts.append(f"Общее настроение изображения можно охарактеризовать как {mood_ru}.")

        archive_description = " ".join(parts)

        search_terms = []

        if image_type_field.get("confident") and image_type_ru:
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

        search_text = " ".join(search_terms)

        return {
            "archive_description": archive_description,
            "search_text": search_text
        }
g