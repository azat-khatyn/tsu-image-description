from typing import Dict


class DescriptionBuilder:
    def build(self, result: Dict) -> Dict:
        caption_ru = result["caption"]["ru"]
        metadata = result["metadata"]
        inference = result["inference"]

        image_type = metadata.get("image_type", "изображение")
        style = metadata.get("style", "иллюстративный стиль")
        tags = metadata.get("tags", [])
        theme = inference.get("theme", "общая иллюстративная сцена")
        mood = inference.get("mood", "нейтральное")

        image_type_ru_map = {
            "a postcard": "открытка",
            "a poster": "плакат",
            "a photograph": "фотография",
            "an illustration": "иллюстрация",
        }

        style_ru_map = {
            "vintage illustration": "винтажная иллюстрация",
            "modern design": "современный дизайн",
            "black and white photo": "черно-белая фотография",
            "color photograph": "цветная фотография",
            "painting": "живопись",
            "drawing": "рисунок",
        }

        theme_ru_map = {
            "holiday scene": "праздничная сцена",
            "romantic scene": "романтическая сцена",
            "children scene": "детская сцена",
            "urban scene": "городская сцена",
            "nature scene": "природная сцена",
            "religious scene": "религиозная сцена",
        }

        mood_ru_map = {
            "happy": "радостное",
            "festive": "праздничное",
            "romantic": "романтическое",
            "nostalgic": "ностальгическое",
            "serious": "серьёзное",
        }

        image_type_ru = image_type_ru_map.get(image_type, image_type)
        style_ru = style_ru_map.get(style, style)
        theme_ru = theme_ru_map.get(theme, theme)
        mood_ru = mood_ru_map.get(mood, mood)

        archive_description = (
            f"{image_type_ru.capitalize()} в стиле {style_ru}. "
            f"На изображении: {caption_ru}. "
            f"Предполагаемая тема: {theme_ru}. "
            f"Общее настроение: {mood_ru}."
        )

        search_terms = [image_type_ru, style_ru, theme_ru, mood_ru]
        for tag in tags:
            search_terms.append(str(tag))

        search_text = " ".join(search_terms)

        return {
            "archive_description": archive_description,
            "search_text": search_text
        }
