from typing import Dict
from . import taxonomy


class DescriptionBuilder:
    """Собирает русское архивное описание из подписи и метаданных.

    template_mode:
      - "full"         : вводная фраза + "На изображении: <caption>." + опц. тема/настроение
      - "minimal"      : "На изображении: <caption>."
      - "caption_only" : только очищенная caption_ru

    Русские строки (именительный/родительный) берутся из taxonomy.py по id метки.
    """

    VALID_MODES = {"full", "minimal", "caption_only"}

    def __init__(
        self,
        *,
        template_mode: str = "full",
        include_theme: bool = True,
        include_mood: bool = True,
        drop_generic_style: bool = False,
    ):
        if template_mode not in self.VALID_MODES:
            raise ValueError(
                f"template_mode must be one of {self.VALID_MODES}, got {template_mode!r}"
            )
        self.template_mode = template_mode
        self.include_theme = include_theme
        self.include_mood = include_mood
        self.drop_generic_style = drop_generic_style

    def build(self, result: Dict) -> Dict:
        caption_ru = self._normalize_caption(result["caption"]["ru"])
        metadata = result["metadata"]
        inference = result["inference"]
        ocr = result.get("ocr")

        if self.template_mode == "caption_only":
            archive = self._sentence(caption_ru, capitalize=True)
            search_text = self._search_text(metadata, inference, caption_ru, ocr)
            tags_ru = self._tags_ru(metadata, inference)
            return {"archive_description": archive, "search_text": search_text, "tags_ru": tags_ru}

        if self.template_mode == "minimal":
            archive = f"На изображении: {self._inline(caption_ru)}."
            search_text = self._search_text(metadata, inference, caption_ru, ocr)
            tags_ru = self._tags_ru(metadata, inference)
            return {"archive_description": archive, "search_text": search_text, "tags_ru": tags_ru}

        image_type_field = metadata.get("image_type", {})
        style_field = metadata.get("style", {})

        theme = inference.get("theme")
        mood = inference.get("mood")

        # Русские строки из единого источника — по taxonomy_version из метаданных.
        version = metadata.get("taxonomy_version", taxonomy.DEFAULT_VERSION)
        image_type_map = taxonomy.ru_map(version, "image_type")
        style_map_gen = taxonomy.ru_gen_map(version, "style")
        theme_map = taxonomy.ru_map(version, "theme")
        mood_map = taxonomy.ru_map(version, "mood")

        image_type_label = image_type_field.get("label")
        style_label = style_field.get("label")

        image_type_ru = image_type_map.get(image_type_label)
        style_ru_gen = style_map_gen.get(style_label)
        theme_ru = theme_map.get(theme, theme) if theme else None
        mood_ru = mood_map.get(mood, mood) if mood else None

        parts = []

        intro = self._intro_phrase(
            image_type_field=image_type_field,
            style_field=style_field,
            image_type_ru=image_type_ru,
            style_ru_gen=style_ru_gen,
            image_type_label=image_type_label,
            style_label=style_label,
            generic_styles=taxonomy.generic_styles(version),
        )
        if intro:
            parts.append(intro)

        parts.append(f"На изображении: {self._inline(caption_ru)}.")

        if self.include_theme and theme_ru:
            parts.append(f"Тематика изображения: {theme_ru}.")
        if self.include_mood and mood_ru:
            parts.append(f"Общее настроение изображения можно охарактеризовать как {mood_ru}.")

        archive_description = " ".join(parts)
        search_text = self._search_text(metadata, inference, caption_ru, ocr)
        tags_ru = self._tags_ru(metadata, inference)

        return {
            "archive_description": archive_description,
            "search_text": search_text,
            "tags_ru": tags_ru,
        }

    def _intro_phrase(
        self,
        *,
        image_type_field: Dict,
        style_field: Dict,
        image_type_ru: str | None,
        style_ru_gen: str | None,
        image_type_label: str | None,
        style_label: str | None,
        generic_styles: set,
    ) -> str:
        image_conf = image_type_field.get("confident", False)
        style_conf = style_field.get("confident", False)

        # Полностью убираем generic-стиль из вводной фразы: слова
        # «vintage / decorative / retro» не различают открытки и дают
        # повторяющиеся неинформативные зачины. Тип материала остаётся;
        # стиль в любом случае сохраняется в search_text.
        if self.drop_generic_style and style_label in generic_styles:
            style_conf = False
            style_ru_gen = None

        # отдельные безопасные случаи для фотографий
        if image_conf and image_type_label == "a photograph":
            if style_label == "black and white photo":
                return "Черно-белая фотография."
            if style_label == "color photograph":
                return "Цветная фотография."
            return "Фотография."

        if image_conf and image_type_ru and style_conf and style_ru_gen:
            return f"{image_type_ru.capitalize()} в стиле {style_ru_gen}."

        if image_conf and image_type_ru:
            return f"{image_type_ru.capitalize()}."

        if style_conf and style_ru_gen:
            return f"Изображение в стиле {style_ru_gen}."

        return "Изображение."

    @staticmethod
    def _normalize_caption(text: str) -> str:
        text = (text or "").strip()
        text = text.rstrip(" .,:;!-")
        return text or "изображение"

    @staticmethod
    def _inline(text: str) -> str:
        text = text.strip()
        if not text:
            return "изображение"
        return text[0].lower() + text[1:] if len(text) > 1 else text.lower()

    @staticmethod
    def _sentence(text: str, capitalize: bool = True) -> str:
        text = (text or "").strip().rstrip(" .,:;!-")
        if not text:
            text = "изображение"
        if capitalize:
            text = text[0].upper() + text[1:] if len(text) > 1 else text.upper()
        return f"{text}."

    @classmethod
    def _tags_ru(cls, metadata: Dict, inference: Dict) -> list[str]:
        """Возвращает русские теги по уверенным предсказаниям классификатора.

        В отличие от сырого metadata.tags (англоязычные исходные метки SigLIP),
        этот список содержит локализованные термины из каталожной таксономии
        и пригоден для отображения в UI без дополнительной обработки.
        """
        version = metadata.get("taxonomy_version", taxonomy.DEFAULT_VERSION)
        image_type_map = taxonomy.ru_map(version, "image_type")
        style_map = taxonomy.ru_map(version, "style")
        theme_map = taxonomy.ru_map(version, "theme")
        mood_map = taxonomy.ru_map(version, "mood")

        image_type_field = metadata.get("image_type", {})
        style_field = metadata.get("style", {})

        out: list[str] = []
        if image_type_field.get("confident"):
            ru = image_type_map.get(image_type_field.get("label"))
            if ru:
                out.append(ru)
        if style_field.get("confident"):
            ru = style_map.get(style_field.get("label"))
            if ru:
                out.append(ru)
        theme = inference.get("theme")
        if theme:
            out.append(theme_map.get(theme, theme))
        mood = inference.get("mood")
        if mood:
            out.append(mood_map.get(mood, mood))
        return out

    @classmethod
    def _search_text(
        cls,
        metadata: Dict,
        inference: Dict,
        caption_ru: str | None = None,
        ocr: Dict | None = None,
    ) -> str:
        image_type_field = metadata.get("image_type", {})
        style_field = metadata.get("style", {})
        tags = metadata.get("tags", [])
        version = metadata.get("taxonomy_version", taxonomy.DEFAULT_VERSION)

        image_type_map = taxonomy.ru_map(version, "image_type")
        style_map = taxonomy.ru_map(version, "style")
        theme_map = taxonomy.ru_map(version, "theme")
        mood_map = taxonomy.ru_map(version, "mood")

        search_terms = []

        image_type_ru = image_type_map.get(image_type_field.get("label"))
        style_ru = style_map.get(style_field.get("label"))
        theme_ru = theme_map.get(inference.get("theme"), inference.get("theme")) if inference.get("theme") else None
        mood_ru = mood_map.get(inference.get("mood"), inference.get("mood")) if inference.get("mood") else None

        if image_type_field.get("confident") and image_type_ru:
            search_terms.append(image_type_ru)
        if style_field.get("confident") and style_ru:
            search_terms.append(style_ru)
        if theme_ru:
            search_terms.append(theme_ru)
        if mood_ru:
            search_terms.append(mood_ru)

        # NB: metadata["tags"] хранит сырые англоязычные метки SigLIP
        # ("a postcard", "a chromolithograph", ...). Они УЖЕ покрыты
        # русскими переводами выше через image_type_ru / style_ru / theme_ru.
        # Повторно дублировать их в search_text вредно — это вносит шум
        # в поисковую строку и ухудшает токенизацию ("a postcard" → "a", "postcard").

        if caption_ru:
            caption_norm = caption_ru.strip().rstrip(" .,:;!-")
            if caption_norm:
                search_terms.append(caption_norm.lower())

        # Надпись OCR - только уверенная (гейт по confidence + санити-фильтр),
        # иначе в поисковую строку попал бы шум распознавания.
        if ocr and ocr.get("confident") and ocr.get("text"):
            search_terms.append(ocr["text"].strip().lower())

        return " ".join(search_terms)
