from typing import Dict


class DescriptionBuilder:
    """Builds the Russian archive description from caption + metadata.

    template_mode:
      - "full"         : intro sentence + "На изображении: <caption>." + optional theme/mood
      - "minimal"      : "На изображении: <caption>."
      - "caption_only" : cleaned raw caption_ru only
    """

    VALID_MODES = {"full", "minimal", "caption_only"}

    # Style labels that don't carry archive-relevant discriminative info.
    # Each of them applies to roughly any decorated postcard, so the phrase
    # "Открытка в стиле винтажной иллюстрации" repeats across most items and
    # adds no semantic value. When `drop_generic_style=True`, intro phrase
    # is built without "в стиле X" for these.
    _GENERIC_STYLE_LABELS = {
        "vintage illustration",
        "decorative illustration",
        "retro design",
    }

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

        if self.template_mode == "caption_only":
            archive = self._sentence(caption_ru, capitalize=True)
            search_text = self._search_text(metadata, inference, caption_ru)
            tags_ru = self._tags_ru(metadata, inference)
            return {"archive_description": archive, "search_text": search_text, "tags_ru": tags_ru}

        if self.template_mode == "minimal":
            archive = f"На изображении: {self._inline(caption_ru)}."
            search_text = self._search_text(metadata, inference, caption_ru)
            tags_ru = self._tags_ru(metadata, inference)
            return {"archive_description": archive, "search_text": search_text, "tags_ru": tags_ru}

        image_type_field = metadata.get("image_type", {})
        style_field = metadata.get("style", {})

        theme = inference.get("theme")
        mood = inference.get("mood")

        # Versioned RU mappings — keyed by taxonomy_version stored in metadata.
        taxonomy_version = metadata.get("taxonomy_version", "legacy_v1")
        image_type_map = self._image_type_map_for(taxonomy_version)
        style_map_gen = self._style_map_gen_for(taxonomy_version)
        theme_map = self._theme_map_for(taxonomy_version)
        mood_map = self._mood_map_for(taxonomy_version)

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
        )
        if intro:
            parts.append(intro)

        parts.append(f"На изображении: {self._inline(caption_ru)}.")

        if self.include_theme and theme_ru:
            parts.append(f"Тематика изображения: {theme_ru}.")
        if self.include_mood and mood_ru:
            parts.append(f"Общее настроение изображения можно охарактеризовать как {mood_ru}.")

        archive_description = " ".join(parts)
        search_text = self._search_text(metadata, inference, caption_ru)
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
    ) -> str:
        image_conf = image_type_field.get("confident", False)
        style_conf = style_field.get("confident", False)

        # Drop generic-style label from intro phrase entirely — the words
        # "vintage / decorative / retro" don't help disambiguate postcards
        # and produce repeating, non-informative intros across the corpus.
        # Type stays; style stays in `search_text` regardless.
        if self.drop_generic_style and style_label in self._GENERIC_STYLE_LABELS:
            style_conf = False
            style_ru_gen = None

        # Special safe cases for photographs
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

    # ====================================================================
    # Versioned RU mappings — selected based on metadata["taxonomy_version"].
    # See src/tsu_image_description/siglip_metadata_extractor.py for full
    # taxonomy definitions and source citations.
    # ====================================================================

    # Common мапы типа материала — pattern одинаковый между versions.
    _IMAGE_TYPE_MAP_COMMON = {
        "a postcard": "открытка",
        "a poster": "плакат",
        "a greeting card": "поздравительная карточка",
        "an illustration": "иллюстрация",
        "a photograph": "фотография",
    }

    # ---- legacy_v1 ----
    _STYLE_MAP_LEGACY = {
        "vintage illustration": "винтажная иллюстрация",
        "retro design": "ретро-дизайн",
        "decorative illustration": "декоративная иллюстрация",
        "engraving": "гравюра",
        "drawing": "рисунок",
        "painting": "живопись",
        "black and white photo": "черно-белая фотография",
        "color photograph": "цветная фотография",
    }
    _STYLE_MAP_GEN_LEGACY = {
        "vintage illustration": "винтажной иллюстрации",
        "retro design": "ретро-дизайна",
        "decorative illustration": "декоративной иллюстрации",
        "engraving": "гравюры",
        "drawing": "рисунка",
        "painting": "живописи",
        "black and white photo": "черно-белой фотографии",
        "color photograph": "цветной фотографии",
    }
    _THEME_MAP_LEGACY = {
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
    _MOOD_MAP_LEGACY = {
        "happy": "радостное",
        "festive": "праздничное",
        "romantic": "романтическое",
        "nostalgic": "ностальгическое",
        "calm": "спокойное",
        "serious": "серьёзное",
    }

    # ---- archival_v2 ----
    _STYLE_MAP_ARCHIVAL = {
        "a chromolithograph": "хромолитография",
        "an engraving": "гравюра",
        "an etching": "офорт",
        "a watercolor painting": "акварель",
        "an oil painting": "масляная живопись",
        "a pencil drawing": "карандашный рисунок",
        "a black and white photograph": "черно-белая фотография",
        "a color photograph": "цветная фотография",
    }
    _STYLE_MAP_GEN_ARCHIVAL = {
        "a chromolithograph": "хромолитографии",
        "an engraving": "гравюры",
        "an etching": "офорта",
        "a watercolor painting": "акварели",
        "an oil painting": "масляной живописи",
        "a pencil drawing": "карандашного рисунка",
        "a black and white photograph": "черно-белой фотографии",
        "a color photograph": "цветной фотографии",
    }
    _THEME_MAP_ARCHIVAL = {
        "a landscape": "пейзаж",
        "an urban view": "городской вид",
        "a portrait": "портрет",
        "a genre scene": "жанровая сцена",
        "a still life": "натюрморт",
        "a religious subject": "религиозный сюжет",
        "a military subject": "военный сюжет",
        "a holiday scene": "праздничный сюжет",
    }
    _MOOD_MAP_ARCHIVAL: Dict = {}  # mood удалён в archival_v2

    @classmethod
    def _image_type_map_for(cls, version: str) -> Dict:
        # type mapping одинаковый для обеих versions
        return cls._IMAGE_TYPE_MAP_COMMON

    @classmethod
    def _style_map_for(cls, version: str) -> Dict:
        return cls._STYLE_MAP_ARCHIVAL if version == "archival_v2" else cls._STYLE_MAP_LEGACY

    @classmethod
    def _style_map_gen_for(cls, version: str) -> Dict:
        return cls._STYLE_MAP_GEN_ARCHIVAL if version == "archival_v2" else cls._STYLE_MAP_GEN_LEGACY

    @classmethod
    def _theme_map_for(cls, version: str) -> Dict:
        return cls._THEME_MAP_ARCHIVAL if version == "archival_v2" else cls._THEME_MAP_LEGACY

    @classmethod
    def _mood_map_for(cls, version: str) -> Dict:
        return cls._MOOD_MAP_ARCHIVAL if version == "archival_v2" else cls._MOOD_MAP_LEGACY

    @classmethod
    def _tags_ru(cls, metadata: Dict, inference: Dict) -> list[str]:
        """Возвращает русские теги по уверенным предсказаниям классификатора.

        В отличие от сырого metadata.tags (англоязычные исходные метки SigLIP),
        этот список содержит локализованные термины из каталожной таксономии
        и пригоден для отображения в UI без дополнительной обработки.
        """
        version = metadata.get("taxonomy_version", "legacy_v1")
        image_type_map = cls._image_type_map_for(version)
        style_map = cls._style_map_for(version)
        theme_map = cls._theme_map_for(version)
        mood_map = cls._mood_map_for(version)

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
    def _search_text(cls, metadata: Dict, inference: Dict, caption_ru: str | None = None) -> str:
        image_type_field = metadata.get("image_type", {})
        style_field = metadata.get("style", {})
        tags = metadata.get("tags", [])
        version = metadata.get("taxonomy_version", "legacy_v1")

        image_type_map = cls._image_type_map_for(version)
        style_map = cls._style_map_for(version)
        theme_map = cls._theme_map_for(version)
        mood_map = cls._mood_map_for(version)

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

        return " ".join(search_terms)
