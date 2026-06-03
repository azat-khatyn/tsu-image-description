"""CaptionCleaner — очистка артефактов подписи на обоих языках пайплайна.

Объединяет правила для английской подписи BLIP (clean_en) и русской после
перевода (clean_ru). Лечит только известные классы артефактов (мусорные токены
модели, шаблонные зачины); точечных правок под конкретные изображения нет —
их лечит языковой редактор LLM.
"""

import re
import unicodedata


class CaptionCleaner:
    """Очистка английской и русской подписи от артефактов BLIP/MarianMT."""

    # мусорные токены-галлюцинации BLIP-1 (в т.ч. протёкшие в русский перевод)
    _BAD_TOKEN_PATTERNS = [
        r"\b(?:arafed|арафed)\b",
        r"\bараф(?:ирован(?:ная|ный|ное|ные|ного|ному|ным|ной)?)\b",
    ]

    # шаблонные зачины русского перевода из паттернов there is / there are
    _LEADING_FILLERS = [
        r"^есть\s+",
        r"^там\s+",
        r"^имеется\s+",
        r"^можно\s+увидеть\s+",
        r"^на\s+изображении\s+(?:видно|изображено)\s*[:,-]?\s*",
    ]

    def clean_en(self, text: str) -> str:
        """Очистка английской подписи BLIP: битые токены, повторы, обрывы."""
        if not text:
            return ""

        text = text.strip()

        # битые токены и артефакты
        for pattern in self._BAD_TOKEN_PATTERNS:
            text = re.sub(pattern, " ", text, flags=re.IGNORECASE)

        # повторяющиеся зачины
        text = re.sub(r"^(there is|there are)\s+", "", text, flags=re.IGNORECASE)
        text = re.sub(r"^a close up of\s+", "", text, flags=re.IGNORECASE)

        # сдвоенные слова ("sled sled")
        text = re.sub(r"\b(\w+)\s+\1\b", r"\1", text, flags=re.IGNORECASE)

        # повторы вида "of the date of the date..."
        text = re.sub(r"(of the\s+\w+)(\s+of the\s+\w+){2,}", r"\1", text, flags=re.IGNORECASE)

        # нормализация пробелов
        text = re.sub(r"\s+", " ", text)
        text = text.strip(" ,.;:-")

        return text

    def clean_ru(self, text: str) -> str:
        """Очистка русской подписи после перевода: токены, зачины, пунктуация."""
        if not text:
            return ""

        text = unicodedata.normalize("NFKC", text).strip()

        # нормализация пробелов вокруг пунктуации
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"\s+([,.;:!?])", r"\1", text)

        # удаление известных мусорных токенов модели
        for pattern in self._BAD_TOKEN_PATTERNS:
            text = re.sub(pattern, " ", text, flags=re.IGNORECASE)

        # удаление известных зачинов из паттернов BLIP-1
        for pattern in self._LEADING_FILLERS:
            new_text = re.sub(pattern, "", text, flags=re.IGNORECASE)
            if new_text != text:
                text = new_text
                break

        # повторная чистка пробелов и пунктуации после замен
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"\s+([,.;:!?])", r"\1", text)
        text = text.strip(" \t\n\r,;:.!-")

        # фолбэк, если текст полностью вычищен
        if not text:
            return "изображение"

        return text
