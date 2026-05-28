import re
import unicodedata


class TextPostprocessor:
    """Очистка русской подписи после перевода.

    Содержит только правила для известных классов артефактов пайплайна
    (мусорные токены BLIP, зачины MarianMT). Точечных правок под конкретные
    изображения здесь нет — такие случаи лечит языковой редактор LLM.

    Лечимые классы:
      1. Токен-галлюцинация `arafed` BLIP-1, протекающий в русский (`арафированн*`).
      2. Начала строк `есть/там/имеется/можно увидеть` из паттернов `there is / there are`.
      3. Нормализация пробелов и пунктуации.
    """

    _BAD_TOKEN_PATTERNS = [
        r"\b(?:arafed|арафed)\b",
        r"\bараф(?:ирован(?:ная|ный|ное|ные|ного|ному|ным|ной)?)\b",
    ]

    _LEADING_FILLERS = [
        r"^есть\s+",
        r"^там\s+",
        r"^имеется\s+",
        r"^можно\s+увидеть\s+",
        r"^на\s+изображении\s+(?:видно|изображено)\s*[:,-]?\s*",
    ]

    def clean_ru_caption(self, text: str) -> str:
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
