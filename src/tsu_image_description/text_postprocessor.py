import re
import unicodedata


class TextPostprocessor:
    """Class-level cleanup for translated RU captions.

    Принцип: содержит ТОЛЬКО правила, лечащие *известные классы* артефактов
    моделей пайплайна (BLIP-token residue, MarianMT lead-in artefacts). Не
    содержит case-specific patches под конкретные тестовые изображения —
    такие проблемы должны лечиться более сильной моделью (см. LLM rewriter
    в section_3_architecture.md), а не разрастающимся словарём regex.

    Текущие лечимые классы:
      1. BLIP-1 hallucination token `arafed`, протекающий в RU
         (`арафированн*`). EN-сторона тоже его стрипает; держим parallel
         защиту на случай прямого input.
      2. Leading fillers `есть/там/имеется/можно увидеть`, переведённые с
         BLIP-1 паттернов `there is / there are`. EN-сторона тоже их
         стрипает; держим parallel защиту на случай прямого input.
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

        # Normalize spacing around punctuation
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"\s+([,.;:!?])", r"\1", text)

        # Class-level: remove known model artifact tokens
        for pattern in self._BAD_TOKEN_PATTERNS:
            text = re.sub(pattern, " ", text, flags=re.IGNORECASE)

        # Class-level: remove known leading-filler patterns from BLIP-1 lead-ins
        for pattern in self._LEADING_FILLERS:
            new_text = re.sub(pattern, "", text, flags=re.IGNORECASE)
            if new_text != text:
                text = new_text
                break

        # Clean punctuation / double spaces after replacements
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"\s+([,.;:!?])", r"\1", text)
        text = text.strip(" \t\n\r,;:.!-")

        # Fallback if everything got wiped
        if not text:
            return "изображение"

        return text
