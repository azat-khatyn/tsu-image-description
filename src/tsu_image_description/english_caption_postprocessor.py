import re


class EnglishCaptionPostprocessor:
    """Очистка артефактов английской подписи BLIP: поломанные токены, повторы, обрывы."""

    def clean(self, text: str) -> str:
        if not text:
            return ""

        text = text.strip()

        # битые токены и артефакты
        text = re.sub(r"\barafed\b", "", text, flags=re.IGNORECASE)

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
