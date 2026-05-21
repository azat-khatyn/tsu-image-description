import re


class EnglishCaptionPostprocessor:
    def clean(self, text: str) -> str:
        if not text:
            return ""

        text = text.strip()

        # broken tokens / artifacts
        text = re.sub(r"\barafed\b", "", text, flags=re.IGNORECASE)

        # repetitive lead-ins
        text = re.sub(r"^(there is|there are)\s+", "", text, flags=re.IGNORECASE)
        text = re.sub(r"^a close up of\s+", "", text, flags=re.IGNORECASE)

        # duplicated words like "sled sled"
        text = re.sub(r"\b(\w+)\s+\1\b", r"\1", text, flags=re.IGNORECASE)

        # repeated "of the date of the date..."
        text = re.sub(r"(of the\s+\w+)(\s+of the\s+\w+){2,}", r"\1", text, flags=re.IGNORECASE)

        # normalize quotes/spaces
        text = re.sub(r"\s+", " ", text)
        text = text.strip(" ,.;:-")

        return text
