"""OCRExtractor — распознавание надписей на открытках (PaddleOCR, lang="ru").

Опциональная стадия пайплайна. По бенчмарку (data/ocr_eval/) PaddleOCR лучший на
печатной кириллице и дореформенной орфографии (conf 0.92–0.98), но на большинстве
лицевых сторон текста нет, а рукописный даёт мусор. Поэтому результат проходит
двойной гейт: порог уверенности движка И санити-фильтр (_looks_like_text). В
подсказку LLM уходит только текст с confident=True — иначе шум попал бы в описание.

paddleocr — тяжёлая опциональная зависимость (requirements-ocr.txt). Импорт
ленивый (в __init__): без библиотеки модуль импортируется, ошибка возникает
только при создании экземпляра, и пайплайн мягко деградирует к пустому блоку.
"""

import os
import re

# буквенные токены (кириллица/латиница), без цифр и подчёркивания
_LETTER_TOKEN = re.compile(r"[^\W\d_]+", re.UNICODE)


class OCRExtractor:
    """Извлекает надпись с изображения и гейтит её по уверенности и санити-фильтру.

    Args:
        lang: язык PaddleOCR.
        min_confidence: минимальная средняя уверенность движка для confident.
        min_text_len: минимальная длина очищенного текста.
        min_letter_ratio: минимальная доля букв среди непробельных символов.
    """

    def __init__(
        self,
        *,
        lang: str = "ru",
        min_confidence: float = 0.6,
        min_text_len: int = 3,
        min_letter_ratio: float = 0.5,
    ):
        self.lang = lang
        self.min_confidence = min_confidence
        self.min_text_len = min_text_len
        self.min_letter_ratio = min_letter_ratio

        # модели берутся из локального кэша, без сетевой проверки источника
        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
        try:
            from paddleocr import PaddleOCR
        except ImportError as e:
            raise ImportError(
                "OCRExtractor требует paddleocr (pip install -r requirements-ocr.txt)."
            ) from e
        # PaddleOCR 3.x: use_angle_cls/show_log убраны из конструктора
        self._ocr = PaddleOCR(lang=lang)

    def extract(self, image_path: str) -> dict:
        """Возвращает блок OCR: {text, raw_text, confidence, confident}.

        text — очищенный текст (всегда), confident — прошёл ли он гейт.
        Потребитель (подсказка LLM) использует text только при confident=True.
        """
        raw_text, confidence = self._run_ocr(image_path)
        text = self._normalize(raw_text)
        confident = (
            self._passes_confidence(confidence, self.min_confidence)
            and self._looks_like_text(
                text, min_len=self.min_text_len, min_letter_ratio=self.min_letter_ratio
            )
        )
        return {
            "text": text,
            "raw_text": raw_text,
            "confidence": confidence,
            "confident": confident,
        }

    def _run_ocr(self, image_path: str):
        """Прогон PaddleOCR: склейка rec_texts, средняя rec_scores (как в бенчмарке)."""
        result = self._ocr.predict(str(image_path))
        lines, confs = [], []
        for page in (result or []):
            if hasattr(page, "get"):
                lines.extend(page.get("rec_texts", []))
                confs.extend(float(s) for s in page.get("rec_scores", []))
        raw_text = "\n".join(lines).strip()
        confidence = (sum(confs) / len(confs)) if confs else None
        return raw_text, confidence

    @staticmethod
    def empty_result() -> dict:
        """Пустой блок OCR (стадия выключена или библиотека недоступна)."""
        return {"text": "", "raw_text": "", "confidence": None, "confident": False}

    @staticmethod
    def _normalize(text: str) -> str:
        """Схлопывает пробелы и переводы строк в один пробел."""
        return " ".join((text or "").split())

    @staticmethod
    def _passes_confidence(confidence, min_confidence: float) -> bool:
        return confidence is not None and confidence >= min_confidence

    @staticmethod
    def _looks_like_text(text: str, *, min_len: int = 3, min_letter_ratio: float = 0.5) -> bool:
        """Санити-фильтр против OCR-мусора (одиночные символы, пунктуация, шум).

        Требует: длину >= min_len, хотя бы один буквенный токен из >=3 букв,
        долю букв среди непробельных символов >= min_letter_ratio.
        """
        s = (text or "").strip()
        if len(s) < min_len:
            return False
        if not any(len(tok) >= 3 for tok in _LETTER_TOKEN.findall(s)):
            return False
        non_space = sum(1 for ch in s if not ch.isspace())
        if not non_space:
            return False
        letters = sum(1 for ch in s if ch.isalpha())
        return (letters / non_space) >= min_letter_ratio
