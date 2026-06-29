"""TesseractOCRExtractor — резервный OCR backend.

Используется, когда основной PaddleOCR недоступен, завершился ошибкой
или выбран явно через OCR_BACKEND=tesseract.

Контракт результата совпадает с OCRExtractor:
{text, raw_text, confidence, confident, backend}.
"""

from __future__ import annotations

import re
from typing import Any

import pytesseract
from PIL import Image
from pytesseract import Output


_LETTER_TOKEN = re.compile(r"[^\W\d_]+", re.UNICODE)


class TesseractOCRExtractor:
    """Извлечение текста через Tesseract OCR."""

    backend_name = "tesseract"

    def __init__(
        self,
        *,
        lang: str = "rus+eng",
        min_confidence: float = 0.6,
        min_text_len: int = 3,
        min_letter_ratio: float = 0.5,
        psm: int = 6,
    ) -> None:
        self.lang = lang
        self.min_confidence = min_confidence
        self.min_text_len = min_text_len
        self.min_letter_ratio = min_letter_ratio
        self.psm = psm

        try:
            pytesseract.get_tesseract_version()
        except pytesseract.TesseractNotFoundError as exc:
            raise RuntimeError(
                "Tesseract не найден. Установите системный пакет tesseract-ocr "
                "и языковые данные tesseract-ocr-rus."
            ) from exc

    def extract(self, image_path: str) -> dict[str, Any]:
        image = Image.open(image_path).convert("RGB")

        data = pytesseract.image_to_data(
            image,
            lang=self.lang,
            config=f"--oem 1 --psm {self.psm}",
            output_type=Output.DICT,
        )

        tokens: list[str] = []
        confidences: list[float] = []

        for token, raw_confidence in zip(data["text"], data["conf"]):
            token = (token or "").strip()

            try:
                confidence = float(raw_confidence)
            except (TypeError, ValueError):
                continue

            # Отрицательные значения Tesseract означают служебную позицию.
            if not token or confidence < 0:
                continue

            tokens.append(token)
            confidences.append(confidence / 100.0)

        raw_text = " ".join(tokens).strip()
        text = self._normalize(raw_text)

        confidence = (
            sum(confidences) / len(confidences)
            if confidences
            else None
        )

        confident = (
            self._passes_confidence(confidence, self.min_confidence)
            and self._looks_like_text(
                text,
                min_len=self.min_text_len,
                min_letter_ratio=self.min_letter_ratio,
            )
        )

        return {
            "text": text,
            "raw_text": raw_text,
            "confidence": confidence,
            "confident": confident,
            "backend": self.backend_name,
        }

    @staticmethod
    def _normalize(text: str) -> str:
        return " ".join((text or "").split())

    @staticmethod
    def _passes_confidence(
        confidence: float | None,
        min_confidence: float,
    ) -> bool:
        return confidence is not None and confidence >= min_confidence

    @staticmethod
    def _looks_like_text(
        text: str,
        *,
        min_len: int = 3,
        min_letter_ratio: float = 0.5,
    ) -> bool:
        text = (text or "").strip()

        if len(text) < min_len:
            return False

        if not any(len(token) >= 3 for token in _LETTER_TOKEN.findall(text)):
            return False

        non_space = sum(not char.isspace() for char in text)
        letters = sum(char.isalpha() for char in text)

        if non_space == 0:
            return False

        return letters / non_space >= min_letter_ratio
