"""Сводка надёжности ответа из уже посчитанных сигналов уверенности.

Агрегирует per-field уверенность классификатора (SigLIP) и уверенность OCR в
один блок, чтобы по ответу было видно, насколько ему можно доверять. Чистая
функция без моделей.

`level` отражает уверенность СТРУКТУРИРОВАННЫХ полей (тип/стиль/тема). Семантическое
соответствие всего описания изображению — отдельный reference-free сигнал
`clipscore` (опциональный, считается отдельным компонентом и прокидывается сюда).
Подпись BLIP собственной уверенности не имеет — её надёжность отражает только clipscore.
"""

from typing import Dict

# Оси метаданных по порядку; mood в archival_v2 пуст и пропускается (label=None).
_AXES = ("image_type", "style", "theme", "mood")


def assess(metadata: Dict, ocr: Dict | None = None, clipscore: float | None = None) -> Dict:
    """Собирает блок надёжности из метаданных, OCR и (опц.) CLIPScore."""
    confident: list[str] = []
    scores: list[float] = []
    n_applicable = 0

    for axis in _AXES:
        field = metadata.get(axis) or {}
        if field.get("label") is None:
            continue  # пустая ось (например, mood в archival_v2)
        n_applicable += 1
        if field.get("confident"):
            confident.append(axis)
            if field.get("score") is not None:
                scores.append(float(field["score"]))

    mean_confidence = (sum(scores) / len(scores)) if scores else None
    ocr_confidence = ocr.get("confidence") if (ocr and ocr.get("confident")) else None

    return {
        "level": _level(len(confident), n_applicable),
        "confident_fields": confident,
        "n_confident": len(confident),
        "n_applicable": n_applicable,
        "mean_confidence": mean_confidence,
        "ocr_confidence": ocr_confidence,
        "clipscore": clipscore,
    }


def _level(n_confident: int, n_applicable: int) -> str:
    """Грубая эвристика уровня доверия по доле уверенных осей классификатора.

    high >= 2/3, medium > 1/3, иначе low (<= 1/3); unknown — если осей нет.
    """
    if n_applicable == 0:
        return "unknown"
    ratio = n_confident / n_applicable
    if ratio >= 2 / 3:
        return "high"
    if ratio > 1 / 3:
        return "medium"
    return "low"
