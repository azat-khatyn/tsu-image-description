"""Тесты гейтинга и санити-фильтра OCRExtractor.

Модель не загружается: гейтинг (_passes_confidence/_looks_like_text/_normalize)
вычислителен, а extract() тестируется через object.__new__ с подменой _run_ocr,
по образцу test_siglip_pack_field.py. Так paddleocr не требуется для тестов.
"""

from tsu_image_description.ocr_extractor import OCRExtractor


def make_extractor(**over):
    ext = object.__new__(OCRExtractor)
    ext.min_confidence = over.get("min_confidence", 0.6)
    ext.min_text_len = over.get("min_text_len", 3)
    ext.min_letter_ratio = over.get("min_letter_ratio", 0.5)
    return ext


# ---- санити-фильтр ----

def test_looks_like_text_accepts_printed_caption():
    assert OCRExtractor._looks_like_text("Видъ на прудъ, городъ") is True


def test_looks_like_text_rejects_too_short():
    assert OCRExtractor._looks_like_text("ок") is False


def test_looks_like_text_rejects_no_long_token():
    # только короткие токены (<3 букв) — не текст
    assert OCRExtractor._looks_like_text("a b c d") is False


def test_looks_like_text_rejects_punctuation_noise():
    # типичный мусор NEB на лицевых сторонах без текста
    assert OCRExtractor._looks_like_text("|! — . :: |") is False


def test_looks_like_text_low_letter_ratio_rejected():
    assert OCRExtractor._looks_like_text("12345 !!! 67", min_letter_ratio=0.5) is False


# ---- порог уверенности ----

def test_passes_confidence():
    assert OCRExtractor._passes_confidence(0.92, 0.6) is True
    assert OCRExtractor._passes_confidence(0.4, 0.6) is False
    assert OCRExtractor._passes_confidence(None, 0.6) is False


# ---- нормализация ----

def test_normalize_collapses_whitespace():
    assert OCRExtractor._normalize("Видъ\nна   прудъ\n") == "Видъ на прудъ"
    assert OCRExtractor._normalize(None) == ""


# ---- empty_result ----

def test_empty_result():
    r = OCRExtractor.empty_result()
    assert r == {"text": "", "raw_text": "", "confidence": None, "confident": False}


# ---- extract() с подменой движка ----

def test_extract_confident_on_printed_text():
    ext = make_extractor()
    ext._run_ocr = lambda p: ("Видъ на прудъ", 0.92)
    out = ext.extract("x.jpg")
    assert out["confident"] is True
    assert out["text"] == "Видъ на прудъ"
    assert out["confidence"] == 0.92


def test_extract_not_confident_low_score():
    ext = make_extractor()
    ext._run_ocr = lambda p: ("Видъ на прудъ", 0.3)
    out = ext.extract("x.jpg")
    assert out["confident"] is False
    # текст всё равно возвращается (для отладки), но гейт не пройден
    assert out["text"] == "Видъ на прудъ"


def test_extract_not_confident_on_noise():
    ext = make_extractor()
    ext._run_ocr = lambda p: ("|! — .", 0.95)
    out = ext.extract("x.jpg")
    assert out["confident"] is False


def test_extract_empty_when_no_text():
    ext = make_extractor()
    ext._run_ocr = lambda p: ("", None)
    out = ext.extract("x.jpg")
    assert out["confident"] is False
    assert out["text"] == ""
