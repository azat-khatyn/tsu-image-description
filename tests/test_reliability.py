"""Тесты сводки надёжности (reliability.assess) — без моделей."""

from tsu_image_description import reliability


def _meta(image_type=None, style=None, theme=None, mood=None):
    return {
        "image_type": image_type or {"label": None, "confident": False},
        "style": style or {"label": None, "confident": False},
        "theme": theme or {"label": None, "confident": False},
        "mood": mood or {"label": None, "confident": False},
    }


def test_empty_axes_give_unknown():
    out = reliability.assess(_meta())
    assert out["level"] == "unknown"
    assert out["n_applicable"] == 0
    assert out["n_confident"] == 0
    assert out["mean_confidence"] is None


def test_all_confident_high():
    out = reliability.assess(_meta(
        image_type={"label": "a postcard", "confident": True, "score": 0.9},
        style={"label": "an etching", "confident": True, "score": 0.6},
        theme={"label": "a landscape", "confident": True, "score": 0.5},
    ))
    assert out["level"] == "high"
    assert out["confident_fields"] == ["image_type", "style", "theme"]
    assert out["n_confident"] == 3
    assert out["n_applicable"] == 3
    assert abs(out["mean_confidence"] - (0.9 + 0.6 + 0.5) / 3) < 1e-9


def test_one_of_three_low():
    out = reliability.assess(_meta(
        image_type={"label": "a postcard", "confident": True, "score": 0.9},
        style={"label": "an etching", "confident": False, "score": 0.3},
        theme={"label": "a landscape", "confident": False, "score": 0.2},
    ))
    assert out["level"] == "low"
    assert out["n_confident"] == 1
    assert out["mean_confidence"] == 0.9   # усредняем только уверенные


def test_two_of_three_high_boundary():
    out = reliability.assess(_meta(
        image_type={"label": "a postcard", "confident": True, "score": 0.8},
        style={"label": "an etching", "confident": True, "score": 0.5},
        theme={"label": "a landscape", "confident": False, "score": 0.2},
    ))
    assert out["level"] == "high"   # 2/3 >= 2/3


def test_empty_mood_not_counted():
    # mood пуст (label=None) — не входит в n_applicable
    out = reliability.assess(_meta(
        image_type={"label": "a postcard", "confident": True, "score": 0.9},
    ))
    assert out["n_applicable"] == 1
    assert out["level"] == "high"


def test_ocr_confidence_only_when_confident():
    md = _meta(image_type={"label": "a postcard", "confident": True, "score": 0.9})
    assert reliability.assess(md, ocr={"confident": True, "confidence": 0.92})["ocr_confidence"] == 0.92
    assert reliability.assess(md, ocr={"confident": False, "confidence": 0.4})["ocr_confidence"] is None
    assert reliability.assess(md, ocr=None)["ocr_confidence"] is None


def test_clipscore_passthrough():
    md = _meta(image_type={"label": "a postcard", "confident": True, "score": 0.9})
    assert reliability.assess(md, clipscore=0.31)["clipscore"] == 0.31
    assert reliability.assess(md)["clipscore"] is None
