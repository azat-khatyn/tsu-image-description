"""Тесты гейтинга темы и настроения по уверенности.

Гейтинг живёт рядом с производителем метаданных:
SigLIPMetadataExtractor.infer_theme_mood. Тест не загружает модель —
infer_theme_mood это staticmethod без обращения к весам.
"""

from tsu_image_description.siglip_metadata_extractor import SigLIPMetadataExtractor

infer = SigLIPMetadataExtractor.infer_theme_mood


def test_confident_fields_passed_through():
    md = {
        "theme": {"label": "a landscape", "confident": True, "score": 0.7},
        "mood": {"label": "calm", "confident": True, "score": 0.6},
    }
    out = infer(md)
    assert out["theme"] == "a landscape"
    assert out["mood"] == "calm"
    assert out["theme_confidence"] == 0.7
    assert out["mood_confidence"] == 0.6


def test_non_confident_label_dropped_but_score_kept():
    md = {
        "theme": {"label": "a landscape", "confident": False, "score": 0.4},
        "mood": {"label": "calm", "confident": False, "score": 0.3},
    }
    out = infer(md)
    assert out["theme"] is None
    assert out["mood"] is None
    # балл возвращается независимо от флага уверенности
    assert out["theme_confidence"] == 0.4
    assert out["mood_confidence"] == 0.3


def test_missing_fields():
    out = infer({})
    assert out["theme"] is None
    assert out["mood"] is None
    assert out["theme_confidence"] is None
    assert out["mood_confidence"] is None
