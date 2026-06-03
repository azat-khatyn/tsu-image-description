"""Тесты логики gating экстрактора метаданных (порог + отрыв от второго).

Модель не загружается: методы _top_k / _pack_field / _empty_field чисто
вычислительные, поэтому экземпляр создаётся через object.__new__ в обход
__init__ (который иначе создал бы классификатор и скачал веса).
"""

import pytest

from tsu_image_description.metadata_extractor import MetadataExtractor


def make_extractor():
    return object.__new__(MetadataExtractor)


def test_top_k_orders_and_rounds():
    ext = make_extractor()
    top = ext._top_k({"a": 0.1, "b": 0.5, "c": 0.30001}, k=2)
    assert [t["label"] for t in top] == ["b", "c"]
    assert top[1]["score"] == 0.3


def test_pack_field_confident_when_threshold_and_margin_met():
    ext = make_extractor()
    field = ext._pack_field({"a": 0.5, "b": 0.4, "c": 0.1}, threshold=0.35, margin=0.05)
    assert field["label"] == "a"
    assert field["confident"] is True
    assert field["margin"] == pytest.approx(0.1)


def test_pack_field_not_confident_when_margin_too_small():
    ext = make_extractor()
    field = ext._pack_field({"a": 0.5, "b": 0.48}, threshold=0.35, margin=0.05)
    assert field["confident"] is False


def test_pack_field_not_confident_when_below_threshold():
    ext = make_extractor()
    field = ext._pack_field({"a": 0.30, "b": 0.05}, threshold=0.35, margin=0.05)
    assert field["confident"] is False


def test_empty_field():
    ext = make_extractor()
    field = ext._empty_field()
    assert field["label"] is None
    assert field["confident"] is False
    assert field["alternatives"] == []
