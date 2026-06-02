"""Тесты очистки русской подписи (TextPostprocessor)."""

import pytest

from tsu_image_description.text_postprocessor import TextPostprocessor


@pytest.fixture
def pp():
    return TextPostprocessor()


def test_empty_returns_empty(pp):
    assert pp.clean_ru_caption("") == ""
    assert pp.clean_ru_caption(None) == ""


def test_collapses_whitespace(pp):
    assert pp.clean_ru_caption("большая    собака") == "большая собака"


def test_removes_space_before_punctuation(pp):
    assert pp.clean_ru_caption("кошка , дом") == "кошка, дом"


@pytest.mark.parametrize(
    "src, expected",
    [
        ("там большая собака", "большая собака"),
        ("есть кошка на столе", "кошка на столе"),
        ("имеется корабль", "корабль"),
        ("можно увидеть дом", "дом"),
        ("на изображении видно: корабль", "корабль"),
        ("на изображении изображено - лес", "лес"),
    ],
)
def test_strips_leading_fillers(pp, src, expected):
    assert pp.clean_ru_caption(src) == expected


def test_only_first_leading_filler_removed(pp):
    # после первого совпадения цикл прерывается, второй зачин остаётся
    assert pp.clean_ru_caption("там есть собака") == "есть собака"


def test_removes_arafed_token(pp):
    assert pp.clean_ru_caption("arafed собака") == "собака"


def test_removes_translated_arafed_token(pp):
    assert pp.clean_ru_caption("арафированная собака") == "собака"


def test_fallback_when_fully_cleaned(pp):
    # строка из одного мусорного токена вычищается полностью -> фолбэк
    assert pp.clean_ru_caption("arafed") == "изображение"


def test_strips_trailing_punctuation(pp):
    assert pp.clean_ru_caption("собака.") == "собака"
