"""Тесты объединённого CaptionCleaner (clean_en + clean_ru).

Перенесены 1:1 из тестов прежних двух постпроцессоров: английская очистка
подписи BLIP и русская очистка после перевода — общий мусорный токен arafed.
"""

import pytest

from tsu_image_description.caption_cleaner import CaptionCleaner


@pytest.fixture
def cc():
    return CaptionCleaner()


# ---- clean_en: английская подпись BLIP ----

def test_en_empty_returns_empty(cc):
    assert cc.clean_en("") == ""


@pytest.mark.parametrize(
    "src, expected",
    [
        ("there is a dog", "a dog"),
        ("there are dogs", "dogs"),
        ("a close up of a cat", "a cat"),
    ],
)
def test_en_removes_leading_patterns(cc, src, expected):
    assert cc.clean_en(src) == expected


def test_en_removes_arafed_token(cc):
    assert cc.clean_en("arafed dog") == "dog"


def test_en_collapses_doubled_words(cc):
    assert cc.clean_en("a sled sled in snow") == "a sled in snow"


def test_en_doubled_words_case_insensitive(cc):
    assert cc.clean_en("The the dog") == "The dog"


def test_en_collapses_of_the_repeats(cc):
    src = "of the date of the date of the date"
    assert cc.clean_en(src) == "of the date"


def test_en_strips_trailing_punctuation(cc):
    assert cc.clean_en("a dog.") == "a dog"


# ---- clean_ru: русская подпись после перевода ----

def test_ru_empty_returns_empty(cc):
    assert cc.clean_ru("") == ""
    assert cc.clean_ru(None) == ""


def test_ru_collapses_whitespace(cc):
    assert cc.clean_ru("большая    собака") == "большая собака"


def test_ru_removes_space_before_punctuation(cc):
    assert cc.clean_ru("кошка , дом") == "кошка, дом"


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
def test_ru_strips_leading_fillers(cc, src, expected):
    assert cc.clean_ru(src) == expected


def test_ru_only_first_leading_filler_removed(cc):
    # после первого совпадения цикл прерывается, второй зачин остаётся
    assert cc.clean_ru("там есть собака") == "есть собака"


def test_ru_removes_arafed_token(cc):
    assert cc.clean_ru("arafed собака") == "собака"


def test_ru_removes_translated_arafed_token(cc):
    assert cc.clean_ru("арафированная собака") == "собака"


def test_ru_fallback_when_fully_cleaned(cc):
    # строка из одного мусорного токена вычищается полностью -> фолбэк
    assert cc.clean_ru("arafed") == "изображение"


def test_ru_strips_trailing_punctuation(cc):
    assert cc.clean_ru("собака.") == "собака"
