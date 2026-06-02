"""Тесты очистки английской подписи BLIP (EnglishCaptionPostprocessor)."""

import pytest

from tsu_image_description.english_caption_postprocessor import (
    EnglishCaptionPostprocessor,
)


@pytest.fixture
def pp():
    return EnglishCaptionPostprocessor()


def test_empty_returns_empty(pp):
    assert pp.clean("") == ""


@pytest.mark.parametrize(
    "src, expected",
    [
        ("there is a dog", "a dog"),
        ("there are dogs", "dogs"),
        ("a close up of a cat", "a cat"),
    ],
)
def test_removes_leading_patterns(pp, src, expected):
    assert pp.clean(src) == expected


def test_removes_arafed_token(pp):
    assert pp.clean("arafed dog") == "dog"


def test_collapses_doubled_words(pp):
    assert pp.clean("a sled sled in snow") == "a sled in snow"


def test_doubled_words_case_insensitive(pp):
    assert pp.clean("The the dog") == "The dog"


def test_collapses_of_the_repeats(pp):
    src = "of the date of the date of the date"
    assert pp.clean(src) == "of the date"


def test_strips_trailing_punctuation(pp):
    assert pp.clean("a dog.") == "a dog"
