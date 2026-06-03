"""Тесты сборки русского архивного описания (DescriptionBuilder)."""

import pytest

from tsu_image_description.description_builder import DescriptionBuilder


def make_result(
    *,
    caption_ru="зимний пейзаж",
    image_type=None,
    style=None,
    theme=None,
    mood=None,
    tags=None,
    taxonomy_version="archival_v2",
    inf_theme=None,
    inf_mood=None,
    ocr=None,
):
    return {
        "caption": {"ru": caption_ru},
        "metadata": {
            "image_type": image_type or {"label": None, "confident": False},
            "style": style or {"label": None, "confident": False},
            "theme": theme or {"label": None, "confident": False},
            "mood": mood or {"label": None, "confident": False},
            "tags": tags or [],
            "taxonomy_version": taxonomy_version,
        },
        "inference": {"theme": inf_theme, "mood": inf_mood},
        "ocr": ocr,
    }


def test_invalid_template_mode_raises():
    with pytest.raises(ValueError):
        DescriptionBuilder(template_mode="bogus")


def test_caption_only_mode():
    builder = DescriptionBuilder(template_mode="caption_only")
    out = builder.build(make_result(caption_ru="красивый дом"))
    assert out["archive_description"] == "Красивый дом."


def test_minimal_mode_lowercases_caption():
    builder = DescriptionBuilder(template_mode="minimal")
    out = builder.build(make_result(caption_ru="Красивый дом"))
    assert out["archive_description"] == "На изображении: красивый дом."


def test_full_mode_intro_image_type_and_style():
    builder = DescriptionBuilder()
    out = builder.build(
        make_result(
            caption_ru="зимний пейзаж",
            image_type={"label": "a postcard", "confident": True},
            style={"label": "a chromolithograph", "confident": True},
        )
    )
    assert out["archive_description"].startswith(
        "Открытка в стиле хромолитографии. На изображении: зимний пейзаж."
    )


def test_full_mode_includes_theme_sentence():
    builder = DescriptionBuilder()
    out = builder.build(
        make_result(
            image_type={"label": "a postcard", "confident": True},
            inf_theme="a landscape",
        )
    )
    assert "Тематика изображения: пейзаж." in out["archive_description"]


def test_photograph_special_case_legacy():
    builder = DescriptionBuilder()
    out = builder.build(
        make_result(
            image_type={"label": "a photograph", "confident": True},
            style={"label": "black and white photo", "confident": True},
            taxonomy_version="legacy_v1",
        )
    )
    assert out["archive_description"].startswith("Черно-белая фотография.")


def test_drop_generic_style_removes_vintage_from_intro():
    builder = DescriptionBuilder(drop_generic_style=True)
    out = builder.build(
        make_result(
            image_type={"label": "a postcard", "confident": True},
            style={"label": "vintage illustration", "confident": True},
            taxonomy_version="legacy_v1",
        )
    )
    # generic-стиль убран, остаётся только тип материала
    assert out["archive_description"].startswith("Открытка. ")
    assert "стиле" not in out["archive_description"]


def test_search_text_uses_ru_terms_not_raw_english_tags():
    builder = DescriptionBuilder()
    out = builder.build(
        make_result(
            caption_ru="Зимний Пейзаж",
            image_type={"label": "a postcard", "confident": True},
            style={"label": "a chromolithograph", "confident": True},
            tags=["a postcard", "a chromolithograph"],
            inf_theme="a landscape",
        )
    )
    search = out["search_text"]
    assert search == "открытка хромолитография пейзаж зимний пейзаж"
    assert "postcard" not in search


def test_search_text_appends_confident_ocr_after_caption():
    builder = DescriptionBuilder()
    out = builder.build(
        make_result(
            caption_ru="вид на пруд",
            image_type={"label": "a postcard", "confident": True},
            ocr={"text": "Видъ на прудъ", "confidence": 0.92, "confident": True},
        )
    )
    # надпись OCR дописывается в конце, после подписи
    assert out["search_text"] == "открытка вид на пруд видъ на прудъ"


def test_search_text_ignores_non_confident_ocr():
    builder = DescriptionBuilder()
    out = builder.build(
        make_result(
            caption_ru="вид на пруд",
            image_type={"label": "a postcard", "confident": True},
            ocr={"text": "|! — .", "confidence": 0.95, "confident": False},
        )
    )
    assert "|!" not in out["search_text"]
    assert out["search_text"] == "открытка вид на пруд"


def test_search_text_without_ocr_block_unchanged():
    builder = DescriptionBuilder()
    out = builder.build(
        make_result(
            caption_ru="вид на пруд",
            image_type={"label": "a postcard", "confident": True},
            ocr=None,
        )
    )
    assert out["search_text"] == "открытка вид на пруд"


def test_tags_ru_localized():
    builder = DescriptionBuilder()
    out = builder.build(
        make_result(
            image_type={"label": "a postcard", "confident": True},
            style={"label": "a chromolithograph", "confident": True},
            inf_theme="a landscape",
        )
    )
    assert out["tags_ru"] == ["открытка", "хромолитография", "пейзаж"]


def test_non_confident_fields_give_generic_intro():
    builder = DescriptionBuilder()
    out = builder.build(make_result(caption_ru="нечто"))
    assert out["archive_description"].startswith("Изображение. ")
