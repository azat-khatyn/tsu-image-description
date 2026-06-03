"""Тесты единого источника таксономии (taxonomy.py).

Гарантируют паритет англ. зонда и русского канона: у каждой метки есть ru,
у каждого стиля — ru_gen, generic-флаг стоит только на разговорных метках.
Это закрывает исходный риск P1 (молчаливый рассинхрон EN/RU).
"""

import pytest

from tsu_image_description import taxonomy


@pytest.mark.parametrize("version", taxonomy.VERSIONS)
@pytest.mark.parametrize("field", taxonomy.FIELDS)
def test_every_prompt_has_ru(version, field):
    prompts = taxonomy.prompts(version, field)
    ru = taxonomy.ru_map(version, field)
    assert list(ru.keys()) == prompts          # порядок и состав совпадают
    assert all(ru[p] for p in prompts)         # ни одной пустой строки


@pytest.mark.parametrize("version", taxonomy.VERSIONS)
def test_every_style_has_genitive(version):
    styles = taxonomy.prompts(version, "style")
    gen = taxonomy.ru_gen_map(version, "style")
    assert set(gen.keys()) == set(styles)
    assert all(gen[s] for s in styles)


def test_generic_styles_only_legacy():
    assert taxonomy.generic_styles("legacy_v1") == {
        "vintage illustration", "retro design", "decorative illustration",
    }
    assert taxonomy.generic_styles("archival_v2") == set()


def test_archival_mood_empty():
    assert taxonomy.prompts("archival_v2", "mood") == []
    assert taxonomy.ru_map("archival_v2", "mood") == {}


def test_known_mappings_stable():
    # id метки (англ.) → русский канон; порядок зондов фиксирован
    assert taxonomy.ru_map("archival_v2", "theme")["a holiday scene"] == "праздничный сюжет"
    assert taxonomy.ru_map("archival_v2", "style")["a chromolithograph"] == "хромолитография"
    assert taxonomy.ru_gen_map("archival_v2", "style")["a chromolithograph"] == "хромолитографии"
    assert taxonomy.prompts("archival_v2", "image_type")[0] == "a postcard"


def test_unknown_version_raises():
    with pytest.raises(ValueError):
        taxonomy.prompts("nope_v9", "theme")
