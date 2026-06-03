"""Единый источник таксономии: англ. зонд + русский канон в одной записи.

Одна запись на метку:
  en      — стабильный id метки И zero-shot зонд SigLIP. По итогу пробы Шага 0
            (data/eval/siglip2_ru_probe.json) зонд остаётся английским: русские
            технические термины роняли точность техники до случайной.
  ru      — канон для русского вывода (именительный падеж).
  ru_gen  — родительный падеж (нужен только стилю: «в стиле <ru_gen>»).
  generic — неинформативная метка стиля (vintage/retro/decorative): подходит
            почти к любой открытке, убирается из вводной фразы при drop_generic_style.

Раньше данные были расщеплены: англ. списки-зонды — в metadata_extractor.py
(TAXONOMIES), русские мапы — в description_builder.py (_*_MAP_*), связанные неявным
строковым ключом. Правка метки требовала синхронных изменений в двух файлах, а
рассинхрон молча возвращал английскую строку. Теперь источник один.

Источники archival_v2: Файнштейн Э.Б. «В мире открытки» (1976); MARC 21 п.655;
Getty AAT (Print processes, Subjects); ГОСТ 7.69-2009. legacy_v1 — первая итерация
(разговорные метки стиля + субъективный mood), сохранена для воспроизводимости.
"""

from typing import Dict, List, Set

FIELDS = ("image_type", "style", "theme", "mood")

TAXONOMY: Dict[str, Dict[str, List[Dict]]] = {
    # ====================================================================
    # legacy_v1 — первая итерация. Разговорные метки стиля и mood-категории.
    # ====================================================================
    "legacy_v1": {
        "image_type": [
            {"en": "a postcard", "ru": "открытка"},
            {"en": "a poster", "ru": "плакат"},
            {"en": "a greeting card", "ru": "поздравительная карточка"},
            {"en": "an illustration", "ru": "иллюстрация"},
            {"en": "a photograph", "ru": "фотография"},
        ],
        "style": [
            {"en": "vintage illustration", "ru": "винтажная иллюстрация", "ru_gen": "винтажной иллюстрации", "generic": True},
            {"en": "retro design", "ru": "ретро-дизайн", "ru_gen": "ретро-дизайна", "generic": True},
            {"en": "decorative illustration", "ru": "декоративная иллюстрация", "ru_gen": "декоративной иллюстрации", "generic": True},
            {"en": "engraving", "ru": "гравюра", "ru_gen": "гравюры"},
            {"en": "drawing", "ru": "рисунок", "ru_gen": "рисунка"},
            {"en": "painting", "ru": "живопись", "ru_gen": "живописи"},
            {"en": "black and white photo", "ru": "черно-белая фотография", "ru_gen": "черно-белой фотографии"},
            {"en": "color photograph", "ru": "цветная фотография", "ru_gen": "цветной фотографии"},
        ],
        "theme": [
            {"en": "holiday scene", "ru": "праздничная сцена"},
            {"en": "Easter holiday scene", "ru": "пасхальная сцена"},
            {"en": "Christmas holiday scene", "ru": "рождественская сцена"},
            {"en": "New Year celebration", "ru": "новогодняя сцена"},
            {"en": "romantic scene", "ru": "романтическая сцена"},
            {"en": "children scene", "ru": "детская сцена"},
            {"en": "urban scene", "ru": "городская сцена"},
            {"en": "nature scene", "ru": "сцена природы"},
            {"en": "religious scene", "ru": "религиозная сцена"},
        ],
        "mood": [
            {"en": "happy", "ru": "радостное"},
            {"en": "festive", "ru": "праздничное"},
            {"en": "romantic", "ru": "романтическое"},
            {"en": "nostalgic", "ru": "ностальгическое"},
            {"en": "calm", "ru": "спокойное"},
            {"en": "serious", "ru": "серьёзное"},
        ],
    },

    # ====================================================================
    # archival_v2 — каталожная таксономия (основная). Техники печати вместо
    # разговорных меток; каталожные темы; поле mood удалено (несубъективная опись).
    # ====================================================================
    "archival_v2": {
        "image_type": [
            {"en": "a postcard", "ru": "открытка"},
            {"en": "a greeting card", "ru": "поздравительная карточка"},
            {"en": "an illustration", "ru": "иллюстрация"},
            {"en": "a photograph", "ru": "фотография"},
            {"en": "a poster", "ru": "плакат"},
        ],
        "style": [
            {"en": "a chromolithograph", "ru": "хромолитография", "ru_gen": "хромолитографии"},
            {"en": "an engraving", "ru": "гравюра", "ru_gen": "гравюры"},
            {"en": "an etching", "ru": "офорт", "ru_gen": "офорта"},
            {"en": "a watercolor painting", "ru": "акварель", "ru_gen": "акварели"},
            {"en": "an oil painting", "ru": "масляная живопись", "ru_gen": "масляной живописи"},
            {"en": "a pencil drawing", "ru": "карандашный рисунок", "ru_gen": "карандашного рисунка"},
            {"en": "a black and white photograph", "ru": "черно-белая фотография", "ru_gen": "черно-белой фотографии"},
            {"en": "a color photograph", "ru": "цветная фотография", "ru_gen": "цветной фотографии"},
        ],
        "theme": [
            {"en": "a landscape", "ru": "пейзаж"},
            {"en": "an urban view", "ru": "городской вид"},
            {"en": "a portrait", "ru": "портрет"},
            {"en": "a genre scene", "ru": "жанровая сцена"},
            {"en": "a still life", "ru": "натюрморт"},
            {"en": "a religious subject", "ru": "религиозный сюжет"},
            {"en": "a military subject", "ru": "военный сюжет"},
            {"en": "a holiday scene", "ru": "праздничный сюжет"},
        ],
        # mood удалён — субъективные суждения не каталожны.
        "mood": [],
    },
}

VERSIONS = tuple(TAXONOMY.keys())
DEFAULT_VERSION = "archival_v2"


def _entries(version: str, field: str) -> List[Dict]:
    if version not in TAXONOMY:
        raise ValueError(f"Unknown taxonomy_version={version!r}. Available: {sorted(VERSIONS)}")
    return TAXONOMY[version][field]


def prompts(version: str, field: str) -> List[str]:
    """Англ. зонды SigLIP (они же стабильные id), в исходном порядке."""
    return [e["en"] for e in _entries(version, field)]


def ru_map(version: str, field: str) -> Dict[str, str]:
    """{en_id: ru} — именительный падеж для вывода."""
    return {e["en"]: e["ru"] for e in _entries(version, field)}


def ru_gen_map(version: str, field: str) -> Dict[str, str]:
    """{en_id: ru_gen} — родительный падеж (только записи со стилем-ru_gen)."""
    return {e["en"]: e["ru_gen"] for e in _entries(version, field) if "ru_gen" in e}


def generic_styles(version: str) -> Set[str]:
    """Множество англ. id «неинформативных» меток стиля."""
    return {e["en"] for e in _entries(version, "style") if e.get("generic")}
