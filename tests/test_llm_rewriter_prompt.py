"""Тесты сборки промпта LLM-редактора без загрузки модели.

Проверяется только чистая функция _build_user_prompt: OCR-подсказка попадает
в промпт обоих стилей, а пустая надпись показывается как «—».
"""

import pytest

from tsu_image_description.llm_rewriter import PROMPT_STYLES, _build_user_prompt


@pytest.mark.parametrize("style", ["v1_archival", "v2_curator"])
def test_ocr_text_injected_into_prompt(style):
    prompt = _build_user_prompt(
        PROMPT_STYLES[style],
        caption_en="view of a pond",
        image_type="a postcard (уверенно)",
        style="—",
        theme="—",
        mood="—",
        ocr="Видъ на прудъ, Галичъ",
    )
    assert "Надпись на изображении (OCR): Видъ на прудъ, Галичъ" in prompt
    # правило про OCR присутствует в обоих стилях
    assert "НАДПИСЬ НА ИЗОБРАЖЕНИИ (OCR)" in prompt


@pytest.mark.parametrize("style", ["v1_archival", "v2_curator"])
def test_empty_ocr_renders_dash(style):
    prompt = _build_user_prompt(
        PROMPT_STYLES[style],
        caption_en="view of a pond",
        image_type="a postcard (уверенно)",
        style="—",
        theme="—",
        mood="—",
        ocr="—",
    )
    assert "Надпись на изображении (OCR): —" in prompt
