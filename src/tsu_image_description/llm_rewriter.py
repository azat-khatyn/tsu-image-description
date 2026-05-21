"""llm_rewriter.py — E12 component: local-LLM-based archive description rewriter.

Берёт caption_en (от BLIP) + структурированные метаданные (от SigLIP) и
напрямую генерирует русскоязычное архивное описание, минуя literal MT.

Решает:
  - MarianMT polysemy (painting → картина, не покраска)
  - proper-noun transliteration (holly → падуб)
  - грамматику / naturalness (без `фотографию пары` в винительном)
  - template repetition (LLM не повторяет одну фразу 36% корпуса)

Сохраняет П2 (закрытые таксономии): SigLIP labels передаются в prompt как
структурированный контекст, LLM использует их для архивной стандартизации.

Backend по умолчанию — Qwen2.5-3B-Instruct (29 языков нативно, включая
русский; 3B параметров; ~7 GB fp16). Меняется через `model_path`.
"""

from typing import Dict, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .models import get_device


DEFAULT_MODEL = "Vikhrmodels/Vikhr-Nemo-12B-Instruct-R-21-09-24"


SYSTEM_PROMPT = (
    "Ты — каталогизатор Российской государственной библиотеки. "
    "Ты пишешь описания открыток для электронного архивного каталога "
    "в нейтральном, плотном, фактическом стиле — без оценочной лексики "
    "и литературных украшений."
)

# Списки generic-стилей и стандартных категорий, чтобы не дублировать пояснения в prompt.
GENERIC_STYLES_RU = {
    "vintage illustration", "decorative illustration", "retro design",
}

FEW_SHOT_EXAMPLES = """
=== Пример 1 ===
Английская подпись: "three children riding a sled in snowy forest"
Тип материала: a postcard (уверенно)
Художественный стиль: vintage illustration (уверенно)
Тематическая категория: holiday scene (не уверен)
Эмоциональный тон: festive (не уверен)
Описание: Открытка. Дети едут на санях по зимнему лесу.

=== Пример 2 ===
Английская подпись: "ship in the water with a lot of smoke coming out of it"
Тип материала: a postcard (уверенно)
Художественный стиль: engraving (уверенно)
Тематическая категория: romantic scene (не уверен)
Эмоциональный тон: nostalgic (уверенно)
Описание: Открытка-гравюра с изображением корабля в море, из которого идёт дым.

=== Пример 3 ===
Английская подпись: "a christmas card with a box and holly leaves"
Тип материала: a greeting card (уверенно)
Художественный стиль: vintage illustration (уверенно)
Тематическая категория: Christmas holiday scene (уверенно)
Эмоциональный тон: festive (уверенно)
Описание: Поздравительная рождественская открытка с изображением коробки и ветвей остролиста.

=== Пример 4 ===
Английская подпись: "a man with a turban and long beard"
Тип материала: a photograph (уверенно)
Художественный стиль: color photograph (не уверен)
Тематическая категория: religious scene (не уверен)
Эмоциональный тон: serious (уверенно)
Описание: Фотография мужчины с длинной бородой и тюрбаном.

=== Пример 5 ===
Английская подпись: "vintage postcard with a wave crashing on stones at the beach"
Тип материала: a postcard (уверенно)
Художественный стиль: color photograph (уверенно)
Тематическая категория: nature scene (не уверен)
Эмоциональный тон: nostalgic (не уверен)
Описание: Открытка. Цветная фотография волны, разбивающейся о камни на берегу.
"""


USER_PROMPT_TEMPLATE = """Тебе даны:
Английская подпись от vision-модели: "{caption_en}"
Тип материала: {image_type}
Художественный стиль: {style}
Тематическая категория: {theme}
Эмоциональный тон: {mood}

Напиши одно архивное описание на русском языке, объёмом одно предложение, максимум два.

Жёсткие правила (порядок важности):

1. ФАКТИЧНОСТЬ. Опиши только то, что есть в подписи. НЕ добавляй визуальные детали, которых нет в подписи (не упоминай «трубу» у корабля, «закат» в природной сцене, и т.п.).

2. НИКАКОЙ ЭМОЦИОНАЛЬНОЙ ОКРАСКИ. НИКОГДА не упоминай настроение или эмоциональный тон: «праздничный», «ностальгический», «торжественный», «романтический», «серьёзный», «спокойный». Это субъективные оценки, неприемлемые для архивного каталога. Полностью игнорируй поле «эмоциональный тон».

3. КУЛЬТУРНО-ИСТОРИЧЕСКИЙ КОНТЕКСТ — ТОЛЬКО КОНКРЕТНЫЙ.
   - Допустимо: «Рождество», «Пасха», «Новый год» — если в подписи буквально есть «christmas», «easter», «new year», «holly leaves», «easter eggs», «christmas tree» и т.п.
   - Допустимо: «военная сцена» — если в подписи буквально есть «soldiers», «ship», «battle», «military».
   - Допустимо: «городская сцена» — если в подписи есть «bridge», «buildings», «street», конкретные урбанистические объекты.
   - ЗАПРЕЩЕНО: писать «религиозная сцена», «романтическая сцена», «детская сцена» — это категории классификатора, а не реальный контекст. Игнорируй такие лейблы из поля «тематическая категория», если они не подкреплены сюжетом в подписи.

4. НЕ ИСПОЛЬЗУЙ РАЗМЫТЫЕ ЯРЛЫКИ СТИЛЯ. Слова «винтажный», «винтажная иллюстрация», «декоративный», «ретро» — пустые. Если уверенный стиль = vintage / decorative / retro, опусти упоминание стиля.

5. НИКАКИХ ОЦЕНОЧНЫХ СЛОВ: «прекрасный», «великолепный», «милый», «чудесный», «изящный».

6. НИКАКИХ ФИЛЛЕРОВ: «на этой открытке», «мы видим», «изображение показывает», «отражает», «создаёт настроение».

7. НЕ ПЕРЕЧИСЛЯЙ КАТЕГОРИИ ДОСЛОВНО. Никогда не пиши «тематическая категория — X», «эмоциональный тон — Y». Это технические поля, не часть описания.

8. ИСПРАВЛЯЙ ОШИБКИ ПЕРЕВОДА. Если в английской подписи имя собственное вместо ботанического термина («Holly» вместо «holly»), используй правильный термин («остролист», «падуб»).

9. СТРУКТУРА: одно предложение, начинающееся с типа материала. Минимум, что должно быть в выходе — тип + сюжет.

Несколько примеров для образца:
{few_shot}

Теперь твой ход. Ответь только текстом описания, без префиксов и комментариев.
Описание:"""


def _build_user_prompt(caption_en: str, image_type: str, style: str, theme: str, mood: str) -> str:
    return USER_PROMPT_TEMPLATE.format(
        caption_en=caption_en,
        image_type=image_type,
        style=style,
        theme=theme,
        mood=mood,
        few_shot=FEW_SHOT_EXAMPLES.strip(),
    )


def _format_field(field: Dict, fallback: str = "—") -> str:
    """Format SigLIP-field dict into a human-readable hint for the prompt.

    Включает явный confidence-маркер так, чтобы LLM мог легко его
    распознать через few-shot pattern (см. FEW_SHOT_EXAMPLES).
    """
    if not field:
        return fallback
    label = field.get("label")
    if not label:
        return fallback
    if field.get("confident", False):
        return f"{label} (уверенно)"
    return f"{label} (не уверен)"


class LLMRewriter:
    def __init__(
        self,
        model_path: str = DEFAULT_MODEL,
        *,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
    ):
        self.model_path = model_path
        self.device = get_device()
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        print(f"[LLMRewriter] Loading {model_path} on {self.device}...")
        dtype = torch.float16 if self.device != "cpu" else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
        ).to(self.device)
        self.model.eval()
        print(f"[LLMRewriter] Loaded.")

    def rewrite(
        self,
        caption_en: str,
        metadata: Dict,
        inference: Optional[Dict] = None,
    ) -> str:
        inference = inference or {}

        # Build hints from SigLIP fields. Theme/mood могут не иметь
        # `confident` ключа в inference dict — берём из metadata, если есть.
        image_type_hint = _format_field(metadata.get("image_type", {}))
        style_hint = _format_field(metadata.get("style", {}))
        theme_hint = _format_field(metadata.get("theme") or {})
        mood_hint = _format_field(metadata.get("mood") or {})

        user_prompt = _build_user_prompt(
            caption_en=caption_en,
            image_type=image_type_hint,
            style=style_hint,
            theme=theme_hint,
            mood=mood_hint,
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        prompt_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.temperature > 0,
                temperature=self.temperature if self.temperature > 0 else 1.0,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Strip input tokens
        new_tokens = output[0][inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return text
