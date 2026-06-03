"""llm_rewriter.py — языковой редактор архивного описания на локальной LLM.

Берёт caption_en (от BLIP) и структурированные метаданные (от SigLIP) и
напрямую генерирует русское архивное описание, минуя дословный машинный перевод.

Решает:
  - многозначность MarianMT (painting → картина, не покраска)
  - транслитерацию имён собственных (holly → падуб)
  - грамматику и естественность речи
  - повтор шаблонной фразы по корпусу

Сохраняет принцип закрытых таксономий: метки SigLIP передаются в prompt как
структурированный контекст и используются LLM для архивной стандартизации.

Модель по умолчанию — Vikhr-Nemo-12B-Instruct-R (русский нативно, fp16).
Меняется через `model_path`.
"""

from typing import Dict, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .models import get_device


DEFAULT_MODEL = "Vikhrmodels/Vikhr-Nemo-12B-Instruct-R-21-09-24"

# Списки generic-стилей и стандартных категорий, чтобы не дублировать пояснения в prompt.
GENERIC_STYLES_RU = {
    "vintage illustration", "decorative illustration", "retro design",
}


# ===========================================================================
# Версионирование prompt-стилей
# ===========================================================================
# v1_archival — исходная схема. Описание начинается с типа материала:
#               «Открытка-хромолитография. …». Сохраняется для
#               воспроизводимости прежних прогонов.
#
# v2_curator  — настройка под curator-описания РГБ. Текст начинается сразу с
#               визуального содержания, без преамбулы; тип материала и техника
#               не дублируются (хранятся в полях RUSMARC 200, 215). Имитирует
#               стиль поля 327 НЭБ: компактные именные и причастные обороты.
# ===========================================================================

# ----- v1_archival (исходная схема) -----------------------------------------

SYSTEM_PROMPT_V1_ARCHIVAL = (
    "Ты — каталогизатор Российской государственной библиотеки. "
    "Ты пишешь описания открыток для электронного архивного каталога "
    "в нейтральном, плотном, фактическом стиле — без оценочной лексики "
    "и литературных украшений."
)

FEW_SHOT_EXAMPLES_V1_ARCHIVAL = """
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

USER_PROMPT_TEMPLATE_V1_ARCHIVAL = """Тебе даны:
Английская подпись от vision-модели: "{caption_en}"
Тип материала: {image_type}
Художественный стиль: {style}
Тематическая категория: {theme}
Эмоциональный тон: {mood}
Надпись на изображении (OCR): {ocr}

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

10. НАДПИСЬ НА ИЗОБРАЖЕНИИ (OCR). Если поле «Надпись на изображении» непустое и содержит топоним или название (например, «Видъ на прудъ, Галичъ»), включи его дословно в описание. Если поле пустое («—») или содержит нечитаемый набор символов — полностью игнорируй его. НИКОГДА не выдумывай надпись, которой нет в этом поле.

Несколько примеров для образца:
{few_shot}

Теперь твой ход. Ответь только текстом описания, без префиксов и комментариев.
Описание:"""


# ----- v2_curator (под поле 327 НЭБ) ----------------------------------------

SYSTEM_PROMPT_V2_CURATOR = (
    "Ты — каталогизатор Российской государственной библиотеки. "
    "Ты заполняешь поле «Примечание содержания» (RUSMARC 327) в электронном "
    "архивном каталоге открыток. Описание начинается сразу с визуального "
    "содержания — без указания типа материала, техники или стиля "
    "(эти данные хранятся в отдельных полях RUSMARC 200, 215). "
    "Стиль — нейтральный, фактический, компактный, без оценочной лексики и "
    "литературных украшений. Используй именные и причастные обороты "
    "(«Вид на X», «Y, делающий Z», «Сцена с N людьми»)."
)

FEW_SHOT_EXAMPLES_V2_CURATOR = """
=== Пример 1 ===
Английская подпись: "three children riding a sled in snowy forest"
Тип материала: a postcard (уверенно)
Художественный стиль: vintage illustration (уверенно)
Тематическая категория: holiday scene (не уверен)
Эмоциональный тон: festive (не уверен)
Описание: Дети, едущие на санях по зимнему лесу.

=== Пример 2 ===
Английская подпись: "ship in the water with a lot of smoke coming out of it"
Тип материала: a postcard (уверенно)
Художественный стиль: engraving (уверенно)
Тематическая категория: military subject (не уверен)
Эмоциональный тон: nostalgic (уверенно)
Описание: Корабль в море, из которого валит густой дым.

=== Пример 3 ===
Английская подпись: "a christmas card with a box and holly leaves"
Тип материала: a greeting card (уверенно)
Художественный стиль: vintage illustration (уверенно)
Тематическая категория: Christmas holiday scene (уверенно)
Эмоциональный тон: festive (уверенно)
Описание: Подарочная коробка с ветвями остролиста; рождественская тематика.

=== Пример 4 ===
Английская подпись: "a man with a turban and long beard"
Тип материала: a photograph (уверенно)
Художественный стиль: color photograph (не уверен)
Тематическая категория: portrait (не уверен)
Эмоциональный тон: serious (уверенно)
Описание: Мужчина с длинной бородой в тюрбане; погрудный портрет.

=== Пример 5 ===
Английская подпись: "vintage postcard with a wave crashing on stones at the beach"
Тип материала: a postcard (уверенно)
Художественный стиль: color photograph (уверенно)
Тематическая категория: nature scene (не уверен)
Эмоциональный тон: nostalgic (не уверен)
Описание: Волна, разбивающаяся о прибрежные камни.
"""

USER_PROMPT_TEMPLATE_V2_CURATOR = """Тебе даны:
Английская подпись от vision-модели: "{caption_en}"
Тип материала: {image_type}
Художественный стиль: {style}
Тематическая категория: {theme}
Эмоциональный тон: {mood}
Надпись на изображении (OCR): {ocr}

Напиши описание для поля «Примечание содержания» (RUSMARC 327), одно или два кратких предложения.

Жёсткие правила (порядок важности):

1. ФАКТИЧНОСТЬ. Опиши только то, что есть в подписи. НЕ добавляй визуальные детали, которых нет в подписи.

2. БЕЗ ПРЕАМБУЛЫ С ТИПОМ МАТЕРИАЛА. ЗАПРЕЩЕНО начинать описание с «Открытка.», «Открытка-хромолитография.», «Поздравительная открытка.», «Фотография.», «Гравюра.». Эти данные хранятся в полях RUSMARC 200 и 215, в поле 327 они не дублируются. Начинай сразу с visual content.

3. ИМЕННЫЕ И ПРИЧАСТНЫЕ КОНСТРУКЦИИ. Предпочитай: «Вид на X в Y», «Корабль, идущий через шторм», «Сцена с тремя солдатами на привале», «Подарочная коробка с ветвями остролиста». Избегай громоздких глагольных оборотов «На этом изображении мы видим...».

4. НИКАКОЙ ЭМОЦИОНАЛЬНОЙ ОКРАСКИ. НИКОГДА не упоминай настроение: «праздничный», «ностальгический», «торжественный», «романтический». Полностью игнорируй поле «эмоциональный тон».

5. КУЛЬТУРНО-ИСТОРИЧЕСКИЙ КОНТЕКСТ — ТОЛЬКО КОНКРЕТНЫЙ.
   - Допустимо: «Рождество», «Пасха», «Новый год» — если в подписи буквально есть «christmas», «easter», «new year», «holly leaves», «easter eggs», «christmas tree».
   - Допустимо: «военная сцена», «партизаны», «солдаты» — если в подписи есть «soldiers», «ship», «battle», «military».
   - ЗАПРЕЩЕНО: писать «религиозная сцена», «детская сцена» — это категории классификатора, не реальный сюжет.

6. НЕ ИСПОЛЬЗУЙ РАЗМЫТЫЕ ЯРЛЫКИ СТИЛЯ. Слова «винтажный», «декоративный», «ретро» — пустые. Стиль и техника указаны в полях 215, не в описании.

7. НИКАКИХ ОЦЕНОЧНЫХ СЛОВ: «прекрасный», «великолепный», «милый», «чудесный».

8. НИКАКИХ ФИЛЛЕРОВ: «на этой открытке», «мы видим», «изображение показывает», «отражает», «создаёт настроение».

9. НЕ ПЕРЕЧИСЛЯЙ КАТЕГОРИИ ДОСЛОВНО. Никогда не пиши «тематическая категория — X», «эмоциональный тон — Y».

10. ИСПРАВЛЯЙ ОШИБКИ ПЕРЕВОДА. Если в подписи «Holly» (имя) вместо «holly» (растение) — используй «остролист»/«падуб».

11. НАДПИСЬ НА ИЗОБРАЖЕНИИ (OCR). Если поле «Надпись на изображении» непустое и содержит топоним или название, включи его дословно в описание. Если поле пустое («—») или содержит нечитаемый набор символов — полностью игнорируй его. НИКОГДА не выдумывай надпись, которой нет в этом поле.

Несколько примеров для образца:
{few_shot}

Теперь твой ход. Ответь только текстом описания, без префиксов и комментариев.
Описание:"""


# Реестр доступных стилей
PROMPT_STYLES = {
    "v1_archival": {
        "system": SYSTEM_PROMPT_V1_ARCHIVAL,
        "few_shot": FEW_SHOT_EXAMPLES_V1_ARCHIVAL,
        "user_template": USER_PROMPT_TEMPLATE_V1_ARCHIVAL,
        "max_new_tokens": 128,
    },
    "v2_curator": {
        "system": SYSTEM_PROMPT_V2_CURATOR,
        "few_shot": FEW_SHOT_EXAMPLES_V2_CURATOR,
        "user_template": USER_PROMPT_TEMPLATE_V2_CURATOR,
        "max_new_tokens": 192,
    },
}


def _build_user_prompt(style_config: Dict, caption_en: str,
                       image_type: str, style: str, theme: str, mood: str,
                       ocr: str = "—") -> str:
    return style_config["user_template"].format(
        caption_en=caption_en,
        image_type=image_type,
        style=style,
        theme=theme,
        mood=mood,
        ocr=ocr,
        few_shot=style_config["few_shot"].strip(),
    )


def _format_field(field: Dict, fallback: str = "—") -> str:
    """Форматирует поле SigLIP в человекочитаемую подсказку для prompt.

    Добавляет явный маркер уверенности, чтобы LLM распознавал его через
    few-shot примеры (см. FEW_SHOT_EXAMPLES).
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
    """Языковой редактор архивного описания на локальной LLM.

    Args:
        model_path: идентификатор модели HF (по умолчанию Vikhr-Nemo-12B-Instruct-R).
        prompt_style: одно из {"v1_archival", "v2_curator"}. v1_archival —
            исходный prompt с преамбулой «Открытка.»; v2_curator — стиль под
            поле 327 НЭБ, без преамбулы.
        max_new_tokens: переопределяет длину генерации. Если None — берётся
            значение по умолчанию выбранного prompt_style.
    """

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL,
        *,
        prompt_style: str = "v1_archival",
        max_new_tokens: Optional[int] = None,
        temperature: float = 0.0,
    ):
        if prompt_style not in PROMPT_STYLES:
            raise ValueError(
                f"Unknown prompt_style={prompt_style!r}. "
                f"Available: {sorted(PROMPT_STYLES.keys())}"
            )
        self.model_path = model_path
        self.prompt_style = prompt_style
        self.style_config = PROMPT_STYLES[prompt_style]
        self.device = get_device()
        self.max_new_tokens = max_new_tokens or self.style_config["max_new_tokens"]
        self.temperature = temperature

        print(f"[LLMRewriter] Loading {model_path} on {self.device} "
              f"(prompt_style={prompt_style}, max_new_tokens={self.max_new_tokens})...")
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
        ocr_text: Optional[str] = None,
    ) -> str:
        inference = inference or {}

        # Подсказки из полей SigLIP. У theme/mood может не быть ключа
        # `confident` в inference — берём из metadata, если есть.
        image_type_hint = _format_field(metadata.get("image_type", {}))
        style_hint = _format_field(metadata.get("style", {}))
        theme_hint = _format_field(metadata.get("theme") or {})
        mood_hint = _format_field(metadata.get("mood") or {})

        # OCR-подсказка: пайплайн передаёт только уверенный текст; пустую
        # надпись показываем как «—», промпт-правило велит её игнорировать.
        ocr_hint = ocr_text.strip() if ocr_text and ocr_text.strip() else "—"

        user_prompt = _build_user_prompt(
            self.style_config,
            caption_en=caption_en,
            image_type=image_type_hint,
            style=style_hint,
            theme=theme_hint,
            mood=mood_hint,
            ocr=ocr_hint,
        )

        messages = [
            {"role": "system", "content": self.style_config["system"]},
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

        # отбрасываем токены входного prompt
        new_tokens = output[0][inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return text
