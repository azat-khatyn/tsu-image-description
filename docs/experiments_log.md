# Experiments log

Tracker для всех Implementation tasks (I-x) и Experiments (E-x) из [docs/experiments.md](experiments.md).

Формат записи: `[STATUS]` + дата + краткое описание + ссылки на артефакты.
Статусы: `TODO` / `IN PROGRESS` / `DONE` / `BLOCKED` / `SKIPPED`.

---

## Implementation tasks

### I-1: Multilingual CLIP scorer integration

- **Статус:** DONE (2026-05-10)
- **Цель:** интегрировать multilingual CLIP scorer (`M-CLIP/XLM-Roberta-Large-Vit-B-32`) в `scripts/evaluate.py`. Image encoder — open_clip ViT-B-32 OpenAI (общий с M-CLIP).
- **Реализация:** прямая загрузка весов через `huggingface_hub` (минуя пакет `multilingual-clip`, который не работает с новыми transformers — конфликт meta-device init и nested `from_pretrained`). Custom encoder в `scripts/evaluate.py:MCLIPTextEncoder`.
- **Артефакт:** `scripts/evaluate.py`, `data/eval/results/final/metrics_triad_baseline.json`.

### I-2: Switch canonical metric to archive_description_ru

- **Статус:** DONE (2026-05-10)
- **Цель:** primary metric отчёта — `CLIPScore_RU(image, archive_description_ru)`. CLIPScore_EN(caption_en) сохранён для back-compat.
- **Артефакт:** обновлённый `evaluate.py`, output JSON помечает primary_metric.

### I-3: НЭБ pool scraping

- **Статус:** TODO
- **Цель:** расширить eval pool до 200+ изображений из НЭБ-каталога РГБ. Без ручных аннотаций.
- **Артефакт:** `scripts/scrape_neb.py`, `data/eval/pool/`, манифест.

### I-4: Bootstrap CI / paired-test infrastructure

- **Статус:** DONE (2026-05-10) — реализация написана; нужен прогон против пары конфигов после первого E.
- **Артефакт:** `scripts/eval_stats.py` (`bootstrap_ci`, `paired_bootstrap` + CLI).

### I-5: Retrieval R@1, R@5

- **Статус:** DONE (2026-05-10) — реализовано в `evaluate.py:compute_retrieval`, считает обе направленности (i2t, t2i).
- **Замечание:** на n=20 R@k высокие (0.7–1.0), потому что пул маленький. Реальная значимость метрики проявится после I-3 (200+).

### I-6: LLM-as-judge protocol

- **Статус:** IN PROGRESS (2026-05-10) — script + rubric написаны, не прогнаны (нужен `ANTHROPIC_API_KEY` и pilot).
- **Модель по умолчанию:** Claude Sonnet 4.6.
- **Артефакт:** `scripts/llm_judge.py`, `docs/llm_judge_rubric.md` (черновик; freeze после pilot).

### I-7: Cross-metric correlation analysis

- **Статус:** TODO
- **Зависимости:** I-1, I-5, I-6 + первые E-результаты.
- **Артефакт:** `scripts/eval_correlate.py`.

### I-8: Pairwise expert study (опционально)

- **Статус:** TODO (зависит от доступа к эксперту).
- **Артефакт:** `scripts/expert_study.py`.

---

## Experiments

| ID | Block | Эксперимент | Статус | CLIPScore_RU(archive) | R@1 | R@5 | LLM-judge | Latency | Дата |
|---|---|---|---|---|---|---|---|---|---|
**Все запуски на n=220 пуле (20 РГБ + 200 NYPL).** Pool: `data/eval/pool_nypl.jsonl`. CLIPScore_RU(archive) через `M-CLIP/XLM-Roberta-Large-Vit-B-32`. Retrieval = text→image; случайный baseline R@1 = 1/220 ≈ 0.0045.

**Factorial 2 backbone × 4 template_mode = 8 конфигов:**

| Backbone | template_mode | CLIPScore_RU(archive) | R@1 | R@5 | R@10 | p vs base |
|---|---|---|---|---|---|---|
| BLIP-base  | full                  | **0.2804** [0.276, 0.284] | 0.159 | 0.377 | 0.464 | — |
| BLIP-base  | full -theme -mood (E-16) | 0.2865 [0.282, 0.291] | 0.141 | 0.405 | 0.536 | <0.001 |
| BLIP-base  | minimal (E-16b)       | 0.2663 [0.262, 0.270]     | 0.141 | 0.427 | 0.577 | <0.001 |
| BLIP-base  | caption_only (E-16c)  | 0.2540 [0.250, 0.258]     | 0.145 | 0.423 | 0.573 | <0.001 |
| BLIP-large | full (E-1)            | 0.2883 [0.283, 0.294]     | 0.214 | 0.418 | 0.491 | <0.001 |
| BLIP-large | full -theme -mood (E-1+E-16) | **0.2977** [0.293, 0.303] | 0.241 | 0.477 | 0.605 | <0.001 |
| BLIP-large | minimal (E-1+E-16b)   | 0.2848 [0.280, 0.290]     | 0.264 | 0.500 | 0.600 | <0.001 |
| BLIP-large | caption_only (E-1+E-16c) | 0.2735 [0.268, 0.279]  | **0.268** | **0.505** | **0.632** | <0.001 |

| Эксперимент по плану | Статус |
|---|---|
| baseline / E-1 / E-16 / E-1+E-16 (full template) | DONE |
| E-16b / E-16c / E-1+E-16b / E-1+E-16c (template_mode варианты) | DONE |
| E-5+6 (decoding × prefix grid) | TODO |
| E-7 (clean NYPL FT), E-8 (LoRA) | TODO (нужен GPU) |
| E-9 (MarianMT → NLLB-200) | TODO |
| E-11 (Qwen2-VL-2B end-to-end RU) | TODO |
| E-13 (SigLIP threshold calibration) | TODO |
| E-18 (sampling + CLIP rerank) | TODO |

**Подмножества внутри n=220 (CLIPScore_RU archive):**

| Run | rgb (n=20, anchor) | nypl (n=200, pool) | all (n=220) |
|---|---|---|---|
| baseline | 0.3254 | 0.2759 | 0.2804 |
| E-1 | 0.3313 | 0.2840 | 0.2883 |
| E-16 | 0.3262 | 0.2825 | 0.2865 |
| **E-1+E-16** | **0.3341** | **0.2940** | **0.2977** |

---

## Заметки по запускам

### 2026-05-11 (v3) — Template-mode factorial: главная находка о trade-off между faithfulness и searchability

После того как первая n=220 волна показала, что **E-16 (drop theme+mood) вредит retrieval** при n=220 (хотя CLIPScore рос), решил проверить более радикальные варианты упрощения шаблона. Добавил параметр `template_mode` в `DescriptionBuilder`:

- **full** (исходный): `<тип> в стиле <стиль>. На изображении: <caption>. Предположительно, это <тема>. Общее настроение <настроение>.`
- **full -theme -mood** (E-16, уже было): `<тип> в стиле <стиль>. На изображении: <caption>.`
- **minimal** (E-16b, новый): `На изображении: <caption>.`
- **caption_only** (E-16c, новый): `<caption>` — без шаблона вообще

Прогнал все 4 варианта на обоих backbone'ах (BLIP-base, BLIP-large). **Главная находка** — обратная корреляция между CLIPScore и retrieval при упрощении шаблона.

**Числа на BLIP-large (best backbone), n=220:**

| Template | CLIPScore | R@1 | R@5 | R@10 |
|---|---|---|---|---|
| full | 0.288 | 0.214 | 0.418 | 0.491 |
| full -t -m | **0.298** ← max | 0.241 | 0.477 | 0.605 |
| minimal | 0.285 | 0.264 | 0.500 | 0.600 |
| caption_only | 0.274 | **0.268** | **0.505** | **0.632** ← max |

**Объяснение trade-off'а.** Метрики оптимизируют разные вещи:

- **CLIPScore_RU(archive)** — это **средняя alignment** между текстом и картинкой. Шаблонные фразы вроде "архивная открытка", "винтажная иллюстрация" дают альтернативный сигнал, который "тянет" CLIPScore вверх, даже если конкретно captioning неточный.
- **Retrieval R@k** — это **relative ranking**. Шаблон делает все описания похожими в эмбеддинг-пространстве (одна и та же словесная "оправка" перед каждым caption'ом → сильный общий компонент в text-embedding'е) → разные открытки становятся хуже различимыми → ниже R@k.

То есть **шаблон одновременно помогает faithfulness и вредит searchability**. Для use case архивного поиска searchability — главная цель. **Для тезиса это означает: оптимальный final-пайплайн — `caption_only`, а не "richest" шаблон.**

**Доменное consistency:** trade-off одинаков на rgb anchor (n=20) и nypl pool (n=200) — это не артефакт NYPL-домена, это структурное свойство нашей шаблонной обвязки.

**Что в итоге включает best-pipeline:**
- BLIP-large для captioning
- `caption_only` для DescriptionBuilder (только перевод caption на русский)
- search_text-tags сохраняем отдельно (они не влияют на archive_description, но дают доп. сигнал для full-text-search в каталоге)

**Импликации для плана:**
- E-16 (наш изначальный "drop low-confidence sentences" эксперимент) **переосмыслен**: проблема не в "wrong sentences", а в **самом шаблоне как репрезентации**. Лучший fix — убрать шаблон, не калибровать его.
- E-12 (LLM-based DescriptionBuilder) теперь смотрится **по-другому**: он может ввести **разнообразие фразирования** между описаниями, что разрешает trade-off (фразы разные → embedding'и разные → retrieval не страдает; одновременно текст содержательный → CLIPScore не падает). Это сильный кандидат **на следующий приоритет**.
- E-13 (SigLIP calibration) становится **менее критичным** в свете caption_only winning — если template убираем, то и пороги калибровать незачем (только для search_text-tags).

**Best system for thesis: BLIP-large + caption_only template** на retrieval; **BLIP-large + full-theme-mood** на CLIPScore. Имеет смысл показать ОБА в работе как "two different optimization targets" с явным обсуждением trade-off'а.

### 2026-05-11 (v2) — Расширение пула до n=220 (20 РГБ + 200 NYPL), все эксперименты значимы

После того как первая волна (n=20) показала, что все Δ статистически неразличимы (p ≥ 0.25), расширил retrieval pool через семплирование 200 NYPL-открыток. Скрипт `scripts/sample_eval_pool.py` исключает изображения, присутствующие в `data/nypl/splits/*.json` (data leakage prevention для будущих E-7 / E-8). Сид = 42, файл — `data/eval/pool_nypl.jsonl`. Все 4 эксперимента перезапущены через `--pool` флаг.

**Главные результаты на n=220 (вся таблица выше):**

1. **Все три эксперимента значимы (p < 0.001)** по CLIPScore_RU(archive), в отличие от n=20. Это полностью оправдывает расширение пула.

2. **E-1+E-16 (combined) — явный winner** по всем метрикам:
   - CLIPScore_RU(archive): 0.2977 (vs baseline 0.2804, +0.017)
   - Retrieval t2i R@1: 0.241 (vs 0.159, +0.082) — **на 51% выше относительно baseline**
   - Latency: 0.89 s/img (vs 0.72) — приемлемо.

3. **E-16 (drop theme+mood) сам по себе ВРЕДИТ retrieval на большом пуле** (R@1: 0.141 < 0.159 baseline), хотя CLIPScore aggregate растёт.
   - Это **обратная картина** от n=20, где E-16 показывал R@1 = 0.75 (выше baseline 0.70).
   - Объяснение: на n=20 пул мал, удаление theme/mood ещё не лишает описание discriminative power. На n=220 короткое описание ("Открытка в стиле X. На изображении: Y.") уже не различимо среди 219 дистракторов.
   - **Методологический вывод:** retrieval на маленьком пуле обманывает. Без n ≥ 100 R@k нельзя серьёзно интерпретировать.

4. **Эффекты E-1 и E-16 аддитивны на CLIPScore (+0.008 + +0.006 ≈ +0.017)**, но **не аддитивны на retrieval**: E-16 alone -0.018, E-1 alone +0.055, combined +0.082 → combined даже лучше суммы. Интерпретация:
   - С BLIP-base короткими captions theme/mood часто несёт хоть какой-то discriminative сигнал даже когда категория неправильная — "праздничная сцена" привязывает текст к чему-то, что баланс рассрочка не выдерживает на больших пулах.
   - С BLIP-large detailed captions theme/mood становится redundant noise, и его удаление чистит alignment.
   - **Это качественный thesis-tier результат** о взаимодействии компонентов пайплайна, который без n=220 + полного A/B-факториала был бы скрыт.

5. **Per-source consistency:** тренд rgb (n=20 anchor) vs nypl (n=200) идёт в одну сторону на каждом эксперименте, что подтверждает переносимость улучшений между доменами. Абсолютные числа на nypl ниже (~0.276–0.294 vs rgb ~0.325–0.334) — NYPL содержит более разнородный материал (trade cards, advertisements), не только открытки.

**Qualitative example (biggest win, Δ = +0.136):**

- Baseline: "Книгу с красной обложкой и черным титулом..." (BLIP-base hallucinated wrong content)
- E-1+E-16: "На ней есть книга с постером турнира для гольфа." (BLIP-large saw the actual content correctly)

**Импликации для плана:**
- Прежний "drop low-confidence sentences" вывод (E-16 как простой fix) требует пересмотра. **E-13 (калибровка порогов) и E-1 как фундаментальный upgrade — теперь центральные**.
- Lead-пайплайн для всех downstream экспериментов: **E-1+E-16 (BLIP-large + simple template)**, не baseline.
- Следующие приоритеты: E-13 (калибровка), E-9 (NLLB-200 для перевода), E-11 (Qwen2-VL для архитектурного сравнения).

### 2026-05-11 (v1) — Первая волна экспериментов после структурной чистки

После cleanup (новая структура `data/eval/images/`, `data/eval/results/final/`) перезапустил baseline и провёл 3 эксперимента. Все результаты — в `data/eval/results/final/`, графики и таблицы — в `notebooks/experiments_results.ipynb`.

**Сводка по триаде (n=20):**

| Run | CLIPScore_RU(archive) | CI95 | R@1 (t2i) | R@5 (t2i) | Δ vs baseline | p-value |
|---|---|---|---|---|---|---|
| baseline | 0.3254 | [0.311, 0.341] | 0.70 | 0.90 | — | — |
| E-1 BLIP-large | 0.3313 | [0.314, 0.349] | 0.80 | 0.95 | +0.006 | 0.37 |
| E-16 drop theme+mood | 0.3262 | [0.309, 0.344] | 0.75 | 0.95 | +0.001 | 0.88 |
| **E-1 + E-16** | **0.3341** | [0.317, 0.351] | **0.90** | 0.95 | +0.009 | 0.25 |

**Главные наблюдения:**

1. **CLIPScore deltas не значимы при n=20** (paired bootstrap p ≥ 0.25 для всех). CIs перекрываются. Это **не сюрприз** — 20 точек дают недостаточную статистическую мощность для различий на третьем знаке. Это ещё раз подчёркивает зависимость от I-3 (расширение пула до 200+).

2. **Retrieval показывает большие, легко интерпретируемые сдвиги:**
   - baseline: 14 из 20 правильных top-1 (t2i)
   - E-1: 16 из 20 (+2)
   - E-16: 15 из 20 (+1)
   - **E-1+E-16: 18 из 20** (+4)

3. **BLIP-large даёт значительно более информативные описания** (см. постер 3 в notebook'е: "Вид на реку с мостом и зданиями" vs "черно-белое фото реки"). Это объясняет retrieval-прирост и без CLIPScore-улучшения.

4. **Latency BLIP-large** ≈ 1.0 s/img vs 0.71 для base (+40%). На MPS ещё переживаемо; на GPU не имеет значения.

5. **Эффекты E-1 и E-16 аддитивны на retrieval** (R@1 t2i: +0.10 от E-1 + +0.05 от E-16 = +0.20 в комбинации, против факта +0.20 — почти точно). Это полезный качественный вывод о независимости вкладов разных частей пайплайна.

**Импликации для плана:**
- Lead по retrieval: E-1+E-16 — текущий "best" пайплайн. Все downstream эксперименты (E-7, E-8, E-11) теперь сравниваются с этой комбинацией.
- На CLIPScore тренд (E-1 > E-16 ≈ baseline) сохраняется, но без значимости. Ожидаем proof после I-3.
- Стоит сначала сделать I-3 (НЭБ scraping → 200+ pool), потом продолжать E.


### 2026-05-10 — I-1 + I-2 + I-4: первый baseline на триаде (partial)

Прогон `scripts/evaluate.py` на текущих 20 открытках:

```
CLIPScore_EN_caption_en       : 0.298
CLIPScore_RU_caption_ru       : 0.301
CLIPScore_RU_archive_ru       : 0.325   ← primary
CLIPScore_RU_reference_ru     : 0.309
Latency mean                  : 0.70 s
```

Результат: см. `data/eval/results/final/metrics_triad_baseline.json`.

**Главное наблюдение:** под M-CLIP archive_description (0.325) выше reference (0.309) и выше caption_ru (0.301). Под старым EN-CLIP archive был ≈ 0.20 и казался ниже всех. Метрика была неправильной.

**Импликации для плана:**

- E-16 (drop low-confidence sentences) сохраняется, но ожидаемая магнитуда меньше.
- Часть экспериментов (E-7, E-8) теперь нужно оценивать по **другому** baseline. Каждый E должен сравниваться с triad-baseline 0.325, не с EN 0.306.
- §3 диагностики обновлён с разделением D1–D4 (sustained / revised / new).

**Технические заметки:**

- Пакет `multilingual-clip` (PyPI) не работает с современным `transformers` (≥ 4.40, meta-device init): nested `from_pretrained` падает с `RuntimeError: meta device context manager`. Bypass — прямая загрузка весов через `huggingface_hub`. См. `MCLIPTextEncoder` в `scripts/evaluate.py`.
- Конфиг M-CLIP содержит `model_type: "M-CLIP"` неизвестный AutoConfig — читаем JSON напрямую.
- Загрузка XLM-R Large + projection: 19.6 s. Первый encode на MPS: 2.1 s (MPS JIT). Дальше — миллисекунды.

### 2026-05-10 — I-5 retrieval + E-16 (blunt drop theme/mood)

Реализован retrieval R@k в `scripts/evaluate.py:compute_retrieval`. Запущен E-16 (drop theme + mood без условий) против baseline.

**Aggregate (n=20):**

| Metric | baseline | E-16 | Δ | paired bootstrap p |
|---|---|---|---|---|
| CLIPScore_RU(archive) | 0.3254 | 0.3262 | +0.0008 | 0.88 (not significant) |
| Retrieval i2t R@1 | 0.75 | 0.80 | +0.05 | — (1 image flip) |
| Retrieval t2i R@1 | 0.70 | 0.75 | +0.05 | — (1 image flip) |

**Per-image (важно):** агрегаты скрывают сильные сдвиги в обе стороны:

- **Большие положительные сдвиги (E-16 помогает):** postcard_7 (+0.052, Розы), postcard_12 (+0.023, Девушка со свечой), postcard_17 (+0.022, Пагода), postcard_20 (+0.022, Девочка с цветами), postcard_14 (+0.018, Целующиеся). Здесь theme/mood были ошибочны — снятие сценарных утверждений помогло.
- **Большие отрицательные сдвиги (E-16 вредит):** postcard_13 (-0.039, Зимний лес → "праздничная сцена" — реально может быть Рождество), postcard_10 (-0.033, Цветы и девочка), postcard_19 (-0.028, Сани), postcard_3 (-0.024, Галич, Рождественская церковь). Здесь theme/mood были корректны — снятие убрало релевантную информацию.

**Вывод и перепланирование:**

- Безусловное снятие theme/mood (blunt E-16) — wash на агрегате. Per-image эффекты симметричны: помогает там, где был mistake; вредит там, где был hit.
- **E-13 (калибровка порогов SigLIP)** — теперь явно центральный в блоке F, а не E-16. Цель: повысить confidence-threshold так, чтобы wrong-but-confident случаи отсекались, а right-and-confident проходили.
- **Новый вариант E-16b:** более тонкий — повышение confidence-threshold для theme/mood с 0.18 до, скажем, 0.30 или 0.40. По сути это тот же эксперимент, что E-13 с условием на confidence-вместо-F1. Стоит запустить параллельно с E-13.

### TODO следующее

- **I-3:** реализовать `scripts/scrape_neb.py` для расширения eval pool до 200+. Без I-3 (n=20) bootstrap CI всё равно широкий — реальные сравнения экспериментов будут размытыми.
- **I-5:** добавить retrieval R@k в evaluate.py (нужен I-3 для осмысленного пула).
- **I-6 pilot:** прогнать LLM-judge на baseline 20 (требует `ANTHROPIC_API_KEY` от пользователя).
- После всех I — refresh baseline и запуск первой волны cheap E (E-16 + E-5+6 + E-13).
