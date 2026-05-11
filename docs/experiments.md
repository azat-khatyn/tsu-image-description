# План экспериментов: улучшение качества описаний открыток

**Версия документа:** 0.3 (impl/experiment split, narrowed E-set)
**Дата:** 2026-05
**Контекст:** ВКР, ТГУ, 2026. Система генерации архивных описаний изображений открыток для электронного каталога библиотеки.

---

## 1. Цель и основная гипотеза

**Контекст.** В проекте отсутствует ground truth: библиотека (РГБ / НЭБ) на момент работы не имеет ни эталонных описаний к существующим открыткам, ни внутренней разметки нужного объёма. Это не "недостаток данных", а структурное свойство задачи — система разрабатывается именно потому, что описаний нет. Поэтому методологическая рамка работы — **evaluation в режиме absence of ground truth**.

**Цель.** Получить пайплайн, генерирующий описания открыток, которые:

1. **Faithful** — точно отражают визуальное содержание изображения (нет галлюцинаций).
2. **Searchable** — позволяют находить нужную открытку поиском по описанию (имитация архивного use case).
3. **Stylistically appropriate** — пригодны для библиотечного каталога по форме и лексике.

**Основная метрика — триада, не одно число:**

- `CLIPScore_RU(image, archive_description_ru)` — faithfulness.
- `Recall@1, Recall@5` retrieval по пулу ≥ 200 открыток — searchability.
- `LLM-as-judge` по фиксированной рубрике — style / completeness.

Ни один из трёх сигналов сам по себе не "правда". Прирост признаётся, когда хотя бы два из трёх растут значимо и ни один не падает.

**Дополнительно (опционально):** валидационное исследование с библиотекарем — pairwise preference на 20–30 примерах. Если удастся организовать — служит внешней валидацией триады. Не выступает gating-метрикой.

**Основная гипотеза.** В текущем пайплайне просадка возникает не на этапе captioning, а на этапах перевода и шаблонной сборки описания. Поэтому наибольший прирост даст переход на multilingual-метрику, калибровка метаданных и переработка DescriptionBuilder, а не дальнейшее наращивание fine-tuning BLIP.

Эта гипотеза — главный объект проверки в работе.

---

## 2. Текущее состояние

### 2.1 Пайплайн (baseline)

```
Image
 ├─► BLIP base ─► caption_en ─► MarianMT ─► caption_ru
 └─► SigLIP zero-shot ─► {image_type, style, theme, mood} ─► ThemeInferencer
                                                              │
                  caption_ru + metadata + theme/mood ─► DescriptionBuilder ─► archive_description_ru
```

### 2.2 Дообучение BLIP (существующее)

| ID | Метод | Данные | Гиперпараметры | Статус |
|---|---|---|---|---|
| `caplift_v1` | Full FT | `train_v1.json` (≈15k пар) | по умолчанию | артефакт `models/blip_nypl_epoch_1` |
| `caplift_v2` | Full FT | `train_v2.json` (≈7.5k пар, capfilt-фильтр) | lr=3e-5, bs=4, 2 epochs | актуальный код в `train_blip_nypl.py` |
| `lora_v3` | LoRA r=8 | v1 | — | артефакт `models/blip_caplift_v3_lora_epoch_1` |
| `lora_v4` | LoRA | v1 | — | артефакт `models/blip_caplift_v4_lora_epoch_4` |
| `lora_v5` | LoRA r=16, α=32, dropout=0.05, target=Q/K/V | v1 | lr=3e-6, bs=4, 7 epochs | актуальный код в `train_blip_nypl_lora.py` |

### 2.3 Подготовка обучающих данных (CapFilt-style)

`generate_captions.py` → `filter_captions.py` → `prepare_splits.py`:

1. BLIP-base self-caption на каждом изображении NYPL (6 645 шт).
2. Фильтр по CLIPScore ≥ 0.3 и длине ≥ 4 слов → 2 152 пары.
3. Hybrid caption: `caption + ". " + nypl_title`.
4. 90/10 train/val split.

### 2.4 Текущие числа на baseline (20 открыток eval set)

После I-1 + I-2 (multilingual CLIP + canonical metric):

| Метрика | EN-CLIP (legacy) | M-CLIP RU (canonical) |
|---|---|---|
| `CLIPScore_EN(caption_en)` | **0.306** | 0.298 |
| `CLIPScore_RU(caption_ru)` | 0.205 | **0.301** |
| `CLIPScore_RU(archive_description_ru)` ← **primary** | 0.203 | **0.325** |
| `CLIPScore_RU(reference_short_ru)` | 0.207 | **0.309** |
| Latency, sec/image | 1.0 | 0.70 |

**Ключевое наблюдение после I-1.** Под EN-CLIP русские тексты стояли вровень с reference-ceiling ≈ 0.21, а archive_description казался хуже caption_ru. Под M-CLIP картина переворачивается: archive_description на 0.325 — выше reference (0.309) и выше caption_ru (0.301). Старый "потолок" был артефактом метрики, не свойством системы.

Это инвалидирует часть диагностики §3 (см. поправки в §3.6).

### 2.5 Существующие анализы

- `notebooks/Pairwise comparison.ipynb` — pairwise similarity (image vs 4 видов текста), top-1 accuracy, margin.
- `notebooks/Embeddings exploration.ipynb`, `Untitled.ipynb` — PCA / t-SNE, heatmap.
- `notebooks/Untitled1.ipynb` — кривые train/val loss.

---

## 3. Диагностика: где теряется CLIPScore

### 3.1 Изначальная диагностика (на EN-CLIP, до I-1)

Из `predictions_detailed.json` (20 открыток, EN-CLIP) виден ряд систематических проблем:

1. **Перевод EN→RU стоит ≈ 0.10 CLIPScore.** Open_clip ViT-B-32 OpenAI обучен на английском; русский текст оценивается ниже структурно, независимо от качества перевода. → **На самом деле артефакт метрики**, см. §3.6.
2. **Шаблонное `archive_description` зачастую хуже простого `caption_ru`.** Шаблон добавляет "Предположительно, это X. Общее настроение Y." даже когда X/Y по факту неверны. → **Под M-CLIP это утверждение неверно** — archive в среднем выше, см. §3.6.
3. **Theme/mood из SigLIP часто ошибается.** Закрытый список кандидатов навязывает выбор; пороги (0.18–0.35) подобраны эмпирически без калибровки. → **Сохраняется** независимо от метрики (видно в качественном анализе).
4. **BLIP-base даёт короткие и общие описания** ("a christmas card with a bird and holly"). → **Сохраняется**.
5. **Reference-описания на русском дают CLIPScore ≈ 0.21.** → **Это был артефакт EN-CLIP**, см. §3.6.

### 3.6 Поправки после I-1

Triad baseline под M-CLIP RU:

- archive (0.325) > reference (0.309) > caption_ru (0.301).
- Старый "потолок" в 0.21 был свойством EN-CLIP, не системы.
- Артефакт "перевод режет 0.10 CLIPScore" исчезает: caption_ru под M-CLIP даёт 0.301 vs caption_en 0.298 — практически вровень.

**Что остаётся как реальная диагностика после I-1:**

- **D1 (sustained):** SigLIP theme/mood часто ошибается на конкретных примерах (postcard_5, 13, 18). Закрытый список + неоткалиброванные пороги. Это не виден на агрегированной метрике, но критичен для отдельных описаний → **E-13 + E-14** остаются актуальными.
- **D2 (sustained):** BLIP-base captions короткие и общие; теряют детали открытки → **E-1 + E-7** остаются актуальными.
- **D3 (revised):** Шаблон в среднем не вредит, но когда low-confidence предложение всё-таки вставляется и оно ошибочно — оно вредит конкретному примеру. **E-16 остаётся, но ожидаемая магнитуда меньше**, чем казалось ранее.
- **D4 (new):** Нужно различать **средний прирост на пуле** и **снижение worst-case ошибок**. Триада (особенно LLM-judge с criterion no-hallucinations) поможет ловить второе.

---

## 4. Принципы оценки

### 4.1 Триада основных метрик

Поскольку ground truth отсутствует, оценка опирается на три независимых reference-free сигнала, измеряющих разные стороны качества.

| Сигнал | Что измеряет | Тип | Реализация |
|---|---|---|---|
| `CLIPScore_RU(image, archive_description_ru)` | Faithfulness — соответствие текста изображению | Reference-free | multilingual CLIP, например `sentence-transformers/clip-ViT-B-32-multilingual-v1` или `M-CLIP/XLM-Roberta-Large-Vit-B-32` |
| `Recall@1, Recall@5` retrieval по пулу ≥ 200 изображений | Searchability — пригодность для архивного поиска | Reference-free | косинусное сходство image-emb ↔ text-emb в multilingual CLIP |
| `LLM-as-judge` по фиксированной рубрике (5 критериев × 1–5 шкала) | Style / completeness / accuracy by-proxy | Reference-free, по протоколу | Claude или GPT-4o с зафиксированным system-prompt и рубрикой |

В отчёте по каждому эксперименту приводятся все три. Прирост по одному сигналу при стагнации/просадке другого — повод для обсуждения, не для отказа от эксперимента.

### 4.2 Дополнительные сигналы

| Метрика | Применение |
|---|---|
| `BLIP-ITM` score | Cross-checking faithfulness независимым семейством моделей |
| `CLIPScore_EN(image, caption_en)` | Совместимость с исходной отчётностью; изоляция вклада перевода |
| Margin (diag − best off-diag) на pairwise CLIP-similarity | Контрастность; дополняет CLIPScore |
| BLEU, METEOR, ROUGE-L против weak references из НЭБ | **Стилевая близость, не "правильность"** |
| Latency, images/sec | Регрессионный контроль |

**Weak references.** Брифовые названия из каталога НЭБ для тех изображений, где они есть, используются как один из стилевых сигналов (лексика, длина, структура). BLEU/METEOR против них **не интерпретируются как точность** — это сравнение с одним из возможных стилей описания, а не с эталоном.

### 4.3 Eval pool

| | Текущее | Цель |
|---|---|---|
| Размер пула для retrieval | 20 | 200–500 |
| Источник | НЭБ РГБ (postcards) | то же; собирается через скрейпинг каталога |
| Аннотации | 20 кратких ref_ru (paraphrases НЭБ) | weak refs только для подмножества (где НЭБ их даёт) |

Пул нужен в первую очередь для retrieval (R@k); для CLIPScore_RU и LLM-judge дополнительные аннотации не требуются. Это снимает основной bottleneck расширения eval set.

### 4.4 Validation study (опционально)

Если удастся привлечь 1–2 библиотекарей или предметных экспертов:

- **Дизайн:** pairwise preference, 20–30 случайно выбранных открыток, для каждой — два описания от разных моделей, эксперт выбирает "которое лучше для каталогизации/поиска". Слепое сравнение, без указания на источник.
- **Цель:** проверить, коррелируют ли автоматические сигналы (CLIPScore_RU, R@k, LLM-judge) с экспертным предпочтением.
- **Использование:** дополнительная валидация триады, **не gating-метрика**. Если коррелирует — повышает доверие к автоматическим числам в основной части работы. Если эксперта найти не удастся — отчётность строится только на триаде из §4.1; ограничение явно обсуждается в §9.

### 4.5 Сравнения и критерии решения

Каждое сравнение моделей сопровождается:

- средними и std по eval pool для всех трёх сигналов триады;
- bootstrap-CI 95 % для каждого сигнала;
- paired bootstrap p-value vs baseline (1000 ресэмплов);
- per-image таблицей в приложении.

Эксперимент считается **успешным**, если:

- ≥ 2 из 3 сигналов триады выросли значимо (p < 0.1);
- ни один из 3 не упал значимо;
- Latency-регрессии не критичны (не более ×2 vs baseline).

---

## 5. Implementation tasks (I-1 … I-8)

Это задачи по строительству платформы оценки и подготовки данных. **В тезисную таблицу экспериментов они не входят** — это инфраструктура. Задачи трекаются в `docs/experiments_log.md` отдельным разделом.

| ID | Задача | Артефакт | Зависимости |
|---|---|---|---|
| **I-1** | Multilingual CLIP scorer integration | `scripts/evaluate.py` поддерживает флаг `--clip-multilingual <model>`, считает `CLIPScore_RU` параллельно с `CLIPScore_EN` | — |
| **I-2** | Сделать `archive_description_ru` каноническим текстом для CLIPScore | обновлённый `evaluate.py` отчитывает CLIPScore_RU(archive) как primary | I-1 |
| **I-3** | Расширение eval pool до 200+ через скрейпинг НЭБ-каталога (без ручных аннотаций) | `scripts/scrape_neb.py`, расширенный `data/eval/pool/`, манифест с метаданными НЭБ | — |
| **I-4** | Bootstrap CI и paired bootstrap test инфраструктура | `scripts/eval_stats.py`, утилита `bootstrap_ci(values, n=1000)` и `paired_bootstrap(a, b)` | — |
| **I-5** | Reference-free retrieval R@1, R@5 в evaluate.py | `evaluate.py` считает retrieval-метрики по пулу из I-3 | I-1, I-3 |
| **I-6** | LLM-as-judge протокол: фиксированная рубрика (faithfulness, completeness, style, no-hallucinations, brevity, по 1–5), фиксированный system prompt, фиксированная модель | `scripts/llm_judge.py`, `docs/llm_judge_rubric.md` (закрепляется до запуска E) | I-3 |
| **I-7** | Cross-metric correlation analysis | `scripts/eval_correlate.py` (Spearman / Kendall между CLIPScore_RU, R@k, LLM-judge, BLIP-ITM) | I-1, I-5, I-6 |
| **I-8** *(опционально)* | Pairwise expert study | `scripts/expert_study.py` (генерация пар, сбор результатов, корреляция с триадой) | I-6 |

**Pilot после I-1…I-6.** До запуска экспериментов — один прогон baseline по новой инфраструктуре с фиксацией всех чисел триады и калибровкой LLM-judge рубрики на baseline-описаниях. После этого рубрика и модель LLM-judge замораживаются.

---

## 6. Эксперименты (E-1 … E-9)

9 экспериментов по 7 тематическим блокам. **Каждый = строка в тезисной таблице экспериментов** с гипотезой, конфигурацией и числами триады. Если по результатам нескольких прогонов часть экспериментов окажется избыточной (одинаковые выводы или незначимые изменения) — компактизуем post-hoc до 6–7.

### Блок A — Captioning backbone

#### E-1: BLIP-base vs BLIP-large (zero-shot)

| Параметр | Значение |
|---|---|
| **Гипотеза** | BLIP-large даёт более точные и детальные описания → +0.02–0.05 на CLIPScore_RU(archive). |
| **Метод** | Заменить `Salesforce/blip-image-captioning-base` на `Salesforce/blip-image-captioning-large` в `CaptionGenerator`. Остальной пайплайн без изменений. |
| **Конфигурации** | base (baseline) vs large; одинаковый decoding (greedy или фиксированный из E-5+6 если он уже прошёл). |
| **Метрики решения** | Триада + CLIPScore_EN; latency. |
| **Критерий успеха** | Триада выполнена (см. §4.5), latency ≤ ×2 baseline. |
| **Compute** | ~30 мин evaluation × 200 images на MPS, на каждый из вариантов. |

### Блок B — Decoding & prompting

#### E-5+6: Beam grid × prompted prefix

| Параметр | Значение |
|---|---|
| **Гипотеза** | Beam search с длинной поправкой и доменным prefix-prompt дают аддитивный прирост на CLIPScore_RU. |
| **Метод** | Grid: `num_beams ∈ {1, 3, 5}` × `length_penalty ∈ {1.0, 1.2}` × `prompt_prefix ∈ {"", "a vintage postcard depicting"}` = 12 конфигов. Использовать лучший backbone из E-1. |
| **Метрики решения** | Триада; ablation table. |
| **Критерий успеха** | Лучшая конфигурация бьёт `num_beams=1, prefix=""` хотя бы на 2 из 3 сигналов триады. |
| **Compute** | 12 × 200 images × несколько секунд каждое; ~1 час. |
| **Замечание** | После E-5+6 фиксируем лучшую decoding-конфигурацию для всех downstream-экспериментов. |

### Блок C — Captioning fine-tune

#### E-7: Clean NYPL FT

| Параметр | Значение |
|---|---|
| **Гипотеза** | Hybrid `caption + title` метки шумные (postcard_4 "лукошко" → "violin"). Чистые метки (caption-only либо caption + title только при `CLIPScore(title, image) ≥ 0.25`) дадут лучше CLIPScore_RU после FT. |
| **Метод** | Перегенерировать train data в трёх вариантах: (a) caption only, (b) caption + filtered title, (c) текущий title-concat (как `caplift_v2`, baseline). FT BLIP base, lr=3e-5, bs=4, 2 epochs (как `caplift_v2`). |
| **Метрики решения** | Триада на каждом из 3-х FT моделей; сравнение с zero-shot E-1. |
| **Критерий успеха** | Лучший variant бьёт `caplift_v2` и zero-shot baseline по триаде. |
| **Compute** | ~6–8 ч на конфиг на MPS × 3 = ~1 сутки фоновой работы. На GPU — ~1.5–3 ч. |

#### E-8: LoRA single config

| Параметр | Значение |
|---|---|
| **Гипотеза** | LoRA с правильным lr достигает результата, сравнимого с full FT, при ×10 меньшем compute. |
| **Метод** | Один LoRA-конфиг: r=16, α=32, dropout=0.05, target=Q/K/V, 5 epochs. lr выбираем после короткого pilot (1 epoch на 200 примерах) из {1e-5, 3e-5}. Данные — лучший variant из E-7. |
| **Метрики решения** | Триада vs full FT (E-7 best) и vs baseline. |
| **Критерий успеха** | Триада не уступает full FT > ε значимо; либо явно бьёт. |
| **Compute** | ~3–5 ч на MPS. |

### Блок D — End-to-end RU и translation swap

#### E-9: MarianMT → NLLB-200

| Параметр | Значение |
|---|---|
| **Гипотеза** | NLLB-200 (1.3B distilled) точнее переводит описания изображений на русский → CLIPScore_RU(caption_ru) и CLIPScore_RU(archive) растут. |
| **Метод** | Заменить `Translator` на `facebook/nllb-200-distilled-1.3B` (или `-600M`, если 1.3B не помещается на MPS). Все остальные компоненты неизменны. |
| **Метрики решения** | Триада + CLIPScore_RU(caption_ru) изолированно. |
| **Критерий успеха** | Прирост хотя бы на CLIPScore_RU(caption_ru); на триаде — успех по §4.5. |
| **Compute** | ~30 мин eval. Загрузка модели ~3 ГБ. |

#### E-11: Qwen2-VL-2B end-to-end RU [центральный архитектурный эксперимент]

| Параметр | Значение |
|---|---|
| **Гипотеза** | End-to-end RU VLM убирает компаундирующиеся ошибки этапов BLIP+MarianMT+template и даёт лучшее `archive_description` напрямую. **Альтернатива модульному пайплайну.** |
| **Метод** | Заменить BLIP+MarianMT+DescriptionBuilder одним вызовом `Qwen/Qwen2-VL-2B-Instruct` с фиксированным RU system prompt: "Ты библиотечный каталогизатор. Опиши открытку для архивного поиска: тип, сюжет, стиль, настроение. 2–3 предложения, без галлюцинаций." SigLIP метаданные сохраняем рядом для тегов. |
| **Метрики решения** | Триада, **в особенности CLIPScore_RU(archive)** и R@k. Latency обязательно. |
| **Критерий успеха** | Любой из двух исходов содержателен: (i) Qwen2-VL бьёт модульный пайплайн → центральная контрибуция работы; (ii) проигрывает → значимое сравнение, обсуждение когда модульность выигрывает. |
| **Compute** | ~1–2 ч inference на MPS over 200 images. Модель ~5 ГБ в fp16. |
| **Риск** | Неподходящий system prompt может дать слишком общие/слишком развёрнутые описания. Pilot на 5–10 изображениях для подбора prompt до основного прогона. |

### Блок E — Metadata extractor

#### E-13: SigLIP threshold calibration

| Параметр | Значение |
|---|---|
| **Гипотеза** | Текущие пороги (0.18–0.35) — догадки. Per-category калибровка на размеченном subset снизит false-positives и улучшит CLIPScore_RU(archive) через E-16 (или независимо). |
| **Метод** | Вручную разметить 50 изображений из eval pool по полям image_type/style/theme/mood (закрытый список → быстро, ~30 мин работы). Найти порог per category, максимизирующий F1 (или Cohen's kappa). Прописать в `siglip_metadata_extractor.py`. |
| **Метрики решения** | Триада на полном eval pool до и после; per-category precision/recall на 50-image subset. |
| **Критерий успеха** | Триада улучшилась или нейтральна; per-category F1 вырос. |
| **Compute** | Compute минимальный; ~30 мин ручной работы (закрытая разметка, не open-ended). |

### Блок F — Description assembly

#### E-16: Drop low-confidence sentences

| Параметр | Значение |
|---|---|
| **Гипотеза** | Текущий шаблон вставляет "Предположительно, это X. Настроение Y." всегда, даже при low-confidence. Это активно вредит archive_description (postcard_5/13/17/18). Не вставлять low-confidence предложения → +CLIPScore_RU. |
| **Метод** | В `DescriptionBuilder`: пропускать sentence о theme, если `theme_field.confident == False` (то же для mood). Сравнить с текущим `forced-template`. |
| **Метрики решения** | Триада. |
| **Критерий успеха** | По §4.5 — почти гарантированный win, ожидаем небольшой, но стабильный прирост. |
| **Compute** | Тривиальный. ~1 час на код + eval. |

### Блок G — Inference-time

#### E-18: Sampling + CLIP rerank

| Параметр | Значение |
|---|---|
| **Гипотеза** | Семплирование N=10 кандидатов captionа и реранк по CLIPScore против изображения дают более высокий CLIPScore_RU(archive), чем единичный greedy/beam decode. Дешёвый test-time win. |
| **Метод** | На inference: `do_sample=True, num_return_sequences=10, temperature=1.0` для CaptionGenerator. Каждое из 10 проходит весь пайплайн до archive_description. Выбрать кандидата с максимальным CLIPScore_RU(image, archive_description_ru) — **CLIP-скоринг через тот же multilingual CLIP, что в I-1**. |
| **Метрики решения** | Триада vs лучший configurable single-decode (E-5+6 best). |
| **Критерий успеха** | По §4.5. Latency × ~10 — это ожидаемо. |
| **Compute** | ×10 от обычного inference. ~2–3 ч на 200 images. |
| **Риск** | Circular evaluation: CLIPScore_RU используется и для рerankа, и для метрики. **Метрика для оценки этого эксперимента — отдельная — не CLIPScore_RU, а R@k и LLM-judge.** Это явно фиксируется в анализе. |

---

## 7. Структура отчётности (для тезиса)

### 7.1 Таблица экспериментов в работе

Одна **таблица экспериментов** со столбцами:

| ID | Block | Эксперимент | Конфигурация | CLIPScore_RU | R@1 | R@5 | LLM-judge | Latency | Итог |
|---|---|---|---|---|---|---|---|---|---|

Заполняется по мере прогона. baseline в первой строке.

### 7.2 Дополнительные таблицы

1. Cross-metric correlation matrix (из I-7) — Spearman между CLIPScore_RU / R@k / LLM-judge / BLIP-ITM.
2. Per-experiment breakdown по weak references (BLEU/METEOR — стилевой сигнал).
3. *(Если состоится I-8).* Корреляция экспертного pairwise preference с автоматическими сигналами.

### 7.3 Графики

- Кривые train/val loss для E-7, E-8 (расширение `Untitled1.ipynb`).
- Heatmap pairwise similarity для baseline и best (на основе `Pairwise comparison.ipynb`).
- PCA-проекция image vs archive_description_ru для baseline и best.
- Per-image bar-chart: CLIPScore_RU(archive) baseline vs best.

### 7.4 Качественный анализ

- 5–8 case studies: куда улучшилось / куда сломалось.
- Особенно: открытки, где text/visual content расходятся (postcard_4 "лукошко", postcard_5 "эвенк") — ценны для обсуждения ограничений модульного пайплайна и Qwen2-VL.

---

## 8. Порядок выполнения и зависимости

### 8.1 Implementation phase

```
I-1 (multilingual CLIP) ─┐
I-2 (archive metric)     │
I-3 (НЭБ pool 200+)      ├─► PILOT прогон baseline по новой инфраструктуре
I-4 (bootstrap)          │     (фиксация рубрики LLM-judge, заморозка baseline-чисел)
I-5 (R@k)                │
I-6 (LLM-judge)          ┘
I-7 (correlation)        ── после первых E-результатов
I-8 (expert)             ── опционально, в любой момент после I-6
```

### 8.2 Experiment phase

```
Cheap, run first:
  E-16 (drop low-conf)  ─┐
  E-5+6 (decoding)       ├─► фиксируем выигравшие настройки в пайплайне
  E-13 (threshold cal)   ┘     для downstream-экспериментов

Backbone & FT:
  E-1 (BLIP-large) ─► E-7 (clean FT) ─► E-8 (LoRA)

Translation / E2E (параллельно):
  E-9 (NLLB-200)
  E-11 (Qwen2-VL)

Inference-time (последним):
  E-18 (sampling + rerank)
```

### 8.3 Критические зависимости

- **Все E зависят от I-1..I-6** — без новой инфраструктуры эксперименты ранжируются по EN-CLIP с искажением.
- **E-5+6, E-16, E-13 фиксируются раньше E-7/E-8/E-11**, так как их результаты влияют на конфигурацию downstream.
- **Рубрика LLM-judge замораживается до основной серии E** (после baseline pilot). Изменения рубрики post-hoc запрещены — это превратит triad в circular metric.
- **E-18 использует CLIPScore_RU как rerank-критерий**: для оценки этого эксперимента как primary metric используем R@k и LLM-judge, не CLIPScore_RU.
- **E-7, E-8 — основные потребители compute**. Без GPU — последовательно в фоне; параллельно идут E-9, E-11, E-13.

---

## 9. Риски и открытые вопросы

### 9.1 Технические риски

| Риск | Митигация |
|---|---|
| MPS-нестабильность при FT BLIP (известны баги с fp16/grad clipping) | проверить на CPU/CUDA через Colab Pro как fallback |
| Multilingual CLIP сам по себе хуже EN-CLIP на EN-текстах | репортить обе метрики; не делать выводов о EN-генерации по RU-CLIP |
| Qwen2-VL-2B может не помещаться на MPS в fp16 | fallback на CPU inference (медленно, но работает) или на Qwen2-VL-2B fp8/4bit |
| LoRA при `lr=3e-6` (текущее значение в `train_blip_nypl_lora.py`) — недотренировка | pilot перед E-8 с двумя lr |

### 9.2 Методологические риски

- **Absence of ground truth — структурное свойство задачи.** В работе явно артикулируется, что система оценивается reference-free инструментами; references на аналогичные подходы в литературе (Hessel et al. 2021 — оригинальный CLIPScore; reference-free image captioning evaluation) приводятся в обзорной части.
- **Circular evaluation в E-18:** rerank по CLIPScore_RU и оценка по CLIPScore_RU = читерство. Митигация — явно использовать R@k и LLM-judge как primary metrics для этого эксперимента; CLIPScore_RU отчитывается, но не считается решающим.
- **LLM-as-judge не источник истины, а measurement instrument.** Рубрика и модель фиксируются до основной серии экспериментов; не пересматриваются под результаты. Pilot-прогон на baseline для калибровки рубрики разрешён, но фиксируется в `docs/llm_judge_rubric.md` до запуска E.
- **Validation study (I-8) — не источник истины,** а внешняя проверка триады. Если эксперт сходится с триадой — триада признаётся надёжной; если расходится — это материал для обсуждения ограничений.
- **Конкатенация title в hybrid caption (E-7 baseline) смещает FT-модель к стилю NYPL-метаданных, не к стилю описаний НЭБ.** Это и проверяет E-7.
- **Маленький eval pool:** 20 точек дают высокую дисперсию. Это блокер до завершения I-3 (200+).

### 9.3 Решения, которые надо принять до запуска

1. **Какой multilingual CLIP использовать как scorer?**
   Кандидаты: `sentence-transformers/clip-ViT-B-32-multilingual-v1`, `M-CLIP/XLM-Roberta-Large-Vit-B-32`, `laion/CLIP-ViT-H-14-multilingual`. Решение принимаем после pilot-прогона I-1.
2. **Какая LLM для as-judge?** Кандидаты: Claude Opus, GPT-4o, Qwen2.5-72B. Зависит от доступа и бюджета. Фиксируется до I-6.
3. **GPU-доступ?** Если нет — E-7 идёт в фоне, может удлинить календарный срок до 2.5 недель.
4. **Получится ли организовать I-8?** Если нет — отчётность строится только на триаде, ограничение явно обсуждается в §9.2.

---

## 10. Возможная компактизация после первых прогонов

Текущее количество — 9 экспериментов. После первой волны прогонов решаем по двум критериям, нужно ли сжимать:

- **Дублирующиеся выводы.** Если, например, E-1 (BLIP-large) и E-7 (clean FT) дают примерно одинаковый прирост, и E-7 поглощает E-1, можно оставить только E-7 в финальной таблице. Аналогично E-8 vs E-7.
- **Незначимые изменения.** Эксперимент с приростом ниже шума (по bootstrap CI) и без качественного эффекта — кандидат на сжатие в один абзац обсуждения.

Реалистичный финальный набор после компактизации — **6–7 экспериментов**, оставляя минимум по одному из каждого блока (A, B, C, D, E, F, G), плюс центральный E-11.

---

## 11. Что дальше

После принятия документа:

1. Завести `docs/experiments_log.md` с двумя секциями: Implementation tasks (I-1…I-8) и Experiments (E-1…E-9). Туда стекается прогресс, числа, артефакты.
2. Реализовать I-1, I-2 в `scripts/evaluate.py` — зафиксировать новый baseline на триаде.
3. Параллельно — I-3 (`scripts/scrape_neb.py`).
4. После I-1…I-6 — pilot прогон baseline, заморозка рубрики LLM-judge.
5. Запустить E-16 + E-5+6 как первые "wins" (минимальный compute, гарантированно интерпретируемые).
