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
- **Артефакт:** `scripts/evaluate.py`, `data/eval/results/metrics_triad_v1.json`.

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
| baseline | — | BLIP-base + MarianMT + template | DONE | **0.325** | i2t 0.75 / t2i 0.70 | i2t 1.00 / t2i 0.90 | TBD | 0.70 s | 2026-05-10 |
| E-1 | A | BLIP-base → BLIP-large | TODO | — | — | — | — | — | — |
| E-5+6 | B | Decoding × prefix grid | TODO | — | — | — | — | — | — |
| E-7 | C | Clean NYPL FT | TODO | — | — | — | — | — | — |
| E-8 | C | LoRA single config | TODO | — | — | — | — | — | — |
| E-9 | D | MarianMT → NLLB-200 | TODO | — | — | — | — | — | — |
| E-11 | D | Qwen2-VL-2B end-to-end RU | TODO | — | — | — | — | — | — |
| E-13 | E | SigLIP threshold calibration | TODO | — | — | — | — | — | — |
| E-16 | F | Drop theme+mood (blunt) | DONE — null on aggregate | 0.326 (+0.001, p=0.88) | i2t 0.80 / t2i 0.75 | i2t 0.95 / t2i 0.95 | TBD | 0.70 s | 2026-05-10 |
| E-18 | G | Sampling + CLIP rerank | TODO | — | — | — | — | — | — |

---

## Заметки по запускам

### 2026-05-10 — I-1 + I-2 + I-4: первый baseline на триаде (partial)

Прогон `scripts/evaluate.py` на текущих 20 открытках:

```
CLIPScore_EN_caption_en       : 0.298
CLIPScore_RU_caption_ru       : 0.301
CLIPScore_RU_archive_ru       : 0.325   ← primary
CLIPScore_RU_reference_ru     : 0.309
Latency mean                  : 0.70 s
```

Результат: см. `data/eval/results/metrics_triad_v1.json`.

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
