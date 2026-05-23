# Codebase refactoring plan

**Status:** drafted 2026-05-21, не начат. Запуск — после завершения текущего ref-based eval на НЭБ.

## Проблема

Кодовая база накопила ad-hoc артефакты за время итерационных экспериментов:

- **`scripts/`: 5 176 LOC в 27 плоских файлах.** Половина — legacy (BLIP fine-tune направление, CapFilt, ArtCap-prep), либо task-specific обёртки (`run_on_new_postcards.py`, `run_e08_artcap_eval.sh`, `run_semtest.sh`) — это нарушение принципа generic verb-based CLI.
- **`src/`: 1 dead-модуль** (`metadata_extractor_clip.py`, 120 LOC, не импортируется) + 2 legacy-модуля (`text_postprocessor.py`, `english_caption_postprocessor.py`) от pre-LLM-rewriter эпохи.
- **`evaluate.py` 564 LOC** с устаревшими флагами (`--drop-theme`, `--drop-mood`, `--template-mode`, `--drop-generic-style`).
- **12 legacy JSON-артефактов** в git: `data/eval/results/final/metrics_triad_E-1+16*`, `E-16*` — pre-E00 nomenclature.
- **7 notebooks**, из которых 4 апрельские (возможно устарели) + 1 `Untitled.ipynb`.
- **`docs/`:** `experiments.md` (437 строк, план) и `experiments_log.md` (304, лог) пересекаются.

## Ключевой принцип: verb-based generic CLI

> **Все скрипты в `scripts/` — generic verb-based CLI. Никаких hardcoded путей к датасетам, никаких `run_on_X.py`. Если нужна повторяемая команда — её место в `Makefile` или примере в README, не в коде отдельного скрипта.**

Скрипт = глагол (`evaluate`, `probe`, `ingest`, `benchmark`, `judge`). Объект — через `--input <path>` параметр. Один скрипт работает с любым корректным input в стандартном формате.

### Стандартный input-формат (references.jsonl)

Один формат для всего проекта — superset с **обязательными `image_path` + `reference_ru`**, любым числом optional-полей:

```json
{
  "image_path": "data/.../img.jpg",
  "reference_ru": "Curator-grade описание",
  "type": "Открытка",
  "source": "neb_leningrad_wwii",
  "year": "1943",
  "technique": "хромолитогр.",
  "theme_label": "...",
  "catalog_id": "..."
}
```

Все generic-инструменты (`evaluate.py`, `probe.py`, `eval_reference_based.py`) читают этот формат, игнорируя поля, которые им не нужны.

## P0 — удалить dead и task-specific код, generalize scripts (~1 час)

### 1. Удалить редандантные / dead файлы

**Scripts:**
- `compute_sds.py` (369 LOC, SDS убрали из метрик)
- `demo_vlm.py` (158 LOC, single-use)
- `run_on_new_postcards.py`, `run_e08_artcap_eval.sh`, `run_semtest.sh` — redundant с `evaluate.py --references <any.jsonl>`
- `import_existing.py` — one-off импорт legacy данных

**`src/`:**
- `metadata_extractor_clip.py` (120 LOC, не используется нигде)

**`notebooks/`:**
- `Untitled.ipynb`

**`data/eval/results/final/`:**
- 12 JSON-ов с префиксами `metrics_triad_E-1+16*`, `metrics_triad_E-16*`, `metrics_triad_baseline*` (pre-E00 эпоха)

### 2. Generalize probe-скрипты → `scripts/probe.py`

Объединить три файла (`siglip_theme_probe.py`, `siglip_technique_probe_neb.py`, `probe_on_postcards.py`) в один с subcommands:

```bash
# Обучение
python scripts/probe.py train \
  --data data/eval/references_neb_n224.jsonl \
  --label-field technique \
  --features-cache data/cache/siglip_features_neb.npz \
  --output models/probe_technique_neb.pkl

# Cross-domain inference
python scripts/probe.py predict \
  --probe models/probe_technique_neb.pkl \
  --images data/eval/references.jsonl \
  --output data/probe_predictions_n60.json
```

Контракт: `--data <jsonl>` с любым label-полем; кэш SigLIP-фич переиспользуется автоматически по hash от input file.

### 3. Generalize `scripts/benchmark.py`

Заменяет `benchmark_siglip_on_semart.py`:

```bash
python scripts/benchmark.py \
  --reference-labels data/semart/semart_val.jsonl \
  --method siglip-zero-shot \
  --taxonomy archival_v2 \
  --output data/benchmark_results.json
```

### 4. Реорганизация в подкаталоги

```
scripts/
  evaluate.py              # main: run pipeline on --references
  eval_stats.py            # bootstrap CI + paired tests
  eval_reference_based.py  # BERTScore_RU / chrF / ROUGE-L / BLEU
  llm_judge.py             # LLM-as-judge
  run_demo.py              # single-image inference
  preload_models.py        # cache prefill
  probe.py                 # NEW: unified train/predict/eval (P0)
  benchmark.py             # NEW: generic SigLIP benchmark (P0)
  exp_db.py                # DB ingest (move to src/ in P1)

  data/
    ingest_neb.py          # бывший scrape_neb_collection.py
    sample_eval_pool.py

  _legacy/                 # сохраняем для archival reproducibility
    train_blip.py
    train_blip_lora.py
    train_blip_lora_v2.py
    prepare_artcap.py
    prepare_semart.py
    prepare_splits.py
    generate_captions.py
    filter_captions.py
```

**Принцип `_legacy/`:** код работающий, attribution экспериментов сохраняется (можно воспроизвести E08 ArtCap LoRA при необходимости), но не в основном flow проекта.

### 5. Slim `evaluate.py`

Убрать legacy-флаги:
- `--drop-theme`, `--drop-mood`, `--template-mode`, `--drop-generic-style`

Эти флаги обслуживали E05a-d вариации DescriptionBuilder — все они зафиксированы в JSON-результатах, нет смысла оставлять их как live-флаги. Snapshot CLI с 16 флагов → ~8. Файл с 564 → ~400 LOC.

## P1 — структурные улучшения (~2-3 часа)

### 1. `src/` cleanup

- Удалить `text_postprocessor.py` и `english_caption_postprocessor.py` если они нигде не нужны вне legacy MarianMT-пути. Если нужны для воспроизводимости старых E00-E08 — пометить как `_legacy_text_postprocessor.py`.
- `theme_inference.py` (14 LOC) — либо удалить (вынести в DescriptionBuilder), либо расширить.

### 2. Move `scripts/exp_db.py` → `src/tsu_image_description/experiments_db.py`

Это библиотека (595 LOC, классы для работы с SQLite DB), а не CLI-скрипт. Должна жить в `src/`. Сохранить тонкий CLI-обёртку `scripts/manage_db.py` (~30 LOC) если нужно интерактивно дергать.

### 3. Refactor `evaluate.py` на 3 модуля

```
src/tsu_image_description/eval/
  __init__.py
  cli.py                   # argparse + main()
  metrics.py               # CLIPScore_EN, CLIPScore_RU, t2i_retrieval
  runner.py                # pipeline orchestration loop
```

`scripts/evaluate.py` становится тонкой обёрткой:
```python
from tsu_image_description.eval.cli import main
if __name__ == "__main__":
    main()
```

### 4. Notebooks audit

Для каждого `.ipynb` решить keep / archive / delete:

| Файл | Дата | Решение |
|---|---|---|
| `compare_experiments_per_image.ipynb` | 21 мая | ✅ Keep |
| `semtest_descriptions_by_image.ipynb` | 21 мая | ✅ Keep |
| `experiments_results.ipynb` | 11 мая | 🟡 Проверить, не устарел ли |
| `pairwise_comparison.ipynb` | 27 апр | 🔴 Архив или удалить |
| `embeddings_exploration.ipynb` | 23 апр | 🔴 Архив или удалить |
| `pca_tsne_image_caption.ipynb` | 27 апр | 🔴 Архив или удалить |
| `Untitled.ipynb` | — | 🗑️ Уже в P0 |

Архив = переместить в `notebooks/_archive/`.

### 5. `docs/` merge

`experiments.md` (плановая структура) + `experiments_log.md` (статусный лог) пересекаются. Варианты:
- **Объединить:** один `experiments.md` с двумя секциями `## План` и `## Статус`
- **Разделить роли явно:** оставить как есть, но добавить шапки «План экспериментов (read-only архив)» и «Лог исполнения (живой документ)»

## P2 — polish (~2-3 часа)

### 1. `scripts/README.md`

Таблица: имя скрипта → что делает → когда запускать → требуемый input.

### 2. Docstring sweep

У каждой публичной функции в `src/` — `"""..."""`. Сейчас есть, но не везде.

### 3. Type hints

Добавить в скрипты, где их нет (например, `evaluate.py` сильно недотипизирован).

### 4. Makefile / pyproject scripts

```makefile
.PHONY: eval-neb eval-n60 probe ingest demo

eval-neb:
	PYTHONPATH=src python scripts/evaluate.py \
	  --references data/eval/references_neb_n224.jsonl \
	  --use-llm-rewriter --taxonomy-version archival_v2 \
	  --output data/eval/results/metrics_E12_neb_n224.json

eval-n60:
	PYTHONPATH=src python scripts/evaluate.py \
	  --references data/eval/references.jsonl \
	  --use-llm-rewriter --taxonomy-version archival_v2 \
	  --output data/eval/results/metrics_E12_n60.json

probe:
	PYTHONPATH=src python scripts/probe.py train \
	  --data data/eval/references_neb_n224.jsonl \
	  --label-field technique \
	  --output models/probe_technique.pkl

demo:
	PYTHONPATH=src python scripts/run_demo.py --image $(IMG)
```

### 5. `.gitignore` review

Добавить:
- `data/neb_test/`
- `logs/*.log`
- `**/__pycache__/`
- `.idea/`
- `models/` (если артефакты не должны быть в git)

Убрать tracked-артефакты, которые не должны быть в git.

## Ожидаемый итог

| Метрика | До | После P0 | После P0+P1 | После всего |
|---|---:|---:|---:|---:|
| Total LOC (Python) | ~6 700 | ~5 300 | ~4 800 | ~4 800 + docstrings |
| Scripts files в flat-каталоге | 27 | 6 в корне + категории | 6 в корне + категории | 6 в корне + категории |
| Dead/legacy code в main flow | ~700 LOC | 0 | 0 | 0 |
| `evaluate.py` LOC | 564 | ~400 | ~150 (CLI) + 2 модуля | то же |
| Task-specific run-скрипты | 5 | 0 | 0 | 0 |
| Generic CLI commands | 5 partial | 7 unified | 7 unified | 7 + Makefile |

## Принципы для будущего

1. **Verb-based naming:** `evaluate.py`, не `run_on_new_postcards.py`.
2. **`--input` parameter > hardcoded path.** Если файл уже в репозитории, его путь — default в `argparse`, не assertion.
3. **Standard input format:** один JSONL-schema с обязательными `image_path` + `reference_ru`, optional-полями для domain-specific метаданных.
4. **One source of truth для experiment tracking:** `experiments.db` через `experiments_db.py`, не разрозненные JSON.
5. **`_legacy/` directory для архивных experiments:** код работающий, документирован, но изолирован от main flow.
6. **Каждый скрипт декларирует свой контракт** в docstring (`Input: ..., Output: ..., Side effects: ...`).

## Триггер запуска

После завершения **A — ref-based eval на НЭБ n=224** (текущая задача, ETA ~15 мин на 2026-05-21).

Затем — P0 первым шагом, как самостоятельная commit-серия (или одна крупная коммит с чёткими секциями).
