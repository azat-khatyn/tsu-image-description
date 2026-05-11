# Changelog

Все значимые изменения в проекте фиксируются в этом файле.

## [0.1.0] - 2026-03

### Added
- Собран базовый vision-language пайплайн для генерации описаний изображений открыток и плакатов.
- Реализована генерация caption на английском языке.
- Добавлен перевод caption на русский язык.
- Реализовано извлечение семантических метаданных изображения:
  - тип,
  - стиль,
  - тематика,
  - настроение.
- Добавлены модули агрегации метаданных и сборки итогового описания.
- Добавлены базовые метрики оценки качества генерации.

### Notes
- На этом этапе проект представлял собой исследовательский прототип без отдельного приложения и без контейнеризации.

## [0.2.0] - 2026-04

### Added
- Добавлен REST API для запуска инференса.
- Добавлено минимальное web-приложение для демонстрации работы системы.
- Добавлена Docker-контейнеризация проекта.
- Добавлена поддержка запуска через `docker compose`.
- Добавлена документация по установке, запуску и структуре проекта.
- Добавлена поддержка двух сценариев входа:
  - загрузка изображения как файла;
  - передача пути к локальному изображению.

### Changed
- Исследовательский прототип преобразован в MVP-сервис.
- Обновлена структура проекта под клиент-серверную архитектуру.

### Fixed
- Исправлены ошибки интеграции зависимостей при запуске в Docker.
- Добавлена недостающая зависимость `protobuf` для корректной работы SigLIP tokenizer.

## [0.3.0] - 2026-05

### Added
- Дизайн-документ экспериментов [docs/experiments.md](docs/experiments.md) с reference-free framing и разделением Implementation tasks (I-1…I-8) / Experiments (E-1…E-9).
- Триада reference-free метрик в `scripts/evaluate.py`: CLIPScore_RU (через M-CLIP) + Recall@k (i2t / t2i).
- `scripts/eval_stats.py` — bootstrap CI и paired-test для сравнения экспериментов.
- `scripts/llm_judge.py` + `docs/llm_judge_rubric.md` — каркас LLM-as-judge оценки (Claude Sonnet 4.6 по умолчанию).
- `docs/experiments_log.md` — единый трекер прогресса по I-x и E-x.
- Параметризация `DescriptionBuilder` через `include_theme` / `include_mood` (для E-16).
- Baseline на триаде + первый прогон E-16 (drop theme/mood): см. `data/eval/results/final/`.

### Changed
- **Реструктуризация репозитория:**
  - Entry-point скрипты переехали из `src/` в `scripts/` (`train_blip.py`, `train_blip_lora.py`, `run_demo.py`).
  - `data/` реорганизована: `data/eval/{images,results/final}`, `data/nypl/{images,splits}`.
  - Notebooks переименованы в snake_case.
  - Pipeline принимает `builder_kwargs` для конфигурации DescriptionBuilder.
- Канонической метрикой стал `CLIPScore_RU(image, archive_description_ru)` через M-CLIP вместо `CLIPScore_EN(caption_en)`.

### Removed
- `src/evaluate.py` (старый BLEU/METEOR-оценщик), `scripts/evaluate_full.py`, `scripts/evaluate_lora.py` — функциональность объединена в `scripts/evaluate.py`.
- `src/tsu_image_description/infer_finetuned.py` — заменено вызовом `pipeline.run()` или `scripts/run_demo.py --finetuned`.
- `data/data_check.py`, `notebooks/Untitled1.ipynb`, `notebooks/loss_curve.png`, root `__init__.py` — мёртвый код / устаревшие артефакты.

### Fixed
- `.gitignore` теперь корректно игнорирует регенерируемые артефакты (train splits, embedding-файлы, промежуточные plots/metrics) и трекает только: код, документацию, 20 eval-изображений, `data/eval/results/final/` (baseline + финалы экспериментов).