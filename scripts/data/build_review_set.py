"""build_review_set.py — собрать демонстрационный набор для рецензирования.

Generic verb-based CLI. На вход — два metrics-JSON (например, опорный n=60 и
литературная переработка n=224). На выход — Markdown-файл с парами
«эталонное описание ↔ автоматически сгенерированное» для качественной оценки
экспертом.

Назначение — подготовка пакета изображений и описаний для рецензента РНБ.

Использование (значения по умолчанию воспроизводят набор, описанный в
README → «Демонстрационный набор для рецензирования РНБ»):

    PYTHONPATH=src python scripts/data/build_review_set.py \\
        --output demo/review_set_rgb_rnb.md
"""

import argparse
import json
from pathlib import Path

# По умолчанию используем артефакты, прокомментированные в README.
DEFAULT_RGB_METRICS = "data/eval/results/final/metrics_E06b_archival_v2_n60.json"
DEFAULT_NEB_METRICS = "data/eval/results/metrics_E12_neb_n224.json"

# Целевые наборы файлов (id → краткое описание категории).
DEFAULT_RGB_PICKS = [
    ("postcard_1.jpg",  "природный пейзаж"),
    ("postcard_6.jpg",  "военно-морская техника"),
    ("postcard_11.jpg", "рождественская открытка"),
    ("postcard_16.jpg", "анималистика"),
    ("postcard_20.jpg", "детский портрет"),
]

# НЭБ-открытки выбираем по характерным признакам в curator-описании.
NEB_CATEGORIES = [
    ("городской пейзаж",   lambda r: ("Вид" in r) and ("Ленинграде" in r or "набережн" in r) and "Карикатур" not in r),
    ("военный сюжет",      lambda r: "Стрелк" in r or "Невы" in r and "военно" in r.lower()),
    ("праздник / агитация", lambda r: "Сандружинниц" in r or "пехотинец" in r),
    ("портрет",            lambda r: "Портрет" in r and ("медальон" in r or "Петрова" in r or "погруд" in r)),
    ("карикатура",         lambda r: "Карикатур" in r or "Гитлер" in r),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--rgb-metrics", default=DEFAULT_RGB_METRICS,
                   help="Metrics JSON for RGB / n=60 set")
    p.add_argument("--neb-metrics", default=DEFAULT_NEB_METRICS,
                   help="Metrics JSON for NEB / curator-annotated set")
    p.add_argument("--output", default="demo/review_set_rgb_rnb.md")
    p.add_argument("--n-per-source", type=int, default=5,
                   help="Number of postcards to pick from each source")
    return p.parse_args()


def load_per_item(path):
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    return d.get("per_item", [])


def pick_rgb(items, picks):
    """Pick by exact filename, preserving order in `picks`."""
    by_name = {Path(it["image_path"]).name: it for it in items}
    out = []
    for fname, category in picks:
        if fname not in by_name:
            print(f"[WARN] {fname} not in RGB metrics — skipping")
            continue
        out.append({"item": by_name[fname], "category": category})
    return out


def pick_neb(items, categories, n_target):
    """Pick one item per category by matching reference_ru against predicate."""
    seen_categories = set()
    out = []
    for it in items:
        ref = (it.get("reference_ru") or "").strip()
        if not ref:
            continue
        for cat_name, predicate in categories:
            if cat_name in seen_categories:
                continue
            if predicate(ref):
                out.append({"item": it, "category": cat_name})
                seen_categories.add(cat_name)
                break
        if len(out) >= n_target:
            break
    return out


def render_markdown(rgb_picks, neb_picks):
    L = []
    L.append("# Демонстрационный набор для рецензирования РНБ\n")
    L.append("Сбалансированный набор из 10 открыток с парами «эталонное описание ↔ автоматически "
             "сгенерированное архивное описание», предназначенный для качественной оценки экспертом "
             "Российской государственной библиотеки.\n")
    L.append("**Источники описаний для сравнения:**\n")
    L.append("- Открытки РГБ — ручные эталоны проектной команды;")
    L.append("- Открытки НЭБ — curator-grade описания из RUSMARC поля 327 «Примечание содержания».\n")

    L.append("## Часть 1. Открытки РГБ\n")
    for i, p in enumerate(rgb_picks, 1):
        it = p["item"]
        fn = Path(it["image_path"]).name
        L.append(f"### {i}. {fn} ({p['category']})\n")
        L.append(f"- **Эталон (ref):** {it.get('reference_ru', '—')}")
        L.append(f"- **Описание пайплайна (archive_ru):** {it.get('archive_ru', '—')}")
        L.append(f"- **Файл:** `{it['image_path']}`\n")

    L.append("## Часть 2. Открытки НЭБ (коллекция №529 «Ленинград в ВОВ»)\n")
    for i, p in enumerate(neb_picks, 6):
        it = p["item"]
        fn = Path(it["image_path"]).name
        L.append(f"### {i}. {fn} ({p['category']})\n")
        L.append(f"- **Эталон (RUSMARC 327):** {it.get('reference_ru', '—')}")
        L.append(f"- **Описание пайплайна (archive_ru):** {it.get('archive_ru', '—')}")
        L.append(f"- **Файл:** `{it['image_path']}`\n")

    L.append("## Назначение набора\n")
    L.append("Эксперту предлагается оценить:\n")
    L.append("1. Семантическую близость автоматически сгенерированного описания к экспертному эталону.")
    L.append("2. Удобочитаемость и архивный стиль русского языка.")
    L.append("3. Наличие/отсутствие фабрикаций — деталей, отсутствующих на изображении.")
    L.append("4. Пригодность описания как поискового признака.\n")
    L.append("## Известные ограничения метода\n")
    L.append("- Пайплайн не распознаёт именованные сущности (Адмиралтейство, Медный всадник, А. Гитлер).")
    L.append("- Пайплайн не извлекает текст с открытки (надписи, лозунги, подписи художника).")
    L.append("- Пайплайн не классифицирует историко-культурный контекст (блокада, праздничные кампании).\n")
    L.append("Эти ограничения — следствие выбора BLIP-1 в качестве базовой модели подписей "
             "(см. README → «Будущая работа»).\n")
    return "\n".join(L)


def main():
    args = parse_args()
    rgb_items = load_per_item(args.rgb_metrics)
    neb_items = load_per_item(args.neb_metrics)
    print(f"[INFO] Loaded {len(rgb_items)} RGB items, {len(neb_items)} NEB items")

    rgb_picks = pick_rgb(rgb_items, DEFAULT_RGB_PICKS[: args.n_per_source])
    neb_picks = pick_neb(neb_items, NEB_CATEGORIES, args.n_per_source)
    print(f"[INFO] Picked {len(rgb_picks)} RGB + {len(neb_picks)} NEB")

    md = render_markdown(rgb_picks, neb_picks)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md, encoding="utf-8")
    print(f"[INFO] Wrote {out_path}")


if __name__ == "__main__":
    main()
