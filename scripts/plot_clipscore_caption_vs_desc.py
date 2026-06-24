"""plot_clipscore_caption_vs_desc.py — Рис. 9: CLIPScore базовой подписи vs итогового описания.

База — каноничный прогон metrics_testset_nollm.json (656, по источникам, единая
конфигурация без LLM: CLIPScore устойчив к LLM-редактору). Три ряда: англ. подпись
(caption_en, энкодер CLIP-EN), рус. подпись (caption_ru) и итоговое описание
(archive RU) — последние два в одной модели M-CLIP и сравнимы напрямую.

Использование:
    python scripts/plot_clipscore_caption_vs_desc.py \
        --metrics data/eval/results/metrics_testset_nollm.json \
        --out docs/figures/fig_clipscore_caption_vs_desc.png
"""

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ORDER = [
    ("neb_wwii", "НЭБ, ВОВ\n(224)"),
    ("neb_diverse", "НЭБ, общий\n(220)"),
    ("nypl_curated", "NYPL\n(200)"),
    ("semantic_demo", "Демо\n(12)"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", default="data/eval/results/metrics_testset_nollm.json")
    ap.add_argument("--out", default="docs/figures/fig_clipscore_caption_vs_desc.png")
    args = ap.parse_args()

    bs = json.load(open(args.metrics, encoding="utf-8"))["summary"]["by_source"]
    labels = [lab for _, lab in ORDER]
    en = [bs[s]["CLIPScore_EN_caption_en"] for s, _ in ORDER]
    ru = [bs[s]["CLIPScore_RU_caption_ru"] for s, _ in ORDER]
    ar = [bs[s]["CLIPScore_RU_archive_ru"] for s, _ in ORDER]

    x = np.arange(len(labels))
    w = 0.26
    fig, ax = plt.subplots(figsize=(9, 5.5))
    b1 = ax.bar(x - w, en, w, label="англ. подпись (CLIP-EN)", color="#bdd7e7")
    b2 = ax.bar(x, ru, w, label="рус. подпись (M-CLIP)", color="#6baed6")
    b3 = ax.bar(x + w, ar, w, label="итоговое описание (M-CLIP)", color="#08519c")

    for bars in (b1, b2, b3):
        for r in bars:
            ax.annotate(f"{r.get_height():.3f}", (r.get_x() + r.get_width() / 2, r.get_height()),
                        textcoords="offset points", xytext=(0, 3), ha="center", fontsize=8)

    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("CLIPScore (косинус изображение↔текст)")
    ax.set_ylim(0.25, 0.37)
    ax.set_title("CLIPScore: базовая подпись и итоговое описание по выборкам\n"
                 "итоговое описание ≥ рус. подписи (одна модель M-CLIP) → сборка улучшает привязку",
                 fontsize=11)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    plt.close()
    print("->", args.out)
    for s, lab in ORDER:
        v = bs[s]
        print(f"  {s}: en={v['CLIPScore_EN_caption_en']:.3f} ru={v['CLIPScore_RU_caption_ru']:.3f} "
              f"arch={v['CLIPScore_RU_archive_ru']:.3f}")


if __name__ == "__main__":
    main()
