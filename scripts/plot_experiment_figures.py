"""plot_experiment_figures.py — экспериментальные рисунки 1, 2, 4 для ВКР.

Числа — из главы «Эксперименты» (docs/experiments.md) и сохранённых прогонов
E12/E13/testset; для legacy-таксономии и probe чистых JSON нет, значения взяты
из задокументированных таблиц experiments.md (групп 3, 4.4, 5).

Рисунки:
  1. CLIPScore и средняя позиция целевого изображения для двух таксономий.
  2. Способы формирования описания по CLIPScore и retrieval.
  4. Accuracy zero-shot vs supervised (линейный classifier на признаках SigLIP).

Использование:
    python scripts/plot_experiment_figures.py --out-dir docs/figures
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _label(ax, bars, fmt="{:.3f}"):
    for r in bars:
        ax.annotate(fmt.format(r.get_height()),
                    (r.get_x() + r.get_width() / 2, r.get_height()),
                    textcoords="offset points", xytext=(0, 3), ha="center", fontsize=9)


def fig_taxonomy(out):
    # experiments.md гр.3: Начальная(legacy) 0.340/1.92, Каталожная(archival) 0.319/1.85
    cfgs = ["Начальная\n(legacy_v1)", "Каталожная\n(archival_v2)"]
    clip = [0.340, 0.319]
    rank = [1.92, 1.85]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 5))
    b = a1.bar(cfgs, clip, color=["#9ecae1", "#08519c"], width=0.6)
    _label(a1, b); a1.set_ylim(0.30, 0.35); a1.set_ylabel("CLIPScore (archive RU)")
    a1.set_title("CLIPScore (выше — лучше)")
    b = a2.bar(cfgs, rank, color=["#fdae6b", "#e6550d"], width=0.6)
    _label(a2, b, "{:.2f}"); a2.set_ylim(1.7, 2.0); a2.set_ylabel("средняя позиция в выдаче")
    a2.set_title("Средняя позиция целевого изображения (ниже — лучше)")
    fig.suptitle("Две таксономии: каталожная снижает CLIPScore, но улучшает retrieval", fontsize=12)
    for a in (a1, a2): a.grid(axis="y", alpha=0.25)
    fig.tight_layout(); fig.savefig(out / "fig_taxonomy_clipscore_rank.png", dpi=200, bbox_inches="tight"); plt.close()
    print("->", out / "fig_taxonomy_clipscore_rank.png")


def fig_description_methods(out):
    # CLIPScore (НЭБ-224, pool-независим): шаблон 0.325(no-LLM), v1 0.315(neb224_llm), v2 0.280(E13)
    # R@1 (НЭБ-224, пул-224): v1 0.165(E12), v2 0.179(E13). Шаблон на этом пуле не прогонялся.
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5))
    m = ["Шаблон\n(без LLM)", "LLM v1\n(архивный)", "LLM v2\n(описательный)"]
    clip = [0.325, 0.315, 0.280]
    b = a1.bar(m, clip, color=["#bdbdbd", "#08519c", "#6baed6"], width=0.6)
    _label(a1, b); a1.set_ylim(0.25, 0.34); a1.set_ylabel("CLIPScore (archive RU)")
    a1.set_title("CLIPScore: LLM немного снижает буквальную привязку")
    m2 = ["LLM v1\n(архивный)", "LLM v2\n(описательный)"]
    r1 = [0.165, 0.179]
    b = a2.bar(m2, r1, color=["#08519c", "#6baed6"], width=0.5)
    _label(a2, b, "{:.3f}"); a2.set_ylim(0, 0.25); a2.set_ylabel("t2i R@1 (пул-224)")
    a2.set_title("Retrieval R@1 (стили запроса LLM)")
    fig.suptitle("Способы формирования описания: шаблон vs LLM (v1/v2)", fontsize=12)
    for a in (a1, a2): a.grid(axis="y", alpha=0.25)
    fig.tight_layout(); fig.savefig(out / "fig_description_methods.png", dpi=200, bbox_inches="tight"); plt.close()
    print("->", out / "fig_description_methods.png")


def fig_zeroshot_vs_supervised(out):
    # experiments.md гр.5.3 (SemArt, техника): zero-shot 0.571 -> linear probe 0.895
    cfgs = ["Zero-shot\nSigLIP", "Линейный classifier\nна признаках SigLIP"]
    acc = [0.571, 0.895]
    fig, ax = plt.subplots(figsize=(7, 5.2))
    b = ax.bar(cfgs, acc, color=["#9ecae1", "#08519c"], width=0.55)
    _label(ax, b)
    ax.set_ylim(0, 1.0); ax.set_ylabel("top-1 accuracy")
    ax.set_title("Accuracy zero-shot vs supervised (классификация техники, SemArt)\n"
                 "узкое место zero-shot — текстовые запросы, а не сами признаки SigLIP", fontsize=11)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout(); fig.savefig(out / "fig_zeroshot_vs_supervised.png", dpi=200, bbox_inches="tight"); plt.close()
    print("->", out / "fig_zeroshot_vs_supervised.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="docs/figures")
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    fig_taxonomy(out)
    fig_description_methods(out)
    fig_zeroshot_vs_supervised(out)


if __name__ == "__main__":
    main()
