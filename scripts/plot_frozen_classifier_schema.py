"""plot_frozen_classifier_schema.py — Схема обучения классификатора поверх
замороженных визуальных признаков SigLIP (рисунок для ВКР).

Поток: размеченные изображения -> SigLIP encoder (заморожен) -> векторы признаков
-> линейный классификатор (обучаемый) -> предсказание -> loss; градиент идёт
ТОЛЬКО до линейного слоя (пунктир), энкодер не дообучается.

Использование:
    python scripts/plot_frozen_classifier_schema.py --out docs/figures/fig_frozen_classifier_schema.png
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


NODES = [
    ("Размеченные\nизображения\n(SemArt TYPE /\nНЭБ техника)", "#eef2f7", "#333"),
    ("SigLIP encoder\n❄ заморожен\n(grad off)", "#dcdcdc", "#333"),
    ("Векторы\nпризнаков\n(embeddings)", "#eef2f7", "#333"),
    ("Линейный\nклассификатор\n(обучаемый)", "#08519c", "#fff"),
    ("Предсказание\nкласса", "#eef2f7", "#333"),
    ("Loss\n(cross-entropy)\nvs метка", "#fde0dd", "#333"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/figures/fig_frozen_classifier_schema.png")
    args = ap.parse_args()

    n = len(NODES)
    bw, bh, gap = 2.3, 1.5, 0.9
    xs = [i * (bw + gap) for i in range(n)]
    cy = 0.0

    fig, ax = plt.subplots(figsize=(15, 4.6))
    for x, (text, fc, tc) in zip(xs, NODES):
        ax.add_patch(FancyBboxPatch(
            (x, cy - bh / 2), bw, bh,
            boxstyle="round,pad=0.02,rounding_size=0.15",
            fc=fc, ec="#555", lw=1.3))
        ax.text(x + bw / 2, cy, text, ha="center", va="center",
                fontsize=10, color=tc, fontweight="bold")

    # прямые стрелки (поток)
    for i in range(n - 1):
        x0 = xs[i] + bw
        x1 = xs[i + 1]
        ax.add_patch(FancyArrowPatch((x0, cy), (x1, cy),
                                     arrowstyle="-|>", mutation_scale=18,
                                     color="#333", lw=1.6))

    # пунктирная стрелка градиента: Loss -> Линейный классификатор (дуга сверху)
    loss_top = (xs[5] + bw / 2, cy + bh / 2)
    cls_top = (xs[3] + bw / 2, cy + bh / 2)
    ax.add_patch(FancyArrowPatch(loss_top, cls_top,
                                 connectionstyle="arc3,rad=0.35",
                                 arrowstyle="-|>", mutation_scale=16,
                                 color="#c0392b", lw=1.6, ls="--"))
    midx = (loss_top[0] + cls_top[0]) / 2
    ax.text(midx, cy + bh / 2 + 1.15, "градиент только сюда\n(энкодер не дообучается)",
            ha="center", va="center", fontsize=9.5, color="#c0392b")

    ax.set_xlim(-0.6, xs[-1] + bw + 0.6)
    ax.set_ylim(-1.6, 2.9)
    ax.axis("off")
    ax.set_title("Обучение линейного классификатора поверх замороженных признаков SigLIP",
                 fontsize=13, pad=12)
    fig.tight_layout()
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print("->", out)


if __name__ == "__main__":
    main()
