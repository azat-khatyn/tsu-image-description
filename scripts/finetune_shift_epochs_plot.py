"""finetune_shift_epochs_plot.py — прогрессия доменного сдвига по эпохам.

Из кэша finetune_shift_epochs_export.py строит «скрипки» распределения близости
подписей к домену живописи SemArt по стадиям (база → e0 → e1 → e2) + тренд средних.
Показывает, что сдвиг (и забывание) нарастает с числом эпох дообучения.

Использование:
    python scripts/finetune_shift_epochs_plot.py \
        --cache data/eval/embeddings/finetune_shift_epochs \
        --out docs/figures/fig_finetune_shift_epochs.png
"""

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/eval/embeddings/finetune_shift_epochs")
    ap.add_argument("--out", default="docs/figures/fig_finetune_shift_epochs.png")
    args = ap.parse_args()

    d = np.load(args.cache + ".npz")
    meta = json.load(open(args.cache + ".meta.json", encoding="utf-8"))
    stages = meta["stages"]

    cc = d["corpus"].mean(0); cc /= np.linalg.norm(cc) + 1e-9
    sims = [d[s] @ cc for s in stages]
    means = [float(s.mean()) for s in sims]
    corpus_self = float((d["corpus"] @ cc).mean())

    fig, ax = plt.subplots(figsize=(9, 6))
    pos = list(range(len(stages)))
    colors = ["#1f77b4", "#ff9896", "#e377c2", "#d62728"]

    vp = ax.violinplot(sims, positions=pos, widths=0.7, showmeans=False, showextrema=False)
    for i, b in enumerate(vp["bodies"]):
        b.set_facecolor(colors[i]); b.set_alpha(0.55); b.set_edgecolor("gray")

    # точки-наблюдения (jitter) + средние с трендом
    rng = np.random.RandomState(0)
    for i, s in enumerate(sims):
        ax.scatter(np.full(len(s), i) + rng.uniform(-0.07, 0.07, len(s)),
                   s, s=9, c="black", alpha=0.25, zorder=3)
    ax.plot(pos, means, "-o", color="black", lw=2, zorder=4, label="среднее")
    for i, m in enumerate(means):
        ax.text(i, m + 0.012, f"{m:.3f}", ha="center", fontsize=10, fontweight="bold")

    ax.axhline(corpus_self, color="gray", ls="--", lw=1.2)
    ax.text(len(stages) - 1, corpus_self + 0.004,
            f"домен живописи (корпус, {corpus_self:.3f})", ha="right", fontsize=9, color="gray")

    labels = {"база": "база\n(без дообуч.)"}
    ax.set_xticks(pos)
    ax.set_xticklabels([labels.get(s, f"SemArt-LoRA\n{s}") for s in stages])
    ax.set_ylabel("близость подписи к домену живописи SemArt (косинус)")
    ax.set_title("Прогрессия доменного сдвига при дообучении BLIP\n"
                 "с числом эпох подписи всё ближе к домену живописи (и дальше от открыток)")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"-> {args.out}")
    print("средние по стадиям:", {s: round(m, 3) for s, m in zip(stages, means)})


if __name__ == "__main__":
    main()
