"""plot_training_loss_mockup.py — ИЛЛЮСТРАТИВНАЯ схема динамики loss.

ВАЖНО: это НЕ измеренные данные. Реальное дообучение шло 2-3 эпохи и логов
loss в файл не сохраняло. График — схема для презентации: показывает, что
train-loss продолжает медленно падать, а val-loss выходит на плато и далее
растёт (переобучение) → дообучать дольше 2-3 эпох для этой задачи нет смысла.
Сплошная часть — обученный диапазон, пунктир — иллюстративная экстраполяция.

Использование:
    python scripts/plot_training_loss_mockup.py --out docs/figures/fig_training_loss_mockup.png
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/figures/fig_training_loss_mockup.png")
    args = ap.parse_args()

    ep = np.linspace(0, 10, 101)
    # train: экспоненциальное затухание к низкому плато
    train = 1.30 + 3.1 * np.exp(-1.5 * ep)
    # val: падает, минимум ~эпоха 2.5, затем рост (переобучение)
    val = 1.55 + 2.9 * np.exp(-1.6 * ep) + 0.08 * np.clip(ep - 2, 0, None)

    trained = ep <= 3            # реально обученный диапазон
    imin = int(np.argmin(val))   # минимум val-loss = точка ранней остановки

    fig, ax = plt.subplots(figsize=(7.5, 5))
    # сплошная — обученный диапазон, пунктир — иллюстративная экстраполяция
    ax.plot(ep[trained], train[trained], color="tab:blue", lw=2.4, label="train loss")
    ax.plot(ep[trained], val[trained], color="tab:orange", lw=2.4, label="val loss")
    ax.plot(ep[~trained], train[~trained], color="tab:blue", lw=2.0, ls="--", alpha=0.7)
    ax.plot(ep[~trained], val[~trained], color="tab:orange", lw=2.0, ls="--", alpha=0.7)

    # минимум val-loss = ранняя остановка; подпись у верхнего края, не в середине
    ax.axvline(ep[imin], color="green", ls=":", lw=1.5)
    ax.scatter([ep[imin]], [val[imin]], color="green", s=45, zorder=5)
    ax.annotate("минимум val-loss ≈ 2–3 эпохи\n(ранняя остановка)",
                xy=(ep[imin], val[imin]), xytext=(ep[imin] + 0.5, 3.7),
                fontsize=9, color="green", ha="left",
                arrowprops=dict(arrowstyle="->", color="green"))
    # переобучение — у растущего хвоста val, справа
    ax.annotate("переобучение", xy=(8.5, val[85]), xytext=(6.7, val[85] + 0.6),
                fontsize=9, color="tab:orange",
                arrowprops=dict(arrowstyle="->", color="tab:orange"))

    ax.set_xlabel("эпоха"); ax.set_ylabel("loss")
    ax.set_title("Динамика loss при дообучении BLIP-LoRA (иллюстративная схема)\n"
                 "сплошная — обученный диапазон · пунктир — экстраполяция", fontsize=11)
    ax.legend(loc="upper right")
    ax.set_ylim(1.0, 4.8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight"); plt.close()
    print("->", out)


if __name__ == "__main__":
    main()
