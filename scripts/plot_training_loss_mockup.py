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
    # val: падает, минимум ~эпоха 2.5, затем лёгкий рост (переобучение)
    val = 1.55 + 2.9 * np.exp(-1.4 * ep) + 0.045 * np.clip(ep - 2.5, 0, None)

    trained = ep <= 3  # реально обученный диапазон
    plt.figure(figsize=(7.5, 5))
    # сплошная — обученный диапазон, пунктир — иллюстративная экстраполяция
    plt.plot(ep[trained], train[trained], color="tab:blue", lw=2.2, label="train loss")
    plt.plot(ep[~trained], train[~trained], color="tab:blue", lw=2.0, ls="--", alpha=0.8)
    plt.plot(ep[trained], val[trained], color="tab:orange", lw=2.2, label="val loss")
    plt.plot(ep[~trained], val[~trained], color="tab:orange", lw=2.0, ls="--", alpha=0.8)

    plt.axvspan(2, 3, color="green", alpha=0.10)
    plt.axvline(2.5, color="green", ls=":", lw=1.3)
    plt.annotate("достаточно 2–3 эпох\n(дальше val-loss растёт — переобучение)",
                 xy=(2.5, val[np.argmin(np.abs(ep - 2.5))]),
                 xytext=(4.2, 3.0), fontsize=10,
                 arrowprops=dict(arrowstyle="->", color="green"))
    plt.text(3.05, 0.55, "← обучали   |   иллюстративная экстраполяция →",
             transform=plt.gca().get_xaxis_transform(), fontsize=8, color="gray")

    plt.xlabel("эпоха"); plt.ylabel("loss")
    plt.title("Динамика loss при дообучении BLIP-LoRA (иллюстративная схема)")
    plt.legend(loc="upper right")
    plt.ylim(1.0, 4.8)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=200, bbox_inches="tight"); plt.close()
    print("->", out)


if __name__ == "__main__":
    main()
