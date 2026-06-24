"""plot_blip_lora_train_val.py — динамика train/val loss при дообучении BLIP-LoRA.

Значения — реальные, из прогона с валидацией (logs/train_semart_val.log):
SemArt train_4k (4000 пар), 5 эпох, lr=1e-4, batch=8. Цель рисунка — показать
сходимость и зазор train/val (нет переобучения: val монотонно падает).

Использование:
    python scripts/plot_blip_lora_train_val.py --out docs/figures/fig_blip_lora_train_val.png
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# из logs/train_semart_val.log (эпохи 0-4 -> отображаем 1-5)
EPOCHS = [1, 2, 3, 4, 5]
TRAIN = [6.4646, 6.0773, 6.0081, 5.9493, 5.8963]
VAL = [6.0953, 6.0435, 6.0127, 5.9932, 5.9782]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/figures/fig_blip_lora_train_val.png")
    args = ap.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(EPOCHS, TRAIN, "-o", color="#08519c", lw=2, ms=7, label="train loss")
    ax.plot(EPOCHS, VAL, "-s", color="#e6550d", lw=2, ms=7, label="validation loss")

    # подписи значений только у концов кривых
    for x, y, dy in ((EPOCHS[0], TRAIN[0], 8), (EPOCHS[-1], TRAIN[-1], -14)):
        ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                    xytext=(0, dy), ha="center", color="#08519c", fontsize=9)
    for x, y, dy in ((EPOCHS[0], VAL[0], 8), (EPOCHS[-1], VAL[-1], 8)):
        ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                    xytext=(0, dy), ha="center", color="#e6550d", fontsize=9)

    ax.set_xticks(EPOCHS)
    ax.set_xlabel("эпоха")
    ax.set_ylabel("функция потерь (cross-entropy)")
    ax.set_ylim(5.85, 6.55)
    ax.set_title("Динамика train- и validation-loss при дообучении BLIP-LoRA (SemArt)\n"
                 "val монотонно убывает — признаков переобучения нет", fontsize=11)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print("->", out)


if __name__ == "__main__":
    main()
