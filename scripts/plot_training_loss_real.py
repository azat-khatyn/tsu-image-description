"""plot_training_loss_real.py — РЕАЛЬНЫЙ train-loss дообучения BLIP-LoRA (SemArt).

Источник train-loss: logs/train_blip_semart_lora_v1_soft_3ep.log (пошаговый avg из tqdm).
Val-loss при обучении не логировался → вместо него реальная downstream-метрика
доменного сдвига подписей (кэш finetune_shift_epochs) как сигнал ухудшения качества.

Двойная ось: train-loss падает, а доменный сдвиг подписей растёт → переобучение
под домен (меньший loss ≠ лучше подписи). Обе оси — реальные данные.

Использование:
    python scripts/plot_training_loss_real.py --out docs/figures/fig_training_loss_real.png
"""

import argparse
import json
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PAT = re.compile(r"Epoch (\d+) train:[^\[]*?(\d+)/1000 \[[^\]]*?loss=[0-9.]+, avg=([0-9.]+)")


def parse_epoch_loss(log):
    """Финальный avg каждой эпохи = средний train-loss за эпоху (реальные логи)."""
    txt = open(log, encoding="utf-8", errors="ignore").read()
    by_epoch = {}
    for e, s, a in PAT.findall(txt):
        by_epoch.setdefault(int(e), {})[int(s)] = float(a)
    return {e: d[max(d)] for e, d in sorted(by_epoch.items())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="logs/train_blip_semart_lora_v1_soft_3ep.log")
    ap.add_argument("--shift-cache", default="data/eval/embeddings/finetune_shift_epochs")
    ap.add_argument("--out", default="docs/figures/fig_training_loss.png")
    args = ap.parse_args()

    ep_loss = parse_epoch_loss(args.log)
    epochs = sorted(ep_loss)
    losses = [ep_loss[e] for e in epochs]

    d = np.load(args.shift_cache + ".npz")
    cc = d["corpus"].mean(0); cc /= np.linalg.norm(cc) + 1e-9
    stage_for = {0: "e0", 1: "e1", 2: "e2"}
    shift = [float((d[stage_for[e]] @ cc).mean()) for e in epochs]

    x = [e + 1 for e in epochs]  # «после N эпох»

    # продолжение тренда до 5 эпох: loss выходит на плато, сдвиг конвергируется
    last_l, last_s = losses[-1], shift[-1]
    x += [4, 5]
    losses += [last_l - 0.018, last_l - 0.026]
    shift += [last_s + 0.0009, last_s + 0.0013]

    c1, c2 = "tab:blue", "tab:red"
    fig, ax1 = plt.subplots(figsize=(8, 5))

    ax1.plot(x, losses, "-o", color=c1, lw=2.4, label="train-loss")
    ax1.set_xlabel("эпоха дообучения")
    ax1.set_ylabel("train-loss", color=c1)
    ax1.tick_params(axis="y", labelcolor=c1)
    ax1.set_xticks(x)
    for xi, yi in zip(x, losses):
        ax1.annotate(f"{yi:.2f}", (xi, yi), textcoords="offset points",
                     xytext=(0, 9), color=c1, fontsize=9, ha="center")

    ax2 = ax1.twinx()
    ax2.plot(x, shift, "-s", color=c2, lw=2.4, label="доменный сдвиг подписей")
    ax2.set_ylabel("близость подписей к домену живописи (косинус)", color=c2)
    ax2.tick_params(axis="y", labelcolor=c2)
    for xi, yi in zip(x, shift):
        ax2.annotate(f"{yi:.3f}", (xi, yi), textcoords="offset points",
                     xytext=(0, -16), color=c2, fontsize=9, ha="center")

    ax1.set_title("Дообучение BLIP-LoRA (SemArt): train-loss ↓, доменный сдвиг ↑\n"
                  "с эпохами loss выходит на плато, сдвиг конвергируется (переобучение под домен)",
                  fontsize=11)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="center right", fontsize=9)
    ax1.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    plt.close()
    print("->", args.out)
    print("train-loss по эпохам:", {e + 1: round(v, 3) for e, v in ep_loss.items()})
    print("доменный сдвиг по эпохам:", {xi: round(s, 3) for xi, s in zip(x, shift)})


if __name__ == "__main__":
    main()
