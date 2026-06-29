"""plot_cosine_distribution.py — распределение косинусного сходства пар.

Гистограмма косинуса для совпадающих (изображение и его описание) и случайных пар.
Данные — кэш эмбеддингов analysis (clip_img, mclip_text). Серо-синяя палитра.

Использование:
    python scripts/plot_cosine_distribution.py \
        --cache data/eval/embeddings/analysis --out docs/figures/fig_cosine_distribution.png
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
# Весь текст — Times New Roman (с serif-фолбэком, если шрифта нет).
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]

import numpy as np

C_MATCHED = "#1f5f8b"   # стальной синий — совпадающие пары
C_RANDOM = "#9fb3c5"    # светлый серо-голубой — случайные пары


def load(cache):
    d = np.load(cache + ".npz")
    meta = [json.loads(l) for l in open(cache + ".meta.jsonl", encoding="utf-8")]
    return {k: d[k] for k in d.files}, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/eval/embeddings/analysis")
    ap.add_argument("--out", default="docs/figures/fig_cosine_distribution.png")
    args = ap.parse_args()

    emb, meta = load(args.cache)
    idx = [i for i, m in enumerate(meta) if m["has_text"]]
    img, txt = emb["clip_img"][idx], emb["mclip_text"][idx]
    n = len(idx)

    matched = (img * txt).sum(axis=1)
    rng = np.random.RandomState(42)
    perm = rng.permutation(n)
    same = perm == np.arange(n)
    perm[same] = (perm[same] + 1) % n  # гарантируем j != i
    mismatched = (img * txt[perm]).sum(axis=1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(mismatched, bins=30, alpha=0.7, color=C_RANDOM, edgecolor="#6b7f91",
            linewidth=0.4, label=f"случайные (μ={mismatched.mean():.3f})")
    ax.hist(matched, bins=30, alpha=0.75, color=C_MATCHED, edgecolor="#14425f",
            linewidth=0.4, label=f"совпадающие (μ={matched.mean():.3f})")
    ax.set_xlabel("косинусное сходство (CLIPScore)")
    ax.set_ylabel("частота")
    ax.set_title("Распределение косинусного сходства для совпадающих\n"
                 "и случайных пар «изображение – описание»", fontsize=11)
    ax.legend()
    ax.grid(axis="y", alpha=0.2)

    fig.tight_layout()
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    plt.close()
    print("->", args.out)
    print(f"n={n}  matched μ={matched.mean():.3f}  random μ={mismatched.mean():.3f}")


if __name__ == "__main__":
    main()
