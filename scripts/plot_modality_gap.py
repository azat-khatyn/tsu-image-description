"""plot_modality_gap.py — межмодальный разрыв изображений и текстов (PCA).

Изображения (CLIP) и тексты (M-CLIP) проецируются в 2D (PCA); пары соединены
тонкими линиями. Модальности занимают разные области — modality gap.
Данные — кэш эмбеддингов analysis (clip_img, mclip_text).

Использование:
    python scripts/plot_modality_gap.py \
        --cache data/eval/embeddings/analysis --out docs/figures/fig_modality_gap.png
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
from sklearn.decomposition import PCA

C_IMG = "#0f6e8c"    # насыщенный теал-синий — изображения
C_TXT = "#cf5c2e"    # насыщенная терракота — тексты
C_LINK = "#aab6c0"   # светло-серые связи между парами


def load(cache):
    d = np.load(cache + ".npz")
    meta = [json.loads(l) for l in open(cache + ".meta.jsonl", encoding="utf-8")]
    return {k: d[k] for k in d.files}, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/eval/embeddings/analysis")
    ap.add_argument("--out", default="docs/figures/fig_modality_gap.png")
    args = ap.parse_args()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    emb, meta = load(args.cache)
    idx = [i for i, m in enumerate(meta) if m["has_text"]]
    img, txt = emb["clip_img"][idx], emb["mclip_text"][idx]
    n = len(idx)

    Z = PCA(n_components=2, random_state=42).fit_transform(np.vstack([img, txt]))
    Zi, Zt = Z[:n], Z[n:]

    fig, ax = plt.subplots(figsize=(7.5, 6))
    for a, b in zip(Zi, Zt):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=C_LINK, lw=0.4, alpha=0.4, zorder=1)
    ax.scatter(Zi[:, 0], Zi[:, 1], s=18, c=C_IMG, label="изображение (CLIP)",
               edgecolor="white", linewidth=0.3, zorder=2)
    ax.scatter(Zt[:, 0], Zt[:, 1], s=18, c=C_TXT, label="текст (M-CLIP)",
               edgecolor="white", linewidth=0.3, zorder=2)
    ax.set_title("Межмодальный разрыв изображений и текстов (PCA)", fontsize=12)
    ax.legend(loc="best")
    ax.set_xticks([]); ax.set_yticks([])

    fig.tight_layout()
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    plt.close()
    print("->", args.out)


if __name__ == "__main__":
    main()
