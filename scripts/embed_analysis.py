"""embed_analysis.py — 3 визуализации эмбеддингов из кэша embed_export.

Агностично к датасету: всё кейстся от кэша (<cache>.npz + <cache>.meta.jsonl),
который произведён embed_export.py. Перепрогон на новом датасете = тот же скрипт
на новом кэше.

Разрезы (каждый — под конкретный тезис):
  1. domain_shift  — t-SNE SigLIP-фич, цвет = источник: наши открытки (НЭБ/NYPL) и
                     внешний референс SemArt (живопись) занимают разные области
                     -> перенос probe рискован.
  2. cross_modal   — PCA [картинка ; текст], пары соединены: описания садятся рядом
                     со своими картинками -> кросс-модальная привязка (наши данные).
  3. confidence    — t-SNE SigLIP-фич, цвет = уверенность темы: уверенные точки в
                     ядрах кластеров, неуверенные/OOD — на границах.

Использование:
    python scripts/embed_analysis.py --cache data/eval/embeddings/analysis \
        --out-dir docs/figures
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def load(cache):
    d = np.load(cache + ".npz")
    meta = [json.loads(l) for l in open(cache + ".meta.jsonl", encoding="utf-8")]
    return {k: d[k] for k in d.files}, meta


def _tsne2(X, seed=42, perplexity=30):
    n = X.shape[0]
    Xp = PCA(n_components=min(50, X.shape[1]), random_state=seed).fit_transform(X)
    perp = min(perplexity, max(5, n // 4))
    return TSNE(n_components=2, init="pca", random_state=seed, perplexity=perp).fit_transform(Xp)


def _save(fig_path):
    plt.xticks([]); plt.yticks([]); plt.tight_layout()
    plt.savefig(fig_path, dpi=200, bbox_inches="tight"); plt.close()
    print("  ->", fig_path)


_SRC_LABEL = {  # человекочитаемые подписи источников для легенды
    "neb_wwii": "НЭБ, ВОВ", "neb_diverse": "НЭБ, общий",
    "nypl_curated": "NYPL, общий", "semantic_demo": "демо",
    "semart": "SemArt (живопись, референс)",
}


def fig_domain_shift(emb, meta, out):
    Y = _tsne2(emb["siglip"])
    src = np.array([m["source"] for m in meta])
    plt.figure(figsize=(7, 6))
    for s in sorted(set(src)):
        m = src == s
        label = _SRC_LABEL.get(s, s)
        plt.scatter(Y[m, 0], Y[m, 1], s=12, alpha=0.6, label=f"{label} (n={int(m.sum())})")
    plt.legend(); plt.title("Доменный сдвиг: SigLIP image features (t-SNE)")
    _save(out / "fig_domain_shift.png")


def fig_cross_modal(emb, meta, out):
    idx = [i for i, m in enumerate(meta) if m["has_text"]]
    if len(idx) < 5:
        print("  [skip] cross_modal: нет текстовых пар"); return
    img, txt = emb["clip_img"][idx], emb["mclip_text"][idx]
    n = len(idx)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    # Слева: modality gap — картинки и тексты в разных конусах (PCA).
    Z = PCA(n_components=2, random_state=42).fit_transform(np.vstack([img, txt]))
    Zi, Zt = Z[:n], Z[n:]
    for a, b in zip(Zi, Zt):
        ax1.plot([a[0], b[0]], [a[1], b[1]], color="gray", lw=0.3, alpha=0.3)
    ax1.scatter(Zi[:, 0], Zi[:, 1], s=12, c="tab:blue", label="изображение (CLIP)")
    ax1.scatter(Zt[:, 0], Zt[:, 1], s=12, c="tab:red", label="текст (M-CLIP)")
    ax1.legend(); ax1.set_title("Modality gap (PCA)")
    ax1.set_xticks([]); ax1.set_yticks([])

    # Справа: привязка через косинус — совпадающие пары vs случайные.
    matched = (img * txt).sum(axis=1)
    rng = np.random.RandomState(42)
    perm = rng.permutation(n)
    same = perm == np.arange(n)
    perm[same] = (perm[same] + 1) % n  # гарантируем j != i
    mismatched = (img * txt[perm]).sum(axis=1)
    ax2.hist(mismatched, bins=30, alpha=0.6, color="gray",
             label=f"случайные (μ={mismatched.mean():.3f})")
    ax2.hist(matched, bins=30, alpha=0.7, color="tab:green",
             label=f"совпадающие (μ={matched.mean():.3f})")
    ax2.set_xlabel("CLIPScore (косинус)"); ax2.set_ylabel("частота")
    ax2.legend(); ax2.set_title("Привязка: совпадающие vs случайные пары")

    plt.tight_layout()
    plt.savefig(out / "fig_cross_modal.png", dpi=200, bbox_inches="tight"); plt.close()
    print("  ->", out / "fig_cross_modal.png")


def fig_confidence(emb, meta, out):
    Y = _tsne2(emb["siglip"])
    score = np.array([(m.get("pred_theme_score") or 0.0) for m in meta])
    plt.figure(figsize=(7.5, 6))
    sc = plt.scatter(Y[:, 0], Y[:, 1], s=12, c=score, cmap="viridis", alpha=0.75)
    plt.colorbar(sc, label="уверенность темы (score)")
    plt.title("Геометрия уверенности (SigLIP t-SNE)")
    _save(out / "fig_confidence.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, help="Префикс кэша (без расширения)")
    ap.add_argument("--out-dir", default="docs/figures")
    args = ap.parse_args()

    emb, meta = load(args.cache)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    print(f"[embed_analysis] {len(meta)} точек; фигуры -> {out}")

    fig_domain_shift(emb, meta, out)
    fig_cross_modal(emb, meta, out)
    fig_confidence(emb, meta, out)


if __name__ == "__main__":
    main()
