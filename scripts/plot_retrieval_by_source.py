"""plot_retrieval_by_source.py — Рис. 10: Recall@1 и Recall@5 по выборкам.

Retrieval (t2i: описание-запрос -> изображение) считается per-source ВНУТРИ единого
пула-656 (каноничный прогон metrics_testset_nollm.json, поле per_item.retrieval.rank_t2i).
Единый пул -> сравнение между выборками честное (одна галерея-дистрактор).

Использование:
    python scripts/plot_retrieval_by_source.py \
        --metrics data/eval/results/metrics_testset_nollm.json \
        --out docs/figures/fig_retrieval_by_source.png
"""

import argparse
import json
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ORDER = [
    ("neb_wwii", "НЭБ, ВОВ\n(224)"),
    ("neb_diverse", "НЭБ, общий\n(220)"),
    ("nypl_curated", "NYPL\n(200)"),
    ("semantic_demo", "Демо\n(12)"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", default="data/eval/results/metrics_testset_nollm.json")
    ap.add_argument("--out", default="docs/figures/fig_retrieval_by_source.png")
    args = ap.parse_args()

    per = json.load(open(args.metrics, encoding="utf-8"))["per_item"]
    ranks = defaultdict(list)
    for it in per:
        r = it.get("retrieval", {})
        if r.get("rank_t2i"):
            ranks[it["source"]].append(r["rank_t2i"])

    def rk(s, k):
        v = ranks[s]
        return sum(x <= k for x in v) / len(v)

    labels = [lab for _, lab in ORDER]
    r1 = [rk(s, 1) for s, _ in ORDER]
    r5 = [rk(s, 5) for s, _ in ORDER]

    x = np.arange(len(labels))
    w = 0.36
    fig, ax = plt.subplots(figsize=(9, 5.5))
    b1 = ax.bar(x - w / 2, r1, w, label="Recall@1", color="#9ecae1")
    b2 = ax.bar(x + w / 2, r5, w, label="Recall@5", color="#3182bd")
    for bars in (b1, b2):
        for r in bars:
            ax.annotate(f"{r.get_height():.2f}", (r.get_x() + r.get_width() / 2, r.get_height()),
                        textcoords="offset points", xytext=(0, 3), ha="center", fontsize=9)

    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("доля попаданий (t2i: описание → изображение)")
    ax.set_ylim(0, 1.0)
    ax.set_title("Recall@1 и Recall@5 по выборкам (единый пул-656)\n"
                 "ниже на однородной коллекции НЭБ-ВОВ (похожие сюжеты труднее различимы)",
                 fontsize=11)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    plt.close()
    print("->", args.out)
    for s, lab in ORDER:
        print(f"  {s}: R@1={rk(s,1):.3f} R@5={rk(s,5):.3f} (n={len(ranks[s])})")


if __name__ == "__main__":
    main()
