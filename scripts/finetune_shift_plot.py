"""finetune_shift_plot.py — фигура доменного сдвига при дообучении BLIP.

Из кэша finetune_shift_export.py строит две панели:
  A. t-SNE подписей: база (открытки) -> дообученная (SemArt-LoRA) -> корпус SemArt.
     Стрелки base->ft показывают направление сдвига к «живописному» домену.
  B. Сдвиг лексики: слова, появляющиеся/исчезающие после дообучения.

Использование:
    python scripts/finetune_shift_plot.py --cache data/eval/embeddings/finetune_shift \
        --out docs/figures/fig_finetune_domain_shift.png
"""

import argparse
import json
import re
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
# Весь текст — Times New Roman (с serif-фолбэком, если шрифта нет).
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]

import numpy as np

STOP = set("a an the of in on at to with and or for from by is are was were be this "
           "that there it its as picture image photo photograph shows show depicts "
           "depicting card postcard view".split())


def words(caps):
    c = Counter()
    for t in caps:
        for w in re.findall(r"[a-z]{3,}", (t or "").lower()):
            if w not in STOP:
                c[w] += 1
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/eval/embeddings/finetune_shift")
    ap.add_argument("--out", default="docs/figures/fig_finetune_domain_shift.png")
    args = ap.parse_args()

    d = np.load(args.cache + ".npz")
    meta = json.load(open(args.cache + ".meta.json", encoding="utf-8"))
    base, ft, corpus = d["base"], d["ft"], d["corpus"]
    n = len(base)

    # количественный сдвиг: близость к центроиду корпуса (косинус)
    cc = corpus.mean(0); cc /= np.linalg.norm(cc) + 1e-9
    base_sim = float((base @ cc).mean())
    ft_sim = float((ft @ cc).mean())

    base_s, ft_s = base @ cc, ft @ cc

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(15, 6))

    # --- A: распределение близости подписей к домену живописи ---
    bins = np.linspace(min(base_s.min(), ft_s.min()), max(base_s.max(), ft_s.max()), 24)
    axL.hist(base_s, bins=bins, alpha=0.6, color="#1f77b4",
             label=f"база BLIP (открытки), μ={base_s.mean():.3f}")
    axL.hist(ft_s, bins=bins, alpha=0.6, color="#d62728",
             label=f"дообуч. SemArt-LoRA, μ={ft_s.mean():.3f}")
    axL.axvline(base_s.mean(), color="#1f77b4", ls="--", lw=1.6)
    axL.axvline(ft_s.mean(), color="#d62728", ls="--", lw=1.6)
    axL.annotate("", xy=(ft_s.mean(), axL.get_ylim()[1] * 0.9),
                 xytext=(base_s.mean(), axL.get_ylim()[1] * 0.9),
                 arrowprops=dict(arrowstyle="-|>", color="black", lw=2))
    axL.set_xlabel("близость подписи к домену живописи SemArt (косинус)")
    axL.set_ylabel("число подписей")
    axL.set_title("Сдвиг распределения подписей к домену живописи\n"
                  f"дообучение смещает вправо (Δμ = +{ft_sim - base_sim:.3f})")
    axL.legend(fontsize=9, loc="upper left")

    # --- B: сдвиг лексики ---
    wb, wf = words([it["base"] for it in meta["images"]]), words([it["ft"] for it in meta["images"]])
    delta = {w: wf[w] / n - wb.get(w, 0) / n for w in set(wb) | set(wf)}
    gained = sorted(delta.items(), key=lambda x: -x[1])[:12]
    lost = sorted(delta.items(), key=lambda x: x[1])[:8]
    rows = list(reversed(gained)) + list(reversed(lost))
    labels = [w for w, _ in rows]
    vals = [v for _, v in rows]
    colors = ["#d62728" if v > 0 else "#1f77b4" for v in vals]
    axR.barh(range(len(rows)), vals, color=colors)
    axR.set_yticks(range(len(rows))); axR.set_yticklabels(labels, fontsize=9)
    axR.axvline(0, color="black", lw=0.8)
    axR.set_xlabel("Δ частоты на подпись (дообуч. − база)")
    axR.set_title("Сдвиг лексики после дообучения\n(красное — появляется, синее — исчезает)")

    fig.tight_layout()
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"-> {args.out}")
    print(f"близость к корпусу: база={base_sim:.3f}, дообуч.={ft_sim:.3f} (Δ={ft_sim-base_sim:+.3f})")
    print("появляются:", ", ".join(f"{w}(+{v*n:.0f})" for w, v in gained[:8]))


if __name__ == "__main__":
    main()
