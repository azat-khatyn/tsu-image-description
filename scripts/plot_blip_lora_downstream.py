"""plot_blip_lora_downstream.py — прикладные метрики дообучения BLIP-LoRA по эпохам.

Соседний к графику потерь рисунок: показывает, что снижение loss не переносится
в downstream-метрики. Источник — metrics_lora_epoch_{0..4}.json (полный пул-656,
без LLM). CLIPScore (archive RU) усредняется по per_item; R@k берётся из
summary.retrieval.t2i (поиск «описание -> изображение» в едином пуле).

Использование:
    python scripts/plot_blip_lora_downstream.py \
        --results-dir data/eval/results --out docs/figures/fig_blip_lora_downstream.png
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

EPOCHS = [0, 1, 2, 3, 4]  # файлы; на оси отображаем как 1..5


def _mean_clipscore(metrics: dict) -> float:
    vals = [it["scores"]["CLIPScore_RU_archive_ru"]
            for it in metrics.get("per_item", [])
            if it.get("scores", {}).get("CLIPScore_RU_archive_ru") is not None]
    return sum(vals) / len(vals) if vals else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="data/eval/results")
    ap.add_argument("--out", default="docs/figures/fig_blip_lora_downstream.png")
    args = ap.parse_args()
    rd = Path(args.results_dir)

    xs, clip, r1, r5, r10 = [], [], [], [], []

    # baseline (исходная BLIP-large без дообучения) — точка x=0
    base = rd / "metrics_testset_nollm.json"
    if base.is_file():
        m = json.loads(base.read_text(encoding="utf-8"))
        t2i = m["summary"]["retrieval"]["t2i"]
        xs.append(0)
        clip.append(_mean_clipscore(m))
        r1.append(t2i["R@1"]); r5.append(t2i["R@5"]); r10.append(t2i["R@10"])
        print(f"baseline: CLIPScore={clip[-1]:.3f} R@1={r1[-1]:.3f} "
              f"R@5={r5[-1]:.3f} R@10={r10[-1]:.3f}")

    for n in EPOCHS:
        f = rd / f"metrics_lora_epoch_{n}.json"
        if not f.is_file():
            print(f"[skip] нет {f}")
            continue
        m = json.loads(f.read_text(encoding="utf-8"))
        t2i = m["summary"]["retrieval"]["t2i"]
        xs.append(n + 1)
        clip.append(_mean_clipscore(m))
        r1.append(t2i["R@1"]); r5.append(t2i["R@5"]); r10.append(t2i["R@10"])
        print(f"эпоха {n+1}: CLIPScore={clip[-1]:.3f} R@1={r1[-1]:.3f} "
              f"R@5={r5[-1]:.3f} R@10={r10[-1]:.3f}")

    xticklabels = ["база" if x == 0 else str(x) for x in xs]

    if not xs:
        print("Нет данных — дождись завершения eval_lora_epochs.sh")
        return

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5))

    a1.plot(xs, clip, "-o", color="#08519c", lw=2, ms=7)
    for x, y in zip(xs, clip):
        a1.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=9, color="#08519c")
    a1.set_xticks(xs); a1.set_xticklabels(xticklabels); a1.set_xlabel("эпоха")
    a1.set_ylabel("CLIPScore (archive RU)")
    a1.set_title("CLIPScore по эпохам", fontfamily="serif")
    a1.grid(alpha=0.25)

    for ys, lab, c, mk in ((r1, "R@1", "#08519c", "o"),
                           (r5, "R@5", "#3182bd", "s"),
                           (r10, "R@10", "#9ecae1", "^")):
        a2.plot(xs, ys, "-", marker=mk, color=c, lw=2, ms=7, label=lab)
    a2.set_xticks(xs); a2.set_xticklabels(xticklabels); a2.set_xlabel("эпоха")
    a2.set_ylabel("Recall@k (t2i, пул-656)")
    a2.set_ylim(0, 1.0)
    a2.set_title("Recall@k по эпохам", fontfamily="serif")
    a2.legend(loc="best", fontsize=9)
    a2.grid(alpha=0.25)

    fig.suptitle("Б) Прикладные метрики при дообучении BLIP-LoRA (полный пул-656)",
                 fontsize=12, fontfamily="serif")
    fig.tight_layout()
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    plt.close()
    print("->", args.out)


if __name__ == "__main__":
    main()
