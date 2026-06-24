"""finetune_shift_export.py — кэш для фигуры доменного сдвига при дообучении BLIP.

Генерирует подписи БАЗОВОЙ и ДООБУЧЕННОЙ (LoRA) модели на одних и тех же тестовых
открытках, плюс берёт подписи корпуса-донора (SemArt, TITLE) как «целевой домен».
Эмбеддит все подписи CLIP-текст-энкодером (EN) и сохраняет кэш для plot-скрипта.

Использование:
    python scripts/finetune_shift_export.py \
        --finetuned models/blip_semart_lora/v1_soft_epoch_2 \
        --n-per-source 40 --n-corpus 120 \
        --out data/eval/embeddings/finetune_shift
"""

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

from tsu_image_description.models import CaptionGenerator, get_device
from tsu_image_description.clip_scorer import CLIPScorer

ROOT = Path(__file__).resolve().parent.parent
SEMART_CSV = ROOT / "data" / "semart" / "SemArt" / "semart_train.csv"


def sample_images(testset, n_per_source, seed):
    by_src = defaultdict(list)
    for line in open(testset, encoding="utf-8"):
        r = json.loads(line)
        if r["source"] in ("neb_wwii", "neb_diverse", "nypl_curated"):
            by_src[r["source"]].append(r["image_path"])
    rng = random.Random(seed)
    picked = []
    for src, paths in sorted(by_src.items()):
        paths = [p for p in paths if Path(p).is_file()]
        picked += [(src, p) for p in rng.sample(paths, min(n_per_source, len(paths)))]
    return picked


def semart_titles(n, seed):
    rows = []
    with open(SEMART_CSV, encoding="latin-1") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            t = (row.get("TITLE") or "").strip()
            if 3 <= len(t) <= 120:
                rows.append(t)
    return random.Random(seed).sample(rows, min(n, len(rows)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--testset", default="data/eval/testset.jsonl")
    ap.add_argument("--finetuned", default="models/blip_semart_lora/v1_soft_epoch_2")
    ap.add_argument("--n-per-source", type=int, default=40)
    ap.add_argument("--n-corpus", type=int, default=120)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/eval/embeddings/finetune_shift")
    args = ap.parse_args()

    imgs = sample_images(args.testset, args.n_per_source, args.seed)
    print(f"[export] изображений: {len(imgs)}")

    # 1) базовые подписи
    base_gen = CaptionGenerator(model_path=None)
    base_caps = [base_gen.generate(p) for _, p in imgs]
    del base_gen

    # 2) дообученные подписи (LoRA) на тех же изображениях
    ft_gen = CaptionGenerator(model_path=args.finetuned)
    ft_caps = [ft_gen.generate(p) for _, p in imgs]
    del ft_gen

    # 3) корпус-донор (целевой домен)
    corpus = semart_titles(args.n_corpus, args.seed)
    print(f"[export] подписи: base={len(base_caps)} ft={len(ft_caps)} corpus={len(corpus)}")

    # 4) эмбеддинги CLIP-текст (EN)
    scorer = CLIPScorer(device=get_device())
    def emb(texts):
        return np.stack([scorer.encode_text(t or ".", "en") for t in texts])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(out) + ".npz",
             base=emb(base_caps), ft=emb(ft_caps), corpus=emb(corpus))
    meta = {
        "finetuned": args.finetuned,
        "images": [{"source": s, "image_path": p,
                    "base": b, "ft": f}
                   for (s, p), b, f in zip(imgs, base_caps, ft_caps)],
        "corpus": corpus,
    }
    json.dump(meta, open(str(out) + ".meta.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"[export] -> {out}.npz / {out}.meta.json")


if __name__ == "__main__":
    main()
