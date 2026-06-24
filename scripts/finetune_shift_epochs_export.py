"""finetune_shift_epochs_export.py — кэш для фигуры прогрессии доменного сдвига.

Генерирует подписи базовой модели и SemArt-LoRA на эпохах 0/1/2 на одних и тех же
открытках (тот же seed, что в finetune_shift_export — сопоставимая выборка), плюс
корпус SemArt. Эмбеддит CLIP-текст-энкодером. Для plot-скрипта прогрессии.

Использование:
    python scripts/finetune_shift_epochs_export.py --n-per-source 40 --n-corpus 120 \
        --out data/eval/embeddings/finetune_shift_epochs
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

STAGES = [
    ("база", None),
    ("e0", "models/blip_semart_lora/v1_soft_epoch_0"),
    ("e1", "models/blip_semart_lora/v1_soft_epoch_1"),
    ("e2", "models/blip_semart_lora/v1_soft_epoch_2"),
]


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
    ap.add_argument("--n-per-source", type=int, default=40)
    ap.add_argument("--n-corpus", type=int, default=120)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/eval/embeddings/finetune_shift_epochs")
    args = ap.parse_args()

    imgs = sample_images(args.testset, args.n_per_source, args.seed)
    print(f"[export] изображений: {len(imgs)}")

    caps_by_stage = {}
    for label, path in STAGES:
        gen = CaptionGenerator(model_path=path)
        caps_by_stage[label] = [gen.generate(p) for _, p in imgs]
        del gen
        print(f"[export] {label}: {len(caps_by_stage[label])} подписей")

    corpus = semart_titles(args.n_corpus, args.seed)

    scorer = CLIPScorer(device=get_device())
    def emb(texts):
        return np.stack([scorer.encode_text(t or ".", "en") for t in texts])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    arrays = {label: emb(caps) for label, caps in caps_by_stage.items()}
    arrays["corpus"] = emb(corpus)
    np.savez(str(out) + ".npz", **arrays)
    meta = {
        "stages": [s[0] for s in STAGES],
        "images": [{"source": s, "image_path": p} for s, p in imgs],
        "captions": caps_by_stage,
        "corpus": corpus,
    }
    json.dump(meta, open(str(out) + ".meta.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"[export] -> {out}.npz / {out}.meta.json")


if __name__ == "__main__":
    main()
