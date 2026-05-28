"""prepare_artcap.py - скачать ArtCap с HuggingFace и привести к формату обучения BLIP.

Скачивает 5unnySunny/artcapDataset (~3606 изображений и подписей).
Каждое изображение кладётся в data/artcap/images/, подписи берутся из
prompts/*.txt. Делим 90/10 train/val с seed=42 для воспроизводимости.

Выходные файлы:
  data/artcap/images/{000000..003605}.png
  data/artcap/train.json   # [{image_path, caption}, ...]
  data/artcap/val.json
  data/artcap/manifest.json   # полный набор для проверки

Использование:
  python scripts/prepare_artcap.py
"""

import json
import random
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download


REPO_ID = "5unnySunny/artcapDataset"
TARGET_DIR = Path("data/artcap")
SEED = 42
VAL_RATIO = 0.1


def main():
    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] Downloading dataset from HF: {REPO_ID}")
    print(f"      (~3 GB, includes 3606 images + prompts)")
    print(f"      Target: {TARGET_DIR}")
    print(f"      This will use the existing HF token / cache.")

    local_path = snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=str(TARGET_DIR / "_raw"),
        # пропускаем карты ControlNet — для обучения BLIP не нужны
        ignore_patterns=["conditions/*"],
    )
    print(f"      Downloaded to: {local_path}")
    print()

    print(f"[2/4] Loading metadata + building (image, caption) pairs")
    metadata_path = TARGET_DIR / "_raw" / "metadata.json"
    with open(metadata_path) as f:
        items = json.load(f)
    print(f"      Loaded {len(items)} item-records")

    pairs = []
    missing_images = 0
    missing_prompts = 0
    for item in items:
        img_rel = item["image"]      # 'images/000123.png'
        prompt_rel = item["prompt"]  # 'prompts/000123.txt'

        img_abs = TARGET_DIR / "_raw" / img_rel
        prompt_abs = TARGET_DIR / "_raw" / prompt_rel

        if not img_abs.exists():
            missing_images += 1
            continue
        if not prompt_abs.exists():
            missing_prompts += 1
            continue

        caption = prompt_abs.read_text(encoding="utf-8").strip()
        if not caption:
            continue

        pairs.append({
            "id": item["id"],
            "image_path": str(img_abs),
            "caption": caption,
        })

    print(f"      Successfully paired: {len(pairs)}")
    if missing_images:
        print(f"      Missing images: {missing_images}")
    if missing_prompts:
        print(f"      Missing prompts: {missing_prompts}")
    print()

    print(f"[3/4] Saving manifest + train/val split")
    manifest_path = TARGET_DIR / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(pairs, f, indent=2, ensure_ascii=False)
    print(f"      Manifest: {manifest_path}")

    random.seed(SEED)
    shuffled = pairs.copy()
    random.shuffle(shuffled)
    n_val = int(len(shuffled) * VAL_RATIO)
    val = shuffled[:n_val]
    train = shuffled[n_val:]

    train_path = TARGET_DIR / "train.json"
    val_path = TARGET_DIR / "val.json"
    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train, f, indent=2, ensure_ascii=False)
    with open(val_path, "w", encoding="utf-8") as f:
        json.dump(val, f, indent=2, ensure_ascii=False)
    print(f"      Train: {train_path}  ({len(train)} items)")
    print(f"      Val:   {val_path}    ({len(val)} items)")
    print()

    print(f"[4/4] Sample captions for sanity check:")
    for i in range(min(5, len(train))):
        print(f"      [{i}] {train[i]['caption'][:120]}")
    print()
    print(f"DONE. Ready for BLIP LoRA training.")


if __name__ == "__main__":
    main()
