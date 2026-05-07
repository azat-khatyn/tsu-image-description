import json
import random
import os

INPUT_PATH = "data/nypl/capfilt_filtered_v2.jsonl"

TRAIN_OUT = "data/nypl/train_v2.json"
VAL_OUT = "data/nypl/val_v2.json"

VAL_SIZE = 0.1

data = []

# counters
skipped_url = 0
skipped_missing = 0
kept = 0


with open(INPUT_PATH, "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)

        image_path = item["image"]

        # пропускаем URL
        if isinstance(image_path, str) and image_path.startswith("http"):
            skipped_url += 1
            continue

        # пропускаем отсутствующие файлы
        if not os.path.exists(image_path):
            skipped_missing += 1
            continue

        # берём caption
        caption = item.get("text") or item.get("caption")

        if not caption:
            continue

        kept += 1

        data.append({
            "image": image_path,
            "caption": caption
        })
random.shuffle(data)

split_idx = int(len(data) * (1 - VAL_SIZE))

train = data[:split_idx]
val = data[split_idx:]

with open(TRAIN_OUT, "w", encoding="utf-8") as f:
    json.dump(train, f, ensure_ascii=False, indent=2)

with open(VAL_OUT, "w", encoding="utf-8") as f:
    json.dump(val, f, ensure_ascii=False, indent=2)

print(len(train), len(val))
print(f"Kept: {kept}")
print(f"Skipped URL: {skipped_url}")
print(f"Skipped missing: {skipped_missing}")
