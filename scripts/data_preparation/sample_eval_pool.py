"""sample_eval_pool.py — собрать дополнительный пул NYPL для оценки.

Семплирует случайные изображения NYPL, исключая те, что присутствуют в обучающих
сплитах (`data/nypl/splits/train_*.json`, `val_*.json`). Это нужно, чтобы оценка
не пересекалась с обучающей выборкой.

Выход - JSONL-файл по одной записи на изображение, без `reference_short_ru`
(у NYPL нет разметки; используется только как пул для поиска).

    python scripts/sample_eval_pool.py --n 200 --seed 42
"""

import argparse
import json
import os
import random
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=200, help="how many images to sample")
    p.add_argument("--seed", type=int, default=42, help="random seed")
    p.add_argument("--images-dir", type=str, default="data/nypl/images")
    p.add_argument(
        "--splits-dir",
        type=str,
        default="data/nypl/splits",
        help="excludes images used in train_*.json / val_*.json from these splits",
    )
    p.add_argument(
        "--output",
        type=str,
        default="data/eval/pool_nypl.jsonl",
    )
    return p.parse_args()


def collect_excluded_filenames(splits_dir: Path) -> set:
    """Прочитать все train_*.json и val_*.json в splits_dir и вернуть имена файлов."""
    excluded = set()
    if not splits_dir.exists():
        print(f"[WARN] splits dir {splits_dir} not found — no exclusions applied")
        return excluded

    for split_file in sorted(splits_dir.glob("*.json")):
        try:
            with open(split_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"[WARN] could not read {split_file}: {e}")
            continue

        for item in data:
            img = item.get("image", "")
            if img:
                excluded.add(os.path.basename(img))

        print(f"  read {split_file.name}: {len(data)} entries")

    return excluded


def main():
    args = parse_args()

    images_dir = Path(args.images_dir)
    splits_dir = Path(args.splits_dir)
    output_path = Path(args.output)

    print(f"[INFO] images_dir: {images_dir}")
    print(f"[INFO] splits_dir: {splits_dir}")
    print(f"[INFO] output:     {output_path}")
    print(f"[INFO] n:          {args.n}")
    print(f"[INFO] seed:       {args.seed}\n")

    # 1. имена файлов для исключения из обучающих сплитов
    print("[INFO] Reading splits to determine excluded filenames...")
    excluded = collect_excluded_filenames(splits_dir)
    print(f"[INFO] Total excluded filenames: {len(excluded)}\n")

    # 2. все доступные изображения
    all_images = sorted(
        f.name for f in images_dir.iterdir()
        if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    print(f"[INFO] Total images in {images_dir}: {len(all_images)}")

    # 3. кандидаты = доступные минус исключённые
    candidates = [name for name in all_images if name not in excluded]
    print(f"[INFO] Candidates (not in splits): {len(candidates)}")

    if len(candidates) < args.n:
        raise SystemExit(
            f"[ERROR] only {len(candidates)} candidates, asked for {args.n}"
        )

    # 4. воспроизводимая случайная выборка
    rng = random.Random(args.seed)
    sample = sorted(rng.sample(candidates, args.n))

    # 5. запись JSONL
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for name in sample:
            entry = {
                "image_path": str(images_dir / name),
                "reference_short_ru": None,
                "source": "nypl",
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"\n[INFO] Wrote {len(sample)} entries to {output_path}")
    print(f"[INFO] First 3:")
    for name in sample[:3]:
        print(f"  - {name}")


if __name__ == "__main__":
    main()
