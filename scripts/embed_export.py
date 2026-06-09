"""embed_export.py — выгрузка эмбеддингов для анализа, агностично к датасету.

Вход — манифест (JSONL), одна запись на изображение:
    {"image_path": "...", "source": "neb|nypl|rgb|semart",
     "theme": <gt|null>, "technique": <gt|null>, "text_ru": <текст|null>}

Считает через те же src-энкодеры, что eval/прод (согласованность):
  - SigLIP image features        (SigLIPZeroShotClassifier.encode_image)
  - CLIP ViT-B-32 image features (CLIPScorer.encode_image)
  - M-CLIP text features          (CLIPScorer.encode_text, только если есть text_ru)
  - предсказанные theme/technique + уверенность (MetadataExtractor.extract)

Выход:
  <out>.npz           — выровненные по индексу массивы (siglip, clip_img, mclip_text)
  <out>.meta.jsonl    — построчные метаданные (source, gt/pred-метки, has_text)

Перепрогон на обновлённом датасете = другой --manifest. Ничего не хардкодится.

Использование:
    python scripts/embed_export.py --manifest data/eval/testset.jsonl \
        --out data/eval/embeddings/testset
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from tsu_image_description.metadata_extractor import MetadataExtractor
from tsu_image_description.clip_scorer import CLIPScorer


def load_manifest(path):
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def _field(metadata, axis):
    f = metadata.get(axis) or {}
    return f.get("label"), f.get("score"), bool(f.get("confident"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="JSONL манифест набора")
    ap.add_argument("--out", required=True, help="Префикс выходных файлов (без расширения)")
    ap.add_argument("--taxonomy", default="archival_v2")
    ap.add_argument("--limit", type=int, default=None, help="Ограничить число изображений (отладка)")
    ap.add_argument("--no-text", action="store_true", help="Не считать M-CLIP текст-эмбеддинги")
    args = ap.parse_args()

    items = load_manifest(args.manifest)
    if args.limit:
        items = items[: args.limit]
    print(f"[embed_export] {len(items)} записей из {args.manifest}")

    extractor = MetadataExtractor(taxonomy_version=args.taxonomy)
    clip = CLIPScorer(load_ru=not args.no_text)

    siglip_vecs, clip_vecs, mclip_vecs, meta = [], [], [], []

    for i, it in enumerate(items):
        path = it["image_path"]
        if not Path(path).is_file():
            print(f"  [skip] нет файла: {path}")
            continue
        try:
            image = Image.open(path).convert("RGB")
        except Exception as e:
            print(f"  [skip] не открылось {path}: {e}")
            continue

        siglip_vecs.append(np.asarray(extractor.classifier.encode_image(image), dtype=np.float32))
        clip_vecs.append(np.asarray(clip.encode_image(path), dtype=np.float32))

        text = (it.get("text_ru") or "").strip()
        if text and not args.no_text:
            mclip_vecs.append(np.asarray(clip.encode_text(text, "ru"), dtype=np.float32))
            has_text = True
        else:
            mclip_vecs.append(None)
            has_text = False

        md = extractor.extract(path)
        it_label, it_score, it_conf = _field(md, "image_type")
        st_label, st_score, st_conf = _field(md, "style")
        th_label, th_score, th_conf = _field(md, "theme")

        meta.append({
            "image_path": path,
            "source": it.get("source"),
            "gt_theme": it.get("theme"),
            "gt_technique": it.get("technique"),
            "has_text": has_text,
            "pred_image_type": it_label, "pred_image_type_score": it_score, "pred_image_type_confident": it_conf,
            "pred_technique": st_label, "pred_technique_score": st_score, "pred_technique_confident": st_conf,
            "pred_theme": th_label, "pred_theme_score": th_score, "pred_theme_confident": th_conf,
        })
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(items)}")

    siglip = np.stack(siglip_vecs)
    clip_img = np.stack(clip_vecs)
    dim_t = next((v.shape[0] for v in mclip_vecs if v is not None), clip_img.shape[1])
    mclip_text = np.stack([
        v if v is not None else np.full(dim_t, np.nan, dtype=np.float32) for v in mclip_vecs
    ])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out.with_suffix(".npz"), siglip=siglip, clip_img=clip_img, mclip_text=mclip_text)
    with open(out.with_suffix(".meta.jsonl"), "w", encoding="utf-8") as f:
        for m in meta:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    print(f"[embed_export] сохранено: {out.with_suffix('.npz')} "
          f"(siglip {siglip.shape}, clip {clip_img.shape}, text {mclip_text.shape})")
    print(f"[embed_export] метаданные: {out.with_suffix('.meta.jsonl')} ({len(meta)} строк)")


if __name__ == "__main__":
    main()
