"""run_on_new_postcards.py — quick pipeline run on a few specific images.

Использует proposed-архитектуру (BLIP-large + MarianMT + SigLIP) и выводит описания
в обоих режимах (retrieval = caption_only, display = full -theme -mood).

Запускает BLIP/MarianMT/SigLIP один раз на изображение, потом собирает оба варианта
описания из одного и того же caption_ru + metadata.
"""

import json
from pathlib import Path

from tsu_image_description.pipeline import ArchiveDescriptionPipeline
from tsu_image_description.description_builder import DescriptionBuilder


IMAGES = [
    "data/eval/images/postcard_21.jpg",
    "data/eval/images/postcard_22.jpg",
    "data/eval/images/postcard_23.jpg",
]


def main():
    # Один пайплайн с display-builder'ом; retrieval-описание собираем отдельно
    # из того же caption_ru через второй builder. Экономит ~50% времени.
    pipeline = ArchiveDescriptionPipeline(
        model_path="Salesforce/blip-image-captioning-large",
        builder_kwargs={
            "template_mode": "full",
            "include_theme": False,
            "include_mood": False,
        },
    )
    retrieval_builder = DescriptionBuilder(template_mode="caption_only")

    results = []
    for img in IMAGES:
        if not Path(img).exists():
            print(f"[WARN] Not found: {img}")
            continue
        print(f"\n=== {Path(img).name} ===")
        result = pipeline.run(img)  # display-mode description in result['archive_description']
        display_desc = result["archive_description"]

        # rebuild retrieval-mode description from same intermediates
        retrieval_result = retrieval_builder.build(result)
        retrieval_desc = retrieval_result["archive_description"]

        print(f"  caption_en:    {result['caption']['en']}")
        print(f"  caption_ru:    {result['caption']['ru']}")
        print(f"  image_type:    {result['metadata']['image_type']['label']} "
              f"({result['metadata']['image_type']['score']:.2f}, "
              f"confident={result['metadata']['image_type']['confident']})")
        print(f"  style:         {result['metadata']['style']['label']} "
              f"({result['metadata']['style']['score']:.2f}, "
              f"confident={result['metadata']['style']['confident']})")
        print(f"  theme:         {result['inference'].get('theme')}")
        print(f"  mood:          {result['inference'].get('mood')}")
        print(f"  tags:          {result['metadata'].get('tags', [])}")
        print()
        print(f"  >> RETRIEVAL: {retrieval_desc}")
        print(f"  >> DISPLAY:   {display_desc}")

        results.append({
            "image_path": img,
            "caption_en": result["caption"]["en"],
            "caption_ru": result["caption"]["ru"],
            "metadata": result["metadata"],
            "inference": result["inference"],
            "retrieval_description": retrieval_desc,
            "display_description": display_desc,
        })

    out_path = Path("data/eval/results/final/new_postcards_21_22_23.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[INFO] Saved to {out_path}")


if __name__ == "__main__":
    main()
