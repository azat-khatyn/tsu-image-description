import argparse
import json

from tsu_image_description.pipeline import ArchiveDescriptionPipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to image")
    parser.add_argument("--use-ocr", action="store_true", help="Включить стадию OCR")
    parser.add_argument("--use-llm", action="store_true", help="Включить LLM-редактор")
    parser.add_argument("--use-clipscore", action="store_true", help="Считать CLIPScore описания")
    args = parser.parse_args()

    pipeline = ArchiveDescriptionPipeline(
        use_ocr=args.use_ocr,
        use_llm_rewriter=args.use_llm,
        use_clipscore=args.use_clipscore,
    )
    result = pipeline.run(args.image)

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
