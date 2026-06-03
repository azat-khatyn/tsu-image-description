import argparse
import json
import sys

sys.path.insert(0, "src")

from tsu_image_description.pipeline import ArchiveDescriptionPipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to image")
    parser.add_argument("--use-ocr", action="store_true", help="Включить стадию OCR")
    parser.add_argument("--use-llm", action="store_true", help="Включить LLM-редактор")
    args = parser.parse_args()

    pipeline = ArchiveDescriptionPipeline(
        use_ocr=args.use_ocr,
        use_llm_rewriter=args.use_llm,
    )
    result = pipeline.run(args.image)

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
