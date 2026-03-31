import argparse
import json

from tsu_image_description.pipeline import ArchiveDescriptionPipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to image")
    args = parser.parse_args()

    pipeline = ArchiveDescriptionPipeline()
    result = pipeline.run(args.image)

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
