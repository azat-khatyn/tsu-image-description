from __future__ import annotations

from pathlib import Path

from tsu_image_description.config import get_default_config
from tsu_image_description.utils.debug import print_header


def main() -> None:
    config = get_default_config()

    print_header("Project state check")

    paths_to_check = [
        "data/manifests/local_gold.jsonl",
        "data/manifests/europeana.jsonl",
        "data/manifests/wikiart.jsonl",
        "data/manifests/artemis.jsonl",
    ]

    for path_str in paths_to_check:
        path = Path(path_str)
        print(f"{path_str}: {'OK' if path.exists() else 'MISSING'}")

    print("\nConfigured dataset sources:")
    for source in config.dataset_sources:
        print(
            f"- {source.name}: manifest={source.manifest_path}, "
            f"image_root={source.image_root}, enabled={source.enabled}"
        )


if __name__ == "__main__":
    main()
