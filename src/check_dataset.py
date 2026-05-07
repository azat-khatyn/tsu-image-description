from __future__ import annotations

from tsu_image_description.config import get_default_config
from tsu_image_description.data.dataset import MultiSourceArchiveDataset
from tsu_image_description.data.transforms import build_eval_transforms


def main() -> None:
    config = get_default_config()
    dataset = MultiSourceArchiveDataset(
        config=config,
        split="train",
        image_transform=build_eval_transforms(config.training.image_size),
        text_language="ru",
    )

    print(f"Loaded samples: {len(dataset)}")
    item = dataset[0]

    print("Sample ID:", item["sample_id"])
    print("Image path:", item["image_path"])
    print("Text target:", item["target_text"])
    print("Metadata:", item["metadata"])


if __name__ == "__main__":
    main()
