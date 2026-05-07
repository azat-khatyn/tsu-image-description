from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from tsu_image_description.config import get_default_config
from tsu_image_description.data.dataset_pil import (
    MultiSourceArchivePILDataset,
    collate_pil_batch,
)
from tsu_image_description.modeling.label_space import build_label_space
from tsu_image_description.modeling.multitask_model import (
    ArchiveMultitaskModel,
    build_processor,
)
from tsu_image_description.training.batch_preparation import prepare_blip_batch
from tsu_image_description.utils.debug import print_header


def main() -> None:
    config = get_default_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print_header("1. Loading dataset")
    dataset = MultiSourceArchivePILDataset(
        config=config,
        split="train",
        text_language="ru",
        oversample_by_weight=False,
    )
    print(f"Dataset size: {len(dataset)}")

    sample = dataset[0]
    print("Sample ID:", sample["sample_id"])
    print("Image path:", sample["image_path"])
    print("Target text:", sample["target_text"])
    print("Metadata:", sample["metadata"])

    print_header("2. Building dataloader")
    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_pil_batch,
    )
    batch = next(iter(loader))
    print("Batch keys:", list(batch.keys()))
    print("Batch size:", len(batch["sample_id"]))

    print_header("3. Processor + targets")
    label_space = build_label_space(config)
    processor = build_processor(config)

    prepared = prepare_blip_batch(
        batch=batch,
        processor=processor,
        label_space=label_space,
        max_text_length=config.training.max_text_length,
        device=device,
    )

    for key, value in prepared.items():
        if torch.is_tensor(value):
            print(f"{key}: shape={tuple(value.shape)}, dtype={value.dtype}")
        else:
            print(f"{key}: {type(value)}")

    print_header("4. Model forward pass")
    model = ArchiveMultitaskModel(config, label_space).to(device)
    model.eval()

    with torch.no_grad():
        outputs = model(
            pixel_values=prepared["pixel_values"],
            input_ids=prepared["input_ids"],
            attention_mask=prepared["attention_mask"],
            labels=prepared["labels"],
        )

    print("Caption loss:", None if outputs.caption_loss is None else float(outputs.caption_loss.item()))
    print("Style logits:", tuple(outputs.style_logits.shape))
    print("Period logits:", tuple(outputs.period_logits.shape))
    print("Emotion logits:", tuple(outputs.emotion_logits.shape))
    print("Tag logits:", tuple(outputs.tag_logits.shape))

    print_header("5. Generation test")
    with torch.no_grad():
        generated = model.generate(
            pixel_values=prepared["pixel_values"][:1],
            max_new_tokens=32,
            num_beams=2,
        )

    decoded = processor.tokenizer.batch_decode(
        generated["generated_ids"],
        skip_special_tokens=True,
    )
    print("Generated caption:", decoded[0])

    print("\nSmoke test passed.")


if __name__ == "__main__":
    main()
