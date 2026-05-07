from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from tsu_image_description.config import get_default_config
from tsu_image_description.data.dataset_pil import (
    MultiSourceArchivePILDataset,
    collate_pil_batch,
)
from tsu_image_description.modeling.label_space import build_label_space
from tsu_image_description.modeling.losses import ArchiveMultitaskLoss
from tsu_image_description.modeling.multitask_model import (
    ArchiveMultitaskModel,
    build_processor,
)
from tsu_image_description.training.batch_preparation import prepare_blip_batch
from tsu_image_description.training.metrics import (
    masked_accuracy,
    multilabel_f1_micro,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(device_name: str) -> str:
    if device_name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_name


def try_build_dataset(
    config,
    split: str,
) -> Optional[MultiSourceArchivePILDataset]:
    try:
        return MultiSourceArchivePILDataset(
            config=config,
            split=split,
            text_language="ru",
            oversample_by_weight=(split == "train"),
        )
    except RuntimeError:
        return None


def build_dataloader(
    dataset: MultiSourceArchivePILDataset,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_pil_batch,
    )


def run_epoch(
    model: ArchiveMultitaskModel,
    loader: DataLoader,
    processor,
    label_space,
    criterion,
    optimizer,
    scheduler,
    device: str,
    max_text_length: int,
    training: bool = True,
) -> Dict[str, float]:
    if training:
        model.train()
        context_manager = torch.enable_grad()
        progress_desc = "train"
    else:
        model.eval()
        context_manager = torch.no_grad()
        progress_desc = "eval"

    total_loss = 0.0
    total_caption_loss = 0.0
    total_style_acc = 0.0
    total_period_acc = 0.0
    total_emotion_acc = 0.0
    total_tag_f1 = 0.0
    num_steps = 0

    with context_manager:
        for batch in tqdm(loader, desc=progress_desc):
            prepared = prepare_blip_batch(
                batch=batch,
                processor=processor,
                label_space=label_space,
                max_text_length=max_text_length,
                device=device,
            )

            outputs = model(
                pixel_values=prepared["pixel_values"],
                input_ids=prepared["input_ids"],
                attention_mask=prepared["attention_mask"],
                labels=prepared["labels"],
            )

            loss_output = criterion(
                model_output=outputs,
                style_targets=prepared["style_targets"],
                period_targets=prepared["period_targets"],
                emotion_targets=prepared["emotion_targets"],
                tag_targets=prepared["tag_targets"],
            )

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss_output.total_loss.backward()
                optimizer.step()
                scheduler.step()

            style_metrics = masked_accuracy(
                outputs.style_logits,
                prepared["style_targets"],
            )
            period_metrics = masked_accuracy(
                outputs.period_logits,
                prepared["period_targets"],
            )
            emotion_metrics = masked_accuracy(
                outputs.emotion_logits,
                prepared["emotion_targets"],
            )
            tag_f1 = multilabel_f1_micro(
                outputs.tag_logits,
                prepared["tag_targets"],
            )

            total_loss += float(loss_output.total_loss.item())
            total_caption_loss += float(loss_output.caption_loss.item())
            total_style_acc += style_metrics.accuracy
            total_period_acc += period_metrics.accuracy
            total_emotion_acc += emotion_metrics.accuracy
            total_tag_f1 += tag_f1
            num_steps += 1

    if num_steps == 0:
        return {
            "loss": 0.0,
            "caption_loss": 0.0,
            "style_acc": 0.0,
            "period_acc": 0.0,
            "emotion_acc": 0.0,
            "tag_f1": 0.0,
        }

    return {
        "loss": total_loss / num_steps,
        "caption_loss": total_caption_loss / num_steps,
        "style_acc": total_style_acc / num_steps,
        "period_acc": total_period_acc / num_steps,
        "emotion_acc": total_emotion_acc / num_steps,
        "tag_f1": total_tag_f1 / num_steps,
    }


def save_checkpoint(
    path: Path,
    model: ArchiveMultitaskModel,
    epoch: int,
    metrics: Dict[str, float],
    split_name: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "metrics": metrics,
            "split": split_name,
        },
        path,
    )


def main() -> None:
    config = get_default_config()
    set_seed(config.training.seed)

    device = get_device(config.models.device)
    print(f"Using device: {device}")

    train_dataset = try_build_dataset(config, "train")
    valid_dataset = try_build_dataset(config, "valid")

    if train_dataset is None:
        raise RuntimeError(
            "Train dataset is empty or manifests are missing. "
            "Create manifests and images first."
        )

    print(f"Train samples: {len(train_dataset)}")
    if valid_dataset is not None:
        print(f"Valid samples: {len(valid_dataset)}")
    else:
        print("Warning: valid split not found. Training-only mode enabled.")

    train_loader = build_dataloader(
        dataset=train_dataset,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
        shuffle=True,
    )

    valid_loader = None
    if valid_dataset is not None:
        valid_loader = build_dataloader(
            dataset=valid_dataset,
            batch_size=config.training.batch_size,
            num_workers=config.training.num_workers,
            shuffle=False,
        )

    label_space = build_label_space(config)
    processor = build_processor(config)
    model = ArchiveMultitaskModel(config, label_space).to(device)
    criterion = ArchiveMultitaskLoss(config)

    optimizer = AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )

    total_steps = max(1, len(train_loader) * config.training.max_epochs)
    warmup_steps = int(total_steps * config.training.warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    output_dir = Path(config.artifacts_dir) / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)

    best_valid_loss = math.inf

    for epoch in range(config.training.max_epochs):
        print(f"\nEpoch {epoch + 1}/{config.training.max_epochs}")

        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            processor=processor,
            label_space=label_space,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            max_text_length=config.training.max_text_length,
            training=True,
        )
        print("Train:", train_metrics)

        last_ckpt_path = output_dir / "last_multitask.pt"
        save_checkpoint(
            path=last_ckpt_path,
            model=model,
            epoch=epoch,
            metrics=train_metrics,
            split_name="train",
        )
        print(f"Saved last checkpoint to {last_ckpt_path}")

        if valid_loader is not None:
            valid_metrics = run_epoch(
                model=model,
                loader=valid_loader,
                processor=processor,
                label_space=label_space,
                criterion=criterion,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
                max_text_length=config.training.max_text_length,
                training=False,
            )
            print("Valid:", valid_metrics)

            if valid_metrics["loss"] < best_valid_loss:
                best_valid_loss = valid_metrics["loss"]
                best_ckpt_path = output_dir / "best_multitask.pt"
                save_checkpoint(
                    path=best_ckpt_path,
                    model=model,
                    epoch=epoch,
                    metrics=valid_metrics,
                    split_name="valid",
                )
                print(f"Saved best checkpoint to {best_ckpt_path}")


if __name__ == "__main__":
    main()
