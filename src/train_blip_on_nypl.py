# file: src/train_blip_on_nypl.py

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import evaluate
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    BlipForConditionalGeneration,
    BlipProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)


class NYPLCaptionDataset(Dataset):
    def __init__(self, manifest_path: str, processor: BlipProcessor, max_length: int = 96):
        self.manifest_path = Path(manifest_path)
        self.processor = processor
        self.max_length = max_length
        self.rows = self._load_rows()

    def _load_rows(self) -> List[Dict[str, Any]]:
        rows = []
        with self.manifest_path.open("r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                if Path(row["image"]).exists() and row.get("text"):
                    rows.append(row)
        if not rows:
            raise RuntimeError(f"No valid rows found in {self.manifest_path}")
        return rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.rows[idx]
        image = Image.open(row["image"]).convert("RGB")
        text = row["text"]

        model_inputs = self.processor(
            images=image,
            text=text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        model_inputs = {k: v.squeeze(0) for k, v in model_inputs.items()}
        labels = model_inputs["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        model_inputs["labels"] = labels
        return model_inputs


@dataclass
class DataCollatorForBLIP:
    processor: BlipProcessor

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch = {}
        for key in features[0].keys():
            batch[key] = torch.stack([f[key] for f in features])
        return batch


def split_manifest(
    manifest_path: Path,
    train_ratio: float = 0.9,
) -> tuple[List[dict], List[dict]]:
    rows = []
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))

    rng = np.random.default_rng(42)
    idx = np.arange(len(rows))
    rng.shuffle(idx)

    cut = int(len(rows) * train_ratio)
    train_rows = [rows[i] for i in idx[:cut]]
    val_rows = [rows[i] for i in idx[cut:]]

    return train_rows, val_rows


def save_jsonl(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=str,
        default="data/external/nypl/nypl_manifest.jsonl",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Salesforce/blip-image-captioning-base",
    )
    parser.add_argument("--output-dir", type=str, default="artifacts/blip_nypl")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-5)
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    train_rows, val_rows = split_manifest(manifest_path)

    train_manifest = manifest_path.parent / "train.jsonl"
    val_manifest = manifest_path.parent / "val.jsonl"
    save_jsonl(train_manifest, train_rows)
    save_jsonl(val_manifest, val_rows)

    processor = BlipProcessor.from_pretrained(args.model)
    model = BlipForConditionalGeneration.from_pretrained(args.model)

    train_ds = NYPLCaptionDataset(str(train_manifest), processor)
    val_ds = NYPLCaptionDataset(str(val_manifest), processor)

    rouge = evaluate.load("rouge")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred

        if isinstance(predictions, tuple):
            predictions = predictions[0]

        pred_ids = np.argmax(predictions, axis=-1)

        label_ids = labels.copy()
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        pred_texts = processor.batch_decode(pred_ids, skip_special_tokens=True)
        label_texts = processor.batch_decode(label_ids, skip_special_tokens=True)

        scores = rouge.compute(
            predictions=pred_texts,
            references=label_texts,
            use_stemmer=True,
        )
        return {k: round(v, 4) for k, v in scores.items()}

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        logging_steps=20,
        eval_strategy="epoch",
        save_strategy="epoch",
        predict_with_generate=False,
        fp16=torch.cuda.is_available(),
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorForBLIP(processor),
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
