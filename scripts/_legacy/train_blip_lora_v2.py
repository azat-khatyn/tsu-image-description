"""train_blip_lora_v2.py — generic BLIP LoRA training, M1 MPS friendly.

Improvements over scripts/train_blip_lora.py:
  - CLI-configurable (any dataset that follows {image_path, caption} format)
  - BLIP-large by default (was base)
  - MPS-optimized: num_workers=0, periodic mps.empty_cache, fp32
  - Standard LoRA lr (1e-4) instead of suspicious 3e-6 from v5
  - Train/val loss logging + per-step prints
  - Saves checkpoint each epoch

Usage:
  PYTHONPATH=src python scripts/train_blip_lora_v2.py \\
      --train data/artcap/train.json \\
      --val data/artcap/val.json \\
      --output-dir models/blip_artcap_lora \\
      --epochs 3 --batch-size 8 --lr 1e-4 --lora-r 16

Dataset format (JSON list):
  [{"image_path": "...", "caption": "..."}, ...]
"""

import argparse
import gc
import json
import os
import time
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from transformers import BlipProcessor, BlipForConditionalGeneration
from peft import LoraConfig, get_peft_model


def get_device(force: str = None) -> str:
    if force:
        return force
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True, help="JSON list of {image_path, caption}")
    p.add_argument("--val", required=True)
    p.add_argument("--base-model", default="Salesforce/blip-image-captioning-large")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--experiment-name", default="blip_lora")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--max-length", type=int, default=50)
    p.add_argument("--device", default=None, help="cpu/mps/cuda; auto by default")
    p.add_argument("--num-workers", type=int, default=0,
                   help="DataLoader workers; KEEP 0 on Mac MPS")
    p.add_argument("--empty-cache-every", type=int, default=50,
                   help="Call torch.mps.empty_cache() every N steps")
    p.add_argument("--smoke", action="store_true",
                   help="Smoke test: 50 steps + exit")
    p.add_argument("--log-every", type=int, default=20)
    return p.parse_args()


class CaptionDataset(Dataset):
    def __init__(self, path: str, processor, max_length: int = 50):
        with open(path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.processor = processor
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # Backward-compat: support old "image" key as well as "image_path"
        image_path = item.get("image_path") or item.get("image")
        caption = item["caption"]

        image = Image.open(image_path).convert("RGB")

        encoding = self.processor(
            images=image,
            text=caption,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )

        encoding = {k: v.squeeze(0) for k, v in encoding.items()}
        encoding["labels"] = encoding["input_ids"].clone()
        return encoding


def empty_cache(device: str):
    if device == "mps":
        try:
            torch.mps.empty_cache()
        except Exception:
            pass
    elif device == "cuda":
        torch.cuda.empty_cache()
    gc.collect()


def main():
    args = parse_args()
    device = get_device(args.device)
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Base model: {args.base_model}")
    print(f"[INFO] Train: {args.train}")
    print(f"[INFO] Val:   {args.val}")
    print(f"[INFO] Output: {args.output_dir}/{args.experiment_name}_epoch_X")
    print(f"[INFO] LoRA: r={args.lora_r} alpha={args.lora_alpha} dropout={args.lora_dropout}")
    print(f"[INFO] Training: lr={args.lr} batch_size={args.batch_size} epochs={args.epochs}")

    processor = BlipProcessor.from_pretrained(args.base_model)
    model = BlipForConditionalGeneration.from_pretrained(args.base_model)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["query", "key", "value"],
        lora_dropout=args.lora_dropout,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.to(device)

    train_ds = CaptionDataset(args.train, processor, args.max_length)
    val_ds = CaptionDataset(args.val, processor, args.max_length)
    print(f"[INFO] Train items: {len(train_ds)}   Val items: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    os.makedirs(args.output_dir, exist_ok=True)

    smoke_steps = 50

    for epoch in range(args.epochs):
        model.train()
        train_loss_sum = 0.0
        n_steps = 0
        t0 = time.time()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch} train")
        for step, batch in enumerate(pbar):
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"\n[WARN] NaN/Inf loss at step {step}, epoch {epoch}. Skipping.")
                optimizer.zero_grad()
                continue

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_loss_sum += loss.item()
            n_steps += 1

            if step % args.log_every == 0:
                avg = train_loss_sum / max(n_steps, 1)
                elapsed = time.time() - t0
                rate = (step + 1) / max(elapsed, 1e-3)
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "avg": f"{avg:.4f}",
                    "step/s": f"{rate:.2f}",
                })

            if (step + 1) % args.empty_cache_every == 0:
                empty_cache(device)

            if args.smoke and step + 1 >= smoke_steps:
                print(f"\n[INFO] Smoke test reached {smoke_steps} steps, exiting OK.")
                return

        train_loss = train_loss_sum / max(n_steps, 1)

        # ===== VALIDATION =====
        model.eval()
        val_loss_sum = 0.0
        n_val = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch} val"):
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                if torch.isnan(outputs.loss) or torch.isinf(outputs.loss):
                    continue
                val_loss_sum += outputs.loss.item()
                n_val += 1
        val_loss = val_loss_sum / max(n_val, 1)

        elapsed = time.time() - t0
        print(f"\nEpoch {epoch}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  elapsed={elapsed:.0f}s")

        save_path = Path(args.output_dir) / f"{args.experiment_name}_epoch_{epoch}"
        save_path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(save_path))
        processor.save_pretrained(str(save_path))
        print(f"[INFO] Saved checkpoint: {save_path}")
        empty_cache(device)


if __name__ == "__main__":
    main()
