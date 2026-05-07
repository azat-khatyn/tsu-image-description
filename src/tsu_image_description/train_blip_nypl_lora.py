import json
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from transformers import BlipProcessor, BlipForConditionalGeneration
from peft import LoraConfig, get_peft_model
import os

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

TRAIN_PATH = "data/nypl/train_v1.json"
VAL_PATH = "data/nypl/val_v1.json"

OUTPUT_DIR = "models"
EXPERIMENT_NAME = "blip_caplift_v5_lora"

BATCH_SIZE = 4
EPOCHS = 7
LR = 3e-6  # ниже, чем обычный fine-tuning


# =========================
# Dataset
# =========================
class NYPLDataset(Dataset):
    def __init__(self, path, processor):
        with open(path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        image = Image.open(item["image"]).convert("RGB")
        caption = item["caption"]

        encoding = self.processor(
            images=image,
            text=caption,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=50
        )

        encoding = {k: v.squeeze(0) for k, v in encoding.items()}
        encoding["labels"] = encoding["input_ids"]

        return encoding


# =========================
# Training
# =========================
def train():
    processor = BlipProcessor.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )

    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )

    # =========================
    # LoRA CONFIG
    # =========================
    lora_config = LoraConfig(
        r=16,  # было 8
        lora_alpha=32,
        target_modules=["query", "key", "value"],
        lora_dropout=0.05,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    model.to(DEVICE)

    train_ds = NYPLDataset(TRAIN_PATH, processor)
    val_ds = NYPLDataset(VAL_PATH, processor)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        # ===== TRAIN =====
        model.train()
        train_loss = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch} Train"):
            batch = {k: v.to(DEVICE) for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # ===== VALIDATION =====
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch} Val"):
                batch = {k: v.to(DEVICE) for k, v in batch.items()}

                outputs = model(**batch)
                val_loss += outputs.loss.item()

        val_loss /= len(val_loader)

        print(f"\nEpoch {epoch}")
        print(f"Train Loss: {train_loss}")
        print(f"Val Loss: {val_loss}")

        # ===== SAVE CHECKPOINT =====
        save_path = os.path.join(
            OUTPUT_DIR,
            f"{EXPERIMENT_NAME}_epoch_{epoch}"
        )

        os.makedirs(save_path, exist_ok=True)

        model.save_pretrained(save_path)
        processor.save_pretrained(save_path)

        print(f"Saved checkpoint to {save_path}")


if __name__ == "__main__":
    train()