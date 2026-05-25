import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import json
from tqdm import tqdm

DEVICE = (
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

class PostcardDataset(Dataset):
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

        inputs = self.processor(
            images=image,
            text=caption,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        inputs["labels"] = inputs["input_ids"]

        return inputs


def train():
    model_name = "Salesforce/blip-image-captioning-base"

    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForConditionalGeneration.from_pretrained(model_name).to(DEVICE)

    train_ds = PostcardDataset("data/nypl/splits/train_v2.json", processor)
    val_ds = PostcardDataset("data/nypl/splits/val_v2.json", processor)

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=4)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)

    for epoch in range(2):
        model.train()
        total_loss = 0

        for batch in tqdm(train_loader):
            batch = {k: v.to(DEVICE) for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()

        print(f"Epoch {epoch} train loss: {total_loss / len(train_loader)}")

        model.eval()
        val_loss = 0

        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(DEVICE) for k, v in batch.items()}
                outputs = model(**batch)
                val_loss += outputs.loss.item()

        print(f"Epoch {epoch} val loss: {val_loss / len(val_loader)}")

        save_path = f"models/blip_caplift_v2_epoch_{epoch}"
        model.save_pretrained(save_path)
        processor.save_pretrained(save_path)

        print(f"Saved checkpoint to {save_path}")

    model.save_pretrained("models/blip_caplift_v2")
    processor.save_pretrained("models/blip_caplift_v2")


if __name__ == "__main__":
    train()
