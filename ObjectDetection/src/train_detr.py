import os
import torch
from torch.utils.data import DataLoader
from PIL import Image
import json
from transformers import DetrForObjectDetection, DetrImageProcessor
from tqdm import tqdm

print("TRAINING SCRIPT STARTED SUCCESSFULLY")   


class PassportDataset(torch.utils.data.Dataset):
    def __init__(self, json_path, img_folder):
        print("Loading dataset JSON...")   
        self.img_folder = img_folder

        with open(json_path, "r") as f:
            coco = json.load(f)

        self.images = coco["images"]
        self.annotations = coco["annotations"]

        
        self.ann_by_img = {img["id"]: [] for img in self.images}
        for ann in self.annotations:
            self.ann_by_img[ann["image_id"]].append(ann)

        self.processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_info = self.images[idx]
        img_path = os.path.join(self.img_folder, img_info["file_name"])

        image = Image.open(img_path).convert("RGB")

        anns = self.ann_by_img[img_info["id"]]

        
        target = {
            "image_id": img_info["id"],
            "annotations": [
                {
                    "bbox": ann["bbox"],
                    "area": ann["area"],
                    "iscrowd": 0,
                    "category_id": 0  
                }
                for ann in anns
            ]
        }

        encoding = self.processor(
            images=image,
            annotations=target,
            return_tensors="pt"
        )

        encoding["pixel_values"] = encoding["pixel_values"].squeeze(0)
        encoding["labels"] = encoding["labels"][0]

        return encoding


def collate_fn(batch):
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    labels = [item["labels"] for item in batch]

    return {"pixel_values": pixel_values, "labels": labels}


def train():
    print("Initializing dataset...")

    dataset = PassportDataset(
        json_path="data/annotations/annotations.json",
        img_folder="data/passports"
    )

    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        collate_fn=collate_fn
    )

    print("Loading DETR model...")
    model = DetrForObjectDetection.from_pretrained(
        "facebook/detr-resnet-50",
        num_labels=1, 
        ignore_mismatched_sizes=True
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    print("Starting training on CPU...")
    model.train()

    for epoch in range(5):
        print(f"\nEpoch {epoch+1}/5")
        for batch in tqdm(dataloader):
            outputs = model(
                pixel_values=batch["pixel_values"],
                labels=batch["labels"]
            )

            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        print(f"Epoch {epoch+1} Loss: {loss.item():.4f}")

    os.makedirs("models", exist_ok=True)
    model.save_pretrained("models/detr_passport")
    print("Model saved to models/detr_passport")


if __name__ == "__main__":
    train()
