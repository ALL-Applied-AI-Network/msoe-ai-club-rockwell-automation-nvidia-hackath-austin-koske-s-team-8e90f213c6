
import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import to_tensor
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    FasterRCNN_ResNet50_FPN_Weights,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


class LiquidPoolDataset(Dataset):
    def __init__(self, image_dir, annotation_file):
        self.image_dir = Path(image_dir)

        with open(annotation_file) as f:
            data = json.load(f)

        # Only use images with at least one annotation.
        annotations = {}
        for ann in data["annotations"]:
            annotations.setdefault(ann["image_id"], []).append(ann)

        self.images = [
            info for info in data["images"]
            if info["id"] in annotations
        ]
        self.annotations = annotations

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        info = self.images[index]
        image = Image.open(
            self.image_dir / info["file_name"]
        ).convert("RGB")

        boxes = []
        for ann in self.annotations[info["id"]]:
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.ones(len(boxes), dtype=torch.int64),
        }

        return to_tensor(image), target


def collate_fn(batch):
    return tuple(zip(*batch))


def main():
    root = Path("liquid-team-data-v1/starter")
    image_dir = root
    annotation_file = root / "coco/likely_good.json"

    dataset = LiquidPoolDataset(image_dir, annotation_file)

    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        collate_fn=collate_fn,
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)
    print("Training images:", len(dataset))

    model = fasterrcnn_resnet50_fpn(
        weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    )

    # Background + liquid_body
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features, 2
    )

    model.to(device)
    model.train()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.001,
        momentum=0.9,
        weight_decay=0.0005,
    )

    epochs = 10

    for epoch in range(epochs):
        total_loss = 0.0

        for images, targets in loader:
            images = [img.to(device) for img in images]
            targets = [
                {k: v.to(device) for k, v in t.items()}
                for t in targets
            ]

            losses = model(images, targets)
            loss = sum(losses.values())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(
            f"Epoch {epoch + 1}/{epochs} "
            f"- Loss: {total_loss / len(loader):.4f}",
            flush=True,
        )

    torch.save(model.state_dict(), "liquid_pool_model_v5_10epochs.pth")
    print("Saved liquid_pool_model.pth")


if __name__ == "__main__":
    main()
