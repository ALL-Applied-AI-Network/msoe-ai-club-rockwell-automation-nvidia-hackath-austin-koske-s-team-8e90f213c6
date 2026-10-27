
import torch
import torchvision
from torchvision.transforms.functional import to_tensor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image, ImageDraw
from pathlib import Path
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the same Faster R-CNN architecture used for training
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
    weights=None,
    weights_backbone=None
)

num_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(num_features, 2)

model.load_state_dict(
    torch.load("liquid_pool_model_v5_10epochs.pth", map_location=device, weights_only=True)
)

model.to(device)
model.eval()

# Validation images
val_dir = Path("liquid-team-data-v1/starter/images/val")

# Output folder
output_dir = Path("predictions_v5_random30")
output_dir.mkdir(exist_ok=True)

# Select 7 vial-fill images
images = sorted(val_dir.glob("*vial_fill*.jpg"))
random.seed(42)
images = random.sample(images, min(30, len(images)))

print("Device:", device)
print("Images found:", len(images))

with torch.no_grad():
    for image_path in images:
        image = Image.open(image_path).convert("RGB")
        tensor = to_tensor(image).to(device)

        prediction = model([tensor])[0]

        scores = prediction["scores"].cpu()
        boxes = prediction["boxes"].cpu()

        draw = ImageDraw.Draw(image)

        if len(scores) > 0 and scores[0].item() >= 0.3:
            box = boxes[0].tolist()
            score = scores[0].item()

            draw.rectangle(box, outline="red", width=3)
            draw.text(
                (box[0], max(0, box[1] - 15)),
                f"{score:.2f}",
                fill="red"
            )
            print(image_path.name, "Confidence:", round(score, 3))
        else:
            print(image_path.name, "No detection above threshold")

        image.save(output_dir / image_path.name)

print("Saved predictions to:", output_dir)
