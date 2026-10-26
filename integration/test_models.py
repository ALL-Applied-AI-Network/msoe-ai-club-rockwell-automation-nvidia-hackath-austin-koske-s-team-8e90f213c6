import torch
from ultralytics import YOLO
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from sam2.build_sam import build_sam2

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# Team 1: YOLO
yolo = YOLO("integration/models/yolo_best.pt")
print("YOLO loaded")

# Team 2: Faster R-CNN
rcnn = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None)

in_features = rcnn.roi_heads.box_predictor.cls_score.in_features
rcnn.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)

checkpoint = torch.load(
    "liquid_pool_model_v5_10epochs.pth",
    map_location=device,
    weights_only=True
)

rcnn.load_state_dict(checkpoint)
rcnn.to(device).eval()
print("Faster R-CNN loaded")

# Team 3: SAM2
sam = build_sam2(
    "configs/sam2.1/sam2.1_hiera_t.yaml",
    "/home/hackathon8/fluid-mixing-pipeline/checkpoints/sam2.1_hiera_tiny.pt",
    device=device
)

print("SAM2 loaded")
print("All three models loaded successfully!")
