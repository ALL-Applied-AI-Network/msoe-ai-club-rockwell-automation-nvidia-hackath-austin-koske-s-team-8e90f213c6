
from pathlib import Path
import argparse
import contextlib

import numpy as np
import torch
from PIL import Image, ImageDraw
from ultralytics import YOLO
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms.functional import to_tensor
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


ROOT = Path.home() / "fluid-mixing-cnn"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_models():
    yolo = YOLO(str(ROOT / "integration/models/yolo_best.pt"))

    rcnn = fasterrcnn_resnet50_fpn(
        weights=None, weights_backbone=None
    )
    features = rcnn.roi_heads.box_predictor.cls_score.in_features
    rcnn.roi_heads.box_predictor = FastRCNNPredictor(features, 2)

    weights = torch.load(
        ROOT / "liquid_pool_model_v5_10epochs.pth",
        map_location=DEVICE,
        weights_only=True,
    )
    rcnn.load_state_dict(weights)
    rcnn.to(DEVICE).eval()

    sam = build_sam2(
        "configs/sam2.1/sam2.1_hiera_t.yaml",
        str(
            Path.home()
            / "fluid-mixing-pipeline/checkpoints/sam2.1_hiera_tiny.pt"
        ),
        device=DEVICE,
    )
    sam_predictor = SAM2ImagePredictor(sam)

    return yolo, rcnn, sam_predictor


def valid_box(box, width, height):
    x0, y0, x1, y1 = box
    return (
        0 <= x0 < x1 <= width
        and 0 <= y0 < y1 <= height
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("--threshold", type=float, default=0.3)
    args = parser.parse_args()

    image_path = Path(args.image)
    image = Image.open(image_path).convert("RGB")
    rgb = np.asarray(image)
    width, height = image.size

    print("Device:", DEVICE)
    print("Loading models...")
    yolo, rcnn, sam_predictor = load_models()

    # Team 1: detect vial
    yolo_results = yolo.predict(
        source=rgb,
        conf=0.25,
        verbose=False,
    )[0]

    vial_boxes = yolo_results.boxes.xyxy.cpu().numpy()
    vial_scores = yolo_results.boxes.conf.cpu().numpy()
    print("YOLO detections:", len(vial_boxes))

    # Team 2: detect accumulated liquid
    with torch.inference_mode():
        prediction = rcnn([to_tensor(image).to(DEVICE)])[0]

    liquid_boxes = prediction["boxes"].cpu().numpy()
    liquid_scores = prediction["scores"].cpu().numpy()

    selected = np.flatnonzero(liquid_scores >= args.threshold)
    print("Liquid detections:", len(selected))

    output = image.copy()
    draw = ImageDraw.Draw(output)

    # Draw YOLO boxes in green
    for box, score in zip(vial_boxes, vial_scores):
        draw.rectangle(tuple(box), outline="green", width=2)

    if len(selected) == 0:
        print("No liquid detection above threshold")
    else:
        # Highest-confidence liquid prediction
        index = int(selected[0])
        liquid_box = liquid_boxes[index].tolist()
        print("Liquid confidence:", float(liquid_scores[index]))

        if valid_box(liquid_box, width, height):
            # Check liquid center lies inside a detected vial
            cx = (liquid_box[0] + liquid_box[2]) / 2
            cy = (liquid_box[1] + liquid_box[3]) / 2

            matched = any(
                x0 <= cx <= x1 and y0 <= cy <= y1
                for x0, y0, x1, y1 in vial_boxes
            )

            if matched:
                print("Liquid matched to a vial")

                with torch.inference_mode():
                    context = (
                        torch.autocast("cuda", dtype=torch.bfloat16)
                        if DEVICE == "cuda"
                        else contextlib.nullcontext()
                    )
                    with context:
                        sam_predictor.set_image(rgb)
                        masks, scores, _ = sam_predictor.predict(
                            box=np.asarray(liquid_box, dtype=np.float32),
                            multimask_output=False,
                        )

                mask = np.asarray(masks[0], dtype=bool)
                print("SAM2 mask pixels: ",int(mask.sum()))
                mask_image = Image.fromarray(
                    (mask.astype(np.uint8) * 255)
                )
                mask_image.save(ROOT / "integration/sam_mask.png")
                overlay = np.asarray(output).copy()
                overlay[mask] = (
                    overlay[mask] * 0.5
                    + np.array([0, 180, 255]) * 0.5
                ).astype(np.uint8)

                output = Image.fromarray(overlay)
                draw = ImageDraw.Draw(output)
                print("SAM2 mask score:", float(scores[0]))
            else:
                print("Liquid box did not match a detected vial")

            draw.rectangle(tuple(liquid_box), outline="red", width=3)

    output_path = ROOT / "integration/result.jpg"
    output.save(output_path)
    print("Saved:", output_path)


if __name__ == "__main__":
    main()
