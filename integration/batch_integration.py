
from pathlib import Path
import argparse
import contextlib
import csv

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
        weights=None,
        weights_backbone=None
    )
    features = rcnn.roi_heads.box_predictor.cls_score.in_features
    rcnn.roi_heads.box_predictor = FastRCNNPredictor(features, 2)

    weights = torch.load(
        ROOT / "liquid_pool_model_v5_10epochs.pth",
        map_location=DEVICE,
        weights_only=True
    )
    rcnn.load_state_dict(weights)
    rcnn.to(DEVICE).eval()

    sam = build_sam2(
        "configs/sam2.1/sam2.1_hiera_t.yaml",
        str(
            Path.home()
            / "fluid-mixing-pipeline/checkpoints/sam2.1_hiera_tiny.pt"
        ),
        device=DEVICE
    )

    return yolo, rcnn, SAM2ImagePredictor(sam)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--threshold", type=float, default=0.3)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = ROOT / "integration/batch_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        p for p in input_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )[:args.limit]

    if not image_paths:
        raise SystemExit("No images found in input directory")

    print("Device:", DEVICE)
    print("Images:", len(image_paths))
    print("Loading models...")

    yolo, rcnn, sam_predictor = load_models()
    rows = []

    for number, image_path in enumerate(image_paths, 1):
        print(f"[{number}/{len(image_paths)}] {image_path.name}")

        image = Image.open(image_path).convert("RGB")
        rgb = np.array(image, copy=True)
        width, height = image.size

        output = image.copy()
        draw = ImageDraw.Draw(output)

        yolo_result = yolo.predict(
            source=rgb,
            conf=0.25,
            verbose=False
        )[0]

        vial_boxes = yolo_result.boxes.xyxy.cpu().numpy()
        vial_scores = yolo_result.boxes.conf.cpu().numpy()

        with torch.inference_mode():
            prediction = rcnn(
                [to_tensor(image).to(DEVICE)]
            )[0]

        liquid_boxes = prediction["boxes"].cpu().numpy()
        liquid_scores = prediction["scores"].cpu().numpy()

        for box in vial_boxes:
            draw.rectangle(
                tuple(box),
                outline="green",
                width=2
            )

        # Use the highest-confidence liquid detection
        candidates = np.flatnonzero(
            liquid_scores >= args.threshold
        )

        status = "no_liquid_proposal"
        liquid_confidence = None
        sam_score = None
        mask_pixels = 0

        if len(candidates) > 0:
            index = int(candidates[0])
            box = liquid_boxes[index].tolist()
            liquid_confidence = float(liquid_scores[index])

            x0, y0, x1, y1 = box
            valid = (
                0 <= x0 < x1 <= width
                and 0 <= y0 < y1 <= height
            )

            if not valid:
                status = "invalid_liquid_box"
            else:
                cx = (x0 + x1) / 2
                cy = (y0 + y1) / 2

                matched = any(
                    vx0 <= cx <= vx1 and vy0 <= cy <= vy1
                    for vx0, vy0, vx1, vy1 in vial_boxes
                )

                if not matched:
                    status = "unmatched_liquid_box"
                else:
                    with torch.inference_mode():
                        context = (
                            torch.autocast(
                                "cuda", dtype=torch.bfloat16
                            )
                            if DEVICE == "cuda"
                            else contextlib.nullcontext()
                        )
                        with context:
                            sam_predictor.set_image(rgb)
                            masks, scores, _ = sam_predictor.predict(
                                box=np.asarray(
                                    box, dtype=np.float32
                                ),
                                multimask_output=False
                            )

                    mask = np.asarray(masks[0], dtype=bool)
                    mask_pixels = int(mask.sum())
                    sam_score = float(scores[0])

                    mask_image = Image.fromarray(
                        mask.astype(np.uint8) * 255
                    )
                    mask_image.save(
                        output_dir / f"{image_path.stem}_mask.png"
                    )

                    overlay = np.array(output, copy=True)
                    overlay[mask] = (
                        overlay[mask] * 0.5
                        + np.array([0, 180, 255]) * 0.5
                    ).astype(np.uint8)

                    output = Image.fromarray(overlay)
                    draw = ImageDraw.Draw(output)

                    status = (
                        "mask_generated"
                        if mask_pixels > 0
                        else "empty_mask"
                    )

                draw.rectangle(
                    tuple(box),
                    outline="red",
                    width=3
                )

        output.save(
            output_dir / f"{image_path.stem}_result.jpg"
        )

        rows.append({
            "image": image_path.name,
            "vial_detections": len(vial_boxes),
            "liquid_detections": len(candidates),
            "liquid_confidence": liquid_confidence,
            "sam_score": sam_score,
            "mask_pixels": mask_pixels,
            "status": status
        })

        print(
            f"  Status: {status}, "
            f"mask pixels: {mask_pixels}"
        )

    with open(
        output_dir / "summary.csv",
        "w",
        newline=""
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=rows[0].keys()
        )
        writer.writeheader()
        writer.writerows(rows)

    print("\nFinished!")
    print("Results:", output_dir)
    print("Summary:", output_dir / "summary.csv")


if __name__ == "__main__":
    main()
