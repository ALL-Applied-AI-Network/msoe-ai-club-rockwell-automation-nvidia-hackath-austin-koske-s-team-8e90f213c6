
from pathlib import Path
import argparse
import json
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw
from ultralytics import YOLO


ROOT = Path.home() / "fluid-mixing-cnn"
VISION_DIR = ROOT / "setup/vision"

sys.path.insert(0, str(VISION_DIR))
from temporal_pool import TemporalPool


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NOZZLE_X = 305

YOLO_CHECKPOINT = ROOT / "integration/models/yolo_best.pt"

TEMPORAL_CHECKPOINT = (
    Path.home()
    / "temporal-cnn-assets/temporal-cnn-assets-v1"
    / "models/liquid-best.pt"
)

OUTPUT_DIR = ROOT / "integration"


def load_models():
    print("Loading YOLO...")
    yolo = YOLO(str(YOLO_CHECKPOINT))

    print("Loading Temporal CNN...")
    temporal = TemporalPool(
        checkpoint=TEMPORAL_CHECKPOINT,
        device=DEVICE
    )

    return yolo, temporal


def mask_containment(liquid_mask, vial_box):
    height, width = liquid_mask.shape

    x0, y0, x1, y1 = map(float, vial_box)

    x0 = max(0, int(np.floor(x0)))
    y0 = max(0, int(np.floor(y0)))
    x1 = min(width, int(np.ceil(x1)))
    y1 = min(height, int(np.ceil(y1)))

    total_pixels = int(np.count_nonzero(liquid_mask))

    if total_pixels == 0 or x1 <= x0 or y1 <= y0:
        return 0.0

    inside_pixels = int(
        np.count_nonzero(liquid_mask[y0:y1, x0:x1])
    )

    return inside_pixels / total_pixels

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--current",
        type=Path,
        required=True
    )

    parser.add_argument(
        "--reference",
        type=Path,
        required=True
    )

    args = parser.parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for filename in (
        "result.jpg",
        "temporal_mask.png",
        "observation.json"
    ):
        (OUTPUT_DIR / filename).unlink(missing_ok=True)
    current = np.array(
        Image.open(args.current).convert("RGB"),
        copy=True
    )

    reference = np.array(
        Image.open(args.reference).convert("RGB"),
        copy=True
    )

    if current.shape != (480, 640, 3):
        raise ValueError(
            "Current image must be 640x480 RGB"
        )

    if reference.shape != (480, 640, 3):
        raise ValueError(
            "Reference image must be 640x480 RGB"
        )

    print("Device:", DEVICE)

    yolo, temporal = load_models()

    # Team 1: YOLO vial detection
    # PIL input preserves correct RGB handling
    yolo_result = yolo.predict(
        source=Image.fromarray(current),
        conf=0.25,
        classes=[0],
        verbose=False
    )[0]

    vial_boxes = (
        yolo_result.boxes.xyxy.cpu().numpy()
    )

    vial_scores = (
        yolo_result.boxes.conf.cpu().numpy()
    )

    print("YOLO vial detections:", len(vial_boxes))

    # Team 2: Temporal CNN liquid prediction
    result = temporal.predict(
        current=current,
        reference=reference
    )

    presence = float(result["presence"])
    presence_passed = bool(
        result["presence_gate_passed"]
    )

    proposal_box = result["proposal_box"]
    proposal_mask = np.asarray(
        result["proposal_mask"],
        dtype=bool
    )

    print("Liquid presence:", presence)
    print("Presence gate passed:", presence_passed)

    # Associate liquid proposal with a detected vial
    matched_vial = None
    containment = 0.0
    
    if presence_passed and proposal_box is not None:
    	candidates = []

    for index, vial_box in enumerate(vial_boxes):
        x0, y0, x1, y1 = vial_box

        # Only consider vials beneath the nozzle
        if x0 <= NOZZLE_X <= x1:
            candidates.append(index)

    if candidates:
        # Choose the vial whose center is closest to the nozzle
        matched_vial = min(
            candidates,
            key=lambda i: abs(
                (vial_boxes[i][0] + vial_boxes[i][2]) / 2
                - NOZZLE_X
            )
        )

        vial_box = vial_boxes[matched_vial]
        padding = 5

        expanded_vial_box = [
            vial_box[0] - padding,
            vial_box[1] - padding,
            vial_box[2] + padding,
            vial_box[3] + padding
        ]

        containment = mask_containment(
            proposal_mask,
            expanded_vial_box
        )

    association_passed = (
        presence_passed
        and matched_vial is not None
        and containment >= 0.90
    )

    # Save only an associated liquid mask
    accepted_mask = np.zeros(
        (480, 640),
        dtype=np.uint8
    )

    if association_passed:
        accepted_mask = proposal_mask.astype(
            np.uint8
        )

        status = "mask_generated"

    elif not presence_passed:
        status = "no_model_proposal"

    else:
        status = "unmatched_liquid_proposal"

    print("Association passed:", association_passed)
    print("Containment:", containment)
    print("Status:", status)

    # Create visualization
    output = current.copy()

    if association_passed:
        mask = accepted_mask.astype(bool)

        output[mask] = (
            output[mask] * 0.5
            + np.array([0, 180, 255]) * 0.5
        ).astype(np.uint8)

    output_image = Image.fromarray(output)
    draw = ImageDraw.Draw(output_image)

    # All YOLO vial boxes in green
    for vial_box in vial_boxes:
        draw.rectangle(
            tuple(vial_box),
            outline="green",
            width=2
        )

    # Accepted liquid proposal in red
    if presence_passed and proposal_box is not None:
        draw.rectangle(
            tuple(proposal_box),
            outline="red",
            width=3
        )

        # Matched vial in yellow
        if association_passed:
            draw.rectangle(
                tuple(vial_boxes[matched_vial]),
                outline="yellow",
                width=3
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = OUTPUT_DIR / "result.jpg"
    mask_path = OUTPUT_DIR / "temporal_mask.png"
    json_path = OUTPUT_DIR / "observation.json"

    # Avoid leaving a mask from an earlier run
    mask_path.unlink(missing_ok=True)

    output_image.save(output_path)

    Image.fromarray(
        accepted_mask * 255
    ).save(mask_path)

    observation = {
        "current": str(args.current),
        "reference": str(args.reference),
        "device": DEVICE,
        "vial_detections": len(vial_boxes),
        "vial_scores": vial_scores.tolist(),
        "presence": presence,
        "presence_gate_passed": presence_passed,
        "proposal_box": proposal_box,
        "matched_vial_index": (
            int(matched_vial)
            if association_passed
            else None
        ),
        "containment": float(containment),
        "association_passed": association_passed,
        "mask_pixels": int(accepted_mask.sum()),
        "reference_status": "unverified",
        "measurement_QA_executed": False,
        "status": status
    }

    def convert_numpy(obj):
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Cannot serialize {type(obj).__name__}")

    with open(json_path, "w") as f:
        json.dump(
            observation,
            f,
            indent=2,
            default = convert_numpy
        )

    print("Saved:", output_path)
    print("Saved:", mask_path)
    print("Saved:", json_path)


if __name__ == "__main__":
    main()
