
from pathlib import Path
import argparse
import csv
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

def read_image(path):
    image = np.array (
        Image.open(path).convert("RGB"),
        copy=True
    )

    if image.shape != (480, 640, 3):
        raise ValueError(
            f"{path}: expected 640x480 RGB image"
        )

    return image


def load_models():
    print("Loading YOLO...")
    yolo = YOLO(str(YOLO_CHECKPOINT))

    print("Loading Temporal CNN...")
    temporal = TemporalPool(
        checkpoint=TEMPORAL_CHECKPOINT,
        device=DEVICE
    )

    return yolo, temporal


def process_pair(
    current_path,
    reference_path,
    yolo,
    temporal,
    output_dir,
    number
):
    current = read_image(current_path)
    reference = read_image(reference_path)

    # YOLO vial detection
    yolo_result = yolo.predict(
        source=Image.fromarray(current),
        conf=0.25,
        classes=[0],
        verbose=False
    )[0]

    vial_boxes = (
        yolo_result.boxes.xyxy.cpu().numpy()
    )

    # Temporal CNN liquid prediction
    prediction = temporal.predict(
        current=current,
        reference=reference
    )

    presence = float(prediction["presence"])

    presence_passed = bool(
        prediction["presence_gate_passed"]
    )

    proposal_box = prediction["proposal_box"]

    proposal_mask = np.asarray(
        prediction["proposal_mask"],
        dtype=bool
    )

    matched_vial = None
    containment = 0.0
    
    if presence_passed and proposal_box is not None:
    	candidates = []

    for index, vial_box in enumerate(vial_boxes):
        x0, y0, x1, y1 = vial_box

        if x0 <= NOZZLE_X <= x1:
            candidates.append(index)

    if candidates:
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

    # All detected vial boxes
    for vial_box in vial_boxes:
        draw.rectangle(
            tuple(vial_box),
            outline="green",
            width=2
        )

    # Accepted liquid box and matched vial
    if association_passed:
        draw.rectangle(
            tuple(proposal_box),
            outline="red",
            width=3
        )

        draw.rectangle(
            tuple(vial_boxes[matched_vial]),
            outline="yellow",
            width=3
        )

    stem = f"{number:03d}_{current_path.stem}"

    result_path = output_dir / f"{stem}_result.jpg"
    mask_path = output_dir / f"{stem}_mask.png"

    output_image.save(result_path)

    Image.fromarray(
        accepted_mask * 255
    ).save(mask_path)

    return {
        "current": current_path.name,
        "reference": str(reference_path),
        "vial_detections": len(vial_boxes),
        "presence": presence,
        "presence_gate_passed": presence_passed,
        "proposal_box": json.dumps(proposal_box),
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
        "status": status,
        "result_file": result_path.name,
        "mask_file": mask_path.name
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "input_dir",
        type=Path
    )

    parser.add_argument(
        "--pairs",
        type=Path,
        required=True,
        help="CSV with current,reference columns"
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=20
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "integration/batch_results"
    )

    args = parser.parse_args()

    if args.limit < 1:
        parser.error("--limit must be at least 1")

    if not args.input_dir.is_dir():
        parser.error("Input directory does not exist")

    # Read explicitly selected frame/reference pairs
    with open(args.pairs, newline="") as f:
        reader = csv.DictReader(f)

        if not {"current", "reference"}.issubset(
            reader.fieldnames or []
        ):
            parser.error(
                "Pairs CSV needs current,reference columns"
            )

        pairs = list(reader)

    if not pairs:
        parser.error("Pairs CSV is empty")

    pairs = pairs[:args.limit]

    # Check all inputs before loading models
    resolved_pairs = []

    for row in pairs:
        current_path = args.input_dir / row["current"]

        reference_path = Path(row["reference"])

        if not reference_path.is_absolute():
            reference_path = args.input_dir / reference_path

        if not current_path.is_file():
            raise FileNotFoundError(current_path)

        if not reference_path.is_file():
            raise FileNotFoundError(reference_path)

        if current_path.resolve() == reference_path.resolve():
            raise ValueError(
                "Current and reference must be different images"
            )

        resolved_pairs.append(
            (current_path, reference_path)
        )

    # Use a new output directory for each batch run
    args.output_dir.mkdir(
        parents=True,
        exist_ok=False
    )

    print("Device:", DEVICE)
    print("Pairs:", len(resolved_pairs))

    yolo, temporal = load_models()

    rows = []

    for number, (current_path, reference_path) in enumerate(
        resolved_pairs, 1
    ):
        print(
            f"[{number}/{len(resolved_pairs)}] "
            f"{current_path.name}"
        )

        row = process_pair(
            current_path=current_path,
            reference_path=reference_path,
            yolo=yolo,
            temporal=temporal,
            output_dir=args.output_dir,
            number=number
        )

        rows.append(row)

        print(
            f"  Status: {row['status']}, "
            f"presence: {row['presence']:.6f}, "
            f"mask pixels: {row['mask_pixels']}"
        )

    summary_path = args.output_dir / "summary.csv"

    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=rows[0].keys()
        )

        writer.writeheader()
        writer.writerows(rows)

    print("\nFinished!")
    print("Results:", args.output_dir)
    print("Summary:", summary_path)


if __name__ == "__main__":
    main()
