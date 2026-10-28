"""Shared runtime consolidated from Srijan's single and batch integration scripts."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw
from ultralytics import YOLO


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
VISION_DIR = REPOSITORY_ROOT / "setup" / "vision"
for import_root in (REPOSITORY_ROOT, VISION_DIR):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from temporal_pool import TemporalPool

try:
    from .measurement_qa import MeasurementQA
except ImportError:  # Direct script execution keeps integration/ on sys.path.
    from measurement_qa import MeasurementQA


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NOZZLE_X = 305
YOLO_CHECKPOINT = REPOSITORY_ROOT / "integration" / "models" / "yolo_best.pt"
TEMPORAL_CHECKPOINT = (
    Path.home()
    / "temporal-cnn-assets"
    / "temporal-cnn-assets-v1"
    / "models"
    / "liquid-best.pt"
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def mask_containment(liquid_mask: np.ndarray, vial_box: Any) -> float:
    height, width = liquid_mask.shape
    x0, y0, x1, y1 = map(float, vial_box)
    x0 = max(0, int(np.floor(x0)))
    y0 = max(0, int(np.floor(y0)))
    x1 = min(width, int(np.ceil(x1)))
    y1 = min(height, int(np.ceil(y1)))
    total_pixels = int(np.count_nonzero(liquid_mask))
    if total_pixels == 0 or x1 <= x0 or y1 <= y0:
        return 0.0
    inside_pixels = int(np.count_nonzero(liquid_mask[y0:y1, x0:x1]))
    return inside_pixels / total_pixels


def read_image(path: Path) -> np.ndarray:
    image = np.array(Image.open(path).convert("RGB"), copy=True)
    if image.shape != (480, 640, 3):
        raise ValueError(f"{path}: expected 640x480 RGB image")
    return image


def load_models(yolo_checkpoint: Path, temporal_checkpoint: Path):
    print("Loading YOLO...")
    yolo = YOLO(str(yolo_checkpoint))
    print("Loading Temporal CNN...")
    temporal = TemporalPool(checkpoint=temporal_checkpoint, device=DEVICE)
    return yolo, temporal


def process_pair(
    current_path: Path,
    reference_path: Path,
    yolo: Any,
    temporal: Any,
    output_dir: Path,
    number: int,
    qa: MeasurementQA,
    *,
    result_filename: str | None = None,
    mask_filename: str | None = None,
) -> dict[str, Any]:
    """Run the shared vial, liquid, association, rendering and QA pipeline."""
    current_path = Path(current_path)
    reference_path = Path(reference_path)
    output_dir = Path(output_dir)
    current = read_image(current_path)
    reference = read_image(reference_path)

    yolo_result = yolo.predict(
        source=Image.fromarray(current),
        conf=0.25,
        classes=[0],
        verbose=False,
    )[0]
    vial_boxes = yolo_result.boxes.xyxy.cpu().numpy()
    vial_scores = yolo_result.boxes.conf.cpu().numpy()

    prediction = temporal.predict(current=current, reference=reference)
    presence = float(prediction["presence"])
    presence_passed = bool(prediction["presence_gate_passed"])
    proposal_box = prediction["proposal_box"]
    proposal_mask = np.asarray(prediction["proposal_mask"], dtype=bool)

    matched_vial: int | None = None
    containment = 0.0
    if presence_passed and proposal_box is not None:
        candidates = [
            index
            for index, (x0, _y0, x1, _y1) in enumerate(vial_boxes)
            if x0 <= NOZZLE_X <= x1
        ]
        if candidates:
            matched_vial = min(
                candidates,
                key=lambda index: abs(
                    (vial_boxes[index][0] + vial_boxes[index][2]) / 2 - NOZZLE_X
                ),
            )
            vial_box = vial_boxes[matched_vial]
            expanded_vial_box = [
                vial_box[0] - 5,
                vial_box[1] - 5,
                vial_box[2] + 5,
                vial_box[3] + 5,
            ]
            containment = mask_containment(proposal_mask, expanded_vial_box)

    association_passed = bool(
        presence_passed and matched_vial is not None and containment >= 0.90
    )
    accepted_mask = np.zeros((480, 640), dtype=np.uint8)
    if association_passed:
        accepted_mask = proposal_mask.astype(np.uint8)
        status = "mask_generated"
    elif not presence_passed:
        status = "no_model_proposal"
    else:
        status = "unmatched_liquid_proposal"

    measurement = qa.measure(
        rgb=current,
        vial_boxes=vial_boxes,
        vial_scores=vial_scores,
        proposal_box=proposal_box,
        proposal_mask=proposal_mask,
        presence=presence,
        association_passed=association_passed,
    )

    output = current.copy()
    if association_passed:
        mask = accepted_mask.astype(bool)
        output[mask] = (
            output[mask] * 0.5 + np.array([0, 180, 255]) * 0.5
        ).astype(np.uint8)
    output_image = Image.fromarray(output)
    draw = ImageDraw.Draw(output_image)
    for vial_box in vial_boxes:
        draw.rectangle(tuple(vial_box), outline="green", width=2)
    if association_passed:
        draw.rectangle(tuple(proposal_box), outline="red", width=3)
        assert matched_vial is not None
        draw.rectangle(tuple(vial_boxes[matched_vial]), outline="yellow", width=3)

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{number:03d}_{current_path.stem}"
    result_path = output_dir / (result_filename or f"{stem}_result.jpg")
    mask_path = output_dir / (mask_filename or f"{stem}_mask.png")
    output_image.save(result_path)
    Image.fromarray(accepted_mask * 255).save(mask_path)

    measurement_record = _jsonable(measurement)
    return {
        "current": str(current_path),
        "reference": str(reference_path),
        "device": DEVICE,
        "vial_detections": len(vial_boxes),
        "vial_scores": _jsonable(vial_scores),
        "presence": presence,
        "presence_gate_passed": presence_passed,
        "proposal_box": _jsonable(proposal_box),
        "matched_vial_index": int(matched_vial) if association_passed else None,
        "containment": float(containment),
        "association_passed": association_passed,
        "mask_pixels": int(accepted_mask.sum()),
        "reference_status": "unverified",
        "measurement_QA_executed": True,
        "measurement_valid": bool(measurement["measurement_valid"]),
        "quality_flags": _jsonable(measurement["quality_flags"]),
        "measurement": measurement_record,
        "status": status,
        "result_file": result_path.name,
        "mask_file": mask_path.name,
    }
