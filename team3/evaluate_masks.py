#!/usr/bin/env python3
"""Compare SAM pilot masks with reviewed frame masks (CPU-only)."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image, ImageChops


def load_binary_mask(path: Path, expected_size: tuple[int, int]) -> Image.Image:
    with Image.open(path) as image:
        if image.size != expected_size:
            raise ValueError(f"Mask {path} has size {image.size}, expected {expected_size}")
        return image.convert("L").point(lambda value: 255 if value >= 128 else 0).convert("1")


def pixel_count(mask: Image.Image) -> int:
    return mask.histogram()[255]


def locate_prediction(pilot_dir: Path, candidate: dict[str, Any]) -> Path:
    filename = Path(str(candidate.get("mask_file", ""))).name
    if not filename:
        raise ValueError("Candidate is missing mask_file")
    direct = pilot_dir / "masks" / filename
    if direct.is_file():
        return direct
    matches = list(pilot_dir.rglob(filename))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one local copy of {filename} under {pilot_dir}, found {len(matches)}"
        )
    return matches[0]


def score_masks(reference: Image.Image, prediction: Image.Image) -> dict[str, Any]:
    intersection = pixel_count(ImageChops.logical_and(reference, prediction))
    reference_pixels = pixel_count(reference)
    prediction_pixels = pixel_count(prediction)
    union = reference_pixels + prediction_pixels - intersection
    return {
        "reference_pixels": reference_pixels,
        "prediction_pixels": prediction_pixels,
        "intersection_pixels": intersection,
        "union_pixels": union,
        "iou": intersection / union if union else 1.0,
        "dice": (2 * intersection / (reference_pixels + prediction_pixels))
        if reference_pixels + prediction_pixels
        else 1.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-dir", type=Path, required=True, help="Local SAM pilot result directory")
    parser.add_argument("--annotation-dir", type=Path, required=True, help="Directory with annotations.json and masks")
    parser.add_argument("--output", type=Path, help="Report JSON path (defaults to pilot-dir/mask-evaluation.json)")
    args = parser.parse_args()

    pilot_dir = args.pilot_dir.expanduser().resolve()
    annotation_dir = args.annotation_dir.expanduser().resolve()
    pilot_path = pilot_dir / "result.json"
    annotation_path = annotation_dir / "annotations.json"
    pilot = json.loads(pilot_path.read_text(encoding="utf-8"))
    annotations = json.loads(annotation_path.read_text(encoding="utf-8"))
    if annotations.get("split_role") != "development" or annotations.get("training_eligible") is not False:
        raise ValueError("This evaluator requires development annotations marked training_eligible=false")

    annotation_rows = {row["source_frame_index"]: row for row in annotations.get("frames", [])}
    frame_results = {row["source_frame_index"]: row for row in pilot.get("frames", [])}
    if not annotation_rows:
        raise ValueError(f"No annotation frames found in {annotation_path}")

    evaluations: list[dict[str, Any]] = []
    score_groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for frame_index, annotation in sorted(annotation_rows.items()):
        frame = frame_results.get(frame_index)
        if frame is None:
            raise ValueError(f"Pilot result is missing annotated source frame {frame_index}")
        width, height = annotation.get("dimensions", [640, 480])
        expected_size = (int(width), int(height))
        reference_path = annotation_dir / annotation["mask"]
        if not reference_path.is_file():
            reference_path = annotation_dir / "masks" / annotation["mask"]
        reference = load_binary_mask(reference_path, expected_size)
        presence = annotation.get("presence")
        candidates = frame.get("candidates", [])
        item: dict[str, Any] = {
            "source_frame_index": frame_index,
            "pts_seconds": frame.get("pts_seconds"),
            "presence": presence,
            "reference_mask": str(reference_path),
            "reference_pixels": pixel_count(reference),
            "pilot_status": frame.get("status"),
            "presence_detection_correct": (
                frame.get("status") == "no_liquid_proposal" and not candidates
                if presence == "absent"
                else bool(candidates)
            ),
            "candidates": [],
        }
        for candidate in candidates:
            prediction_path = locate_prediction(pilot_dir, candidate)
            prediction = load_binary_mask(prediction_path, expected_size)
            metrics = score_masks(reference, prediction)
            proposal_id = int(candidate["proposal_id"])
            score_groups[proposal_id].append(metrics)
            item["candidates"].append(
                {
                    "proposal_id": proposal_id,
                    "prompt_box_xyxy_exclusive": candidate.get("prompt_box_xyxy_exclusive"),
                    "detector_score": candidate.get("detector_score"),
                    "sam_score": candidate.get("sam_score"),
                    "prediction_mask": str(prediction_path),
                    **metrics,
                }
            )
        if not candidates:
            item["segmentation_metrics"] = None
        evaluations.append(item)

    report = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "pilot_result": str(pilot_path),
        "annotation_manifest": str(annotation_path),
        "model": pilot.get("model"),
        "annotation_split_role": annotations.get("split_role"),
        "annotation_training_eligible": annotations.get("training_eligible"),
        "interpretation": (
            "Exploratory scores against rough human development masks. They are useful for comparing prompts, "
            "not a model-accuracy claim or a basis for weight fine-tuning."
        ),
        "by_proposal_id": {
            str(proposal_id): {
                "candidate_count": len(group),
                "mean_iou": sum(row["iou"] for row in group) / len(group),
                "mean_dice": sum(row["dice"] for row in group) / len(group),
            }
            for proposal_id, group in sorted(score_groups.items())
        },
        "frames": evaluations,
    }
    output = args.output.expanduser().resolve() if args.output else pilot_dir / "mask-evaluation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output}")
    for proposal_id, summary in report["by_proposal_id"].items():
        print(
            f"proposal {proposal_id}: n={summary['candidate_count']} "
            f"mean IoU={summary['mean_iou']:.3f} mean Dice={summary['mean_dice']:.3f}"
        )


if __name__ == "__main__":
    main()
