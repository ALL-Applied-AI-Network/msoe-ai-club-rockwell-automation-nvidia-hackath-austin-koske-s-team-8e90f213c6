#!/usr/bin/env python3
"""Run a provenance-preserving SAM2.1 pilot on selected Run 4 frames."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


DEFAULT_FRAMES = (330, 450, 750, 900)
MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_t.yaml"
PALETTE = ((0, 255, 210), (255, 190, 0), (255, 70, 140), (120, 200, 255))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(path: Path) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or not isinstance(manifest.get("frames"), list):
        raise ValueError(f"Unsupported SAM frame manifest: {path}")
    rows: dict[int, dict[str, Any]] = {}
    for row in manifest["frames"]:
        index = row.get("source_frame_index")
        if not isinstance(index, int) or index in rows:
            raise ValueError(f"Invalid or duplicate source frame index: {index!r}")
        rows[index] = row
    return manifest, rows


def validate_frame(
    row: dict[str, Any], data_root: Path, expected_index: int
) -> tuple[Path, np.ndarray, list[dict[str, Any]], list[dict[str, Any]]]:
    if row.get("run") != 4 or row.get("split_role") != "development":
        raise ValueError(f"Frame {expected_index} is not Run 4 development data")
    if row.get("training_eligible") is not False:
        raise ValueError(f"Frame {expected_index} has unexpected training eligibility")
    if row.get("width") != 640 or row.get("height") != 480:
        raise ValueError(f"Frame {expected_index} has unexpected dimensions")

    image_path = (data_root / row["image_path"]).resolve()
    if not image_path.is_relative_to(data_root.resolve()):
        raise ValueError(f"Image path escapes data root: {row['image_path']}")
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    actual_hash = sha256_file(image_path)
    if actual_hash != row.get("image_sha256"):
        raise ValueError(f"Image SHA-256 mismatch for source frame {expected_index}")

    with Image.open(image_path) as source:
        if source.size != (640, 480):
            raise ValueError(f"Image dimensions disagree with manifest: {image_path}")
        rgb = np.array(source.convert("RGB"), copy=True)

    proposals = row.get("detector_proposals")
    if not isinstance(proposals, list):
        raise ValueError(f"Missing proposal list for source frame {expected_index}")
    valid_liquid: list[tuple[int, dict[str, Any]]] = []
    unavailable: list[dict[str, Any]] = []
    for proposal_index, proposal in enumerate(proposals):
        reason = None
        if not isinstance(proposal, dict):
            reason = "proposal is not an object"
            class_id = None
            raw_box = proposal
        else:
            class_id = proposal.get("class_id")
            raw_box = proposal.get("box")
            if not isinstance(raw_box, list) or len(raw_box) != 4:
                reason = "box must contain four xyxy coordinates"
            else:
                try:
                    coordinates = [float(value) for value in raw_box]
                except (TypeError, ValueError):
                    reason = "box contains a non-numeric coordinate"
                else:
                    if not all(math.isfinite(value) for value in coordinates):
                        reason = "box contains a non-finite coordinate"
                    else:
                        x0, y0, x1, y1 = coordinates
                        if not (0 <= x0 < x1 <= 640 and 0 <= y0 < y1 <= 480):
                            reason = "box is outside the 640x480 frame or has zero area"
                        else:
                            proposal["box"] = coordinates
        if reason is not None:
            unavailable.append(
                {
                    "proposal_id": proposal_index,
                    "class_id": class_id,
                    "prompt_box_xyxy_exclusive": raw_box,
                    "status": "unavailable",
                    "reason": reason,
                }
            )
        elif class_id == 1:
            valid_liquid.append((proposal_index, proposal))
    return image_path, rgb, valid_liquid, unavailable


def make_overlay(rgb: np.ndarray, mask: np.ndarray, box: list[float], label: str, color: tuple[int, int, int]) -> Image.Image:
    blended = rgb.astype(np.float32)
    selected = mask.astype(bool)
    tint = np.asarray(color, dtype=np.float32)
    blended[selected] = blended[selected] * 0.48 + tint * 0.52
    image = Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8), mode="RGB")
    draw = ImageDraw.Draw(image)
    x0, y0, x1, y1 = box
    draw.rectangle((x0, y0, x1 - 1, y1 - 1), outline=(255, 255, 0), width=2)
    draw.rectangle((4, 4, 160, 24), fill=(0, 0, 0))
    draw.text((8, 8), label, fill=(255, 255, 255))
    return image


def write_result(path: Path, result: dict[str, Any]) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True, help="liquid-team-data-v1 directory")
    parser.add_argument("--checkpoint", type=Path, required=True, help="SAM2.1 Hiera Tiny checkpoint")
    parser.add_argument("--output-dir", type=Path, required=True, help="Fresh results directory outside Git")
    parser.add_argument("--frames", type=int, nargs="+", default=DEFAULT_FRAMES)
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    args = parser.parse_args()

    data_root = args.data_root.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    manifest_path = data_root / "sam_starter" / "frame-manifest.json"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    if len(args.frames) != len(set(args.frames)):
        raise ValueError("Frame indices must be unique")

    manifest, rows = load_manifest(manifest_path)
    manifest_hash = sha256_file(manifest_path)
    prepared = []
    for index in args.frames:
        if index not in rows:
            raise ValueError(f"Source frame {index} is not in the SAM development manifest")
        row = rows[index]
        image_path, rgb, liquid, unavailable = validate_frame(row, data_root, index)
        invalid_liquid = any(item["class_id"] == 1 for item in unavailable)
        prepared.append((index, row, image_path, rgb, liquid, unavailable, invalid_liquid))

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "masks").mkdir()
    (output_dir / "overlays").mkdir()

    result: dict[str, Any] = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_run": manifest.get("source_run"),
        "split_role": "development",
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_hash,
        "model": "SAM2.1 Hiera Tiny",
        "model_config": MODEL_CONFIG,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "device": device,
        "torch_version": torch.__version__,
        "requested_source_frame_indices": args.frames,
        "frames": [],
        "ground_truth_masks_available": bool(manifest.get("ground_truth_masks_included")),
        "warning": "Machine proposals and SAM masks are unreviewed development outputs, not ground truth.",
    }
    result_path = output_dir / "result.json"

    model_started = time.perf_counter()
    model = build_sam2(MODEL_CONFIG, str(checkpoint), device=device)
    predictor = SAM2ImagePredictor(model)
    result["model_load_ms"] = round((time.perf_counter() - model_started) * 1000, 3)
    autocast_context = (
        lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device == "cuda"
        else contextlib.nullcontext()
    )

    for index, row, image_path, rgb, liquid_proposals, unavailable, invalid_liquid in prepared:
        frame_started = time.perf_counter()
        frame_result: dict[str, Any] = {
            "sample_id": row["sample_id"],
            "camera": row["camera"],
            "run": row["run"],
            "split_role": row["split_role"],
            "source_frame_index": index,
            "pts": row["pts"],
            "time_base": row["time_base"],
            "pts_seconds": row["pts_seconds"],
            "image_path": str(image_path),
            "image_sha256": row["image_sha256"],
            "source_rgb_sha256": row.get("source_rgb_sha256"),
            "source_video_sha256": row.get("source_video_sha256"),
            "image_dimensions": [640, 480],
            "detector_models": row.get("detector_models", {}),
            "status": (
                "masks_generated"
                if liquid_proposals
                else "unavailable_invalid_liquid_proposal"
                if invalid_liquid
                else "no_liquid_proposal"
            ),
            "candidates": [],
            "unavailable_proposals": unavailable,
        }

        if liquid_proposals:
            with torch.inference_mode(), autocast_context():
                predictor.set_image(rgb)
                for proposal_index, proposal in liquid_proposals:
                    prompt_started = time.perf_counter()
                    box = np.asarray(proposal["box"], dtype=np.float32)
                    masks, scores, _ = predictor.predict(box=box, multimask_output=False)
                    mask = np.asarray(masks)
                    while mask.ndim > 2:
                        mask = mask[0]
                    if mask.shape != (480, 640):
                        raise ValueError(
                            f"SAM mask shape {mask.shape} does not match source frame {index}"
                        )
                    binary_mask = mask.astype(bool)
                    stem = f"frame-{index:06d}_proposal-{proposal_index:02d}"
                    mask_path = output_dir / "masks" / f"{stem}.png"
                    overlay_path = output_dir / "overlays" / f"{stem}.png"
                    Image.fromarray(binary_mask.astype(np.uint8) * 255, mode="L").save(mask_path)
                    color = PALETTE[len(frame_result["candidates"]) % len(PALETTE)]
                    overlay = make_overlay(
                        rgb,
                        binary_mask,
                        [float(value) for value in proposal["box"]],
                        f"frame {index} proposal {proposal_index}",
                        color,
                    )
                    overlay.save(overlay_path)
                    frame_result["candidates"].append(
                        {
                            "proposal_id": proposal_index,
                            "class_id": proposal["class_id"],
                            "class_name": "liquid",
                            "prompt_box_xyxy_exclusive": [float(value) for value in proposal["box"]],
                            "detector_score": float(proposal["confidence"]),
                            "sam_score": float(np.asarray(scores).reshape(-1)[0]),
                            "mask_file": str(mask_path),
                            "overlay_file": str(overlay_path),
                            "mask_dimensions": [640, 480],
                            "foreground_pixel_count": int(binary_mask.sum()),
                            "elapsed_ms": round((time.perf_counter() - prompt_started) * 1000, 3),
                        }
                    )

        frame_result["elapsed_ms"] = round((time.perf_counter() - frame_started) * 1000, 3)
        result["frames"].append(frame_result)
        write_result(result_path, result)
        print(
            f"frame={index} pts={row['pts_seconds']:.3f}s status={frame_result['status']} "
            f"candidates={len(frame_result['candidates'])} elapsed_ms={frame_result['elapsed_ms']}",
            flush=True,
        )

    result["total_elapsed_ms"] = round(
        result["model_load_ms"] + sum(frame["elapsed_ms"] for frame in result["frames"]), 3
    )
    write_result(result_path, result)
    print(f"result={result_path}")


if __name__ == "__main__":
    main()
