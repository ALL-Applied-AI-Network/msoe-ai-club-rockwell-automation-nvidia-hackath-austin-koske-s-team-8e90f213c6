from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .model_pipeline import (
        DEVICE,
        MeasurementQA,
        REPOSITORY_ROOT,
        TEMPORAL_CHECKPOINT,
        YOLO_CHECKPOINT,
        load_models,
        process_pair,
    )
except ImportError:  # Supports `python integration/integration.py`.
    from model_pipeline import (
        DEVICE,
        MeasurementQA,
        REPOSITORY_ROOT,
        TEMPORAL_CHECKPOINT,
        YOLO_CHECKPOINT,
        load_models,
        process_pair,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one current/reference pair through the shared model pipeline."
    )
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--yolo-checkpoint", type=Path, default=YOLO_CHECKPOINT)
    parser.add_argument("--temporal-checkpoint", type=Path, default=TEMPORAL_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=REPOSITORY_ROOT / "integration")
    args = parser.parse_args()

    if not args.current.is_file():
        raise FileNotFoundError(args.current)
    if not args.reference.is_file():
        raise FileNotFoundError(args.reference)
    if args.current.resolve() == args.reference.resolve():
        raise ValueError("Current and reference must be different images")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print("Device:", DEVICE)
    yolo, temporal = load_models(args.yolo_checkpoint, args.temporal_checkpoint)
    observation = process_pair(
        current_path=args.current,
        reference_path=args.reference,
        yolo=yolo,
        temporal=temporal,
        output_dir=args.output_dir,
        number=1,
        qa=MeasurementQA(),
        result_filename="result.jpg",
        mask_filename="temporal_mask.png",
    )
    observation_path = args.output_dir / "observation.json"
    observation_path.write_text(json.dumps(observation, indent=2), encoding="utf-8")

    print("YOLO vial detections:", observation["vial_detections"])
    print("Liquid presence:", observation["presence"])
    print("Association passed:", observation["association_passed"])
    print("Measurement valid:", observation["measurement_valid"])
    print("QA flags:", observation["quality_flags"])
    print("Saved:", args.output_dir / observation["result_file"])
    print("Saved:", args.output_dir / observation["mask_file"])
    print("Saved:", observation_path)


if __name__ == "__main__":
    main()
