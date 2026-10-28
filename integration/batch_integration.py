from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

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
except ImportError:  # Supports `python integration/batch_integration.py`.
    from model_pipeline import (
        DEVICE,
        MeasurementQA,
        REPOSITORY_ROOT,
        TEMPORAL_CHECKPOINT,
        YOLO_CHECKPOINT,
        load_models,
        process_pair,
    )


def _csv_value(value: Any) -> Any:
    return json.dumps(value) if isinstance(value, (dict, list)) else value


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the shared model pipeline over ordered current/reference pairs."
    )
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--pairs", type=Path, required=True, help="CSV with current,reference columns")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPOSITORY_ROOT / "integration" / "batch_results",
    )
    parser.add_argument("--yolo-checkpoint", type=Path, default=YOLO_CHECKPOINT)
    parser.add_argument("--temporal-checkpoint", type=Path, default=TEMPORAL_CHECKPOINT)
    args = parser.parse_args()

    if args.limit < 1:
        parser.error("--limit must be at least 1")
    if not args.input_dir.is_dir():
        parser.error("Input directory does not exist")

    with args.pairs.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not {"current", "reference"}.issubset(reader.fieldnames or []):
            parser.error("Pairs CSV needs current,reference columns")
        pairs = list(reader)[: args.limit]
    if not pairs:
        parser.error("Pairs CSV is empty")

    resolved_pairs: list[tuple[Path, Path]] = []
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
            raise ValueError("Current and reference must be different images")
        resolved_pairs.append((current_path, reference_path))

    args.output_dir.mkdir(parents=True, exist_ok=False)
    print("Device:", DEVICE)
    print("Pairs:", len(resolved_pairs))
    yolo, temporal = load_models(args.yolo_checkpoint, args.temporal_checkpoint)
    qa = MeasurementQA()
    rows: list[dict[str, Any]] = []

    for number, (current_path, reference_path) in enumerate(resolved_pairs, 1):
        print(f"[{number}/{len(resolved_pairs)}] {current_path.name}")
        row = process_pair(
            current_path=current_path,
            reference_path=reference_path,
            yolo=yolo,
            temporal=temporal,
            output_dir=args.output_dir,
            number=number,
            qa=qa,
        )
        rows.append(row)
        print(
            f"  Status: {row['status']}, presence: {row['presence']:.6f}, "
            f"mask pixels: {row['mask_pixels']}, QA valid: {row['measurement_valid']}"
        )
        print("  QA flags:", row["quality_flags"])

    summary_path = args.output_dir / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(
            {key: _csv_value(value) for key, value in row.items()} for row in rows
        )
    print("Finished!")
    print("Results:", args.output_dir)
    print("Summary:", summary_path)


if __name__ == "__main__":
    main()
