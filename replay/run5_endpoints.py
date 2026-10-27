"""Retrospective Run5 completed-vial endpoints through the frozen operator pipeline.

Run5 is diagnostic-only: nothing here fits, tunes or selects models, thresholds or QA
settings. Each organizer-labelled vial gets a fresh QA tracker; the temporal reference is
the frame at the labelled fill start (unreviewed, not certified empty). Endpoints are the
median Lab of QA-valid frames in a fixed window after the labelled fill end.
"""
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from integration.runtime import ModelRuntime

STEP = 15            # frames between samples (0.5 s at 30 fps)
SETTLE = 90          # frames after fill end treated as the completed endpoint (3 s)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True,
                        help="CSV with vial,start_frame,end_frame[,recipe_id], e.g. video5_vial_recipes.csv")
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)

    vials = list(csv.DictReader(args.labels.open()))
    plans = {}
    for row in vials:
        start, end = int(row["start_frame"]), int(row["end_frame"])
        frames = list(range(start, end, STEP)) + list(range(end, end + SETTLE + 1, STEP))
        plans[int(row["vial"])] = {"row": row, "reference": start, "frames": frames, "endpoint_from": end}
    wanted = {f for p in plans.values() for f in [p["reference"], *p["frames"]]}

    capture = cv2.VideoCapture(str(args.video))
    images, index = {}, 0
    last = max(wanted)
    while index <= last:
        ok, bgr = capture.read()
        if not ok:
            break
        if index in wanted:
            images[index] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        index += 1
    capture.release()

    runtime = ModelRuntime(args.assets, args.device)
    results = []
    for vial, plan in plans.items():
        qa = runtime.new_measurement()
        reference = images[plan["reference"]]
        rows = []
        for frame in plan["frames"]:
            if frame not in images:
                continue
            result, measured, _ = runtime.infer(images[frame], reference, qa)
            rows.append({"frame": frame, "endpoint_window": frame >= plan["endpoint_from"],
                         "presence": float(result["presence"]),
                         "proposal_pixels": int(np.asarray(result["proposal_mask"]).sum()),
                         "qa_valid": bool(measured["measurement_valid"]),
                         "flags": list(measured["quality_flags"]),
                         "lab": measured["lab"], "rgb": measured["rgb"]})
        endpoint = [r["lab"] for r in rows if r["endpoint_window"] and r["qa_valid"]]
        row = plan["row"]
        results.append({
            "vial": vial, "recipe_id": row.get("recipe_id"), "start_frame": int(row["start_frame"]),
            "end_frame": int(row["end_frame"]), "reference_frame": plan["reference"],
            "frames_processed": len(rows), "qa_valid_frames": sum(r["qa_valid"] for r in rows),
            "endpoint_frames": sum(r["endpoint_window"] for r in rows), "endpoint_valid": len(endpoint),
            "endpoint_lab": np.median(np.asarray(endpoint), axis=0).tolist() if endpoint else None,
            "frames": rows,
        })
        print(f"vial {vial:2d} recipe {str(row.get('recipe_id')):>4}: {len(endpoint)}/{sum(r['endpoint_window'] for r in rows)} endpoint "
              f"frames valid; endpoint Lab {results[-1]['endpoint_lab']}", flush=True)

    (args.out / "run5-endpoints.json").write_text(json.dumps({
        "schema": "run5-endpoints/v1", "video": str(args.video), "labels": str(args.labels),
        "policy": {"step_frames": STEP, "settle_frames": SETTLE, "reference": "frame at labelled fill start (unreviewed)",
                   "models_and_qa": "released checkpoints and unchanged MeasurementQA; nothing tuned on Run5"},
        "vials": results}, indent=1))
    usable = sum(r["endpoint_lab"] is not None for r in results)
    print(f"Endpoints with QA-valid color: {usable}/{len(results)} vials")


if __name__ == "__main__":
    main()
