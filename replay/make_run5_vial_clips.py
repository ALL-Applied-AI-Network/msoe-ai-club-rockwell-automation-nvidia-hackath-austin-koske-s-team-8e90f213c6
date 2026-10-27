"""Cut per-vial Run5 demo clips and write the recipe library used by the monitor.

Clips are exact labelled fill windows [start_frame, end_frame] re-encoded for browser
playback. Reference colors are copied from the frozen Run5 proxy endpoints; nothing is
recomputed or tuned here.
"""
import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

VIALS = [2, 4, 5, 7, 8]


def frame_at(path, index):
    capture = cv2.VideoCapture(str(path))
    capture.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"Could not read frame {index} of {path}")
    return frame.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True, help="video5_vial_recipes.csv")
    parser.add_argument("--comparison", type=Path, required=True, help="run5-recipe-comparison.json")
    parser.add_argument("--run5", type=Path, required=True, help="client/public/data/run5/run5.json")
    parser.add_argument("--clips", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    args = parser.parse_args()
    args.clips.mkdir(parents=True, exist_ok=True)

    labels = {int(row["vial"]): row for row in csv.DictReader(args.labels.open())}
    comparison = {row["vial"]: row for row in json.loads(args.comparison.read_text())["vials"]}
    clips = []
    for vial in VIALS:
        row = labels[vial]
        start, end = int(row["start_frame"]), int(row["end_frame"])
        out = args.clips / f"run5-vial{vial:02d}-recipe{row['recipe_id']}.mp4"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(args.video),
                        "-vf", f"trim=start_frame={start}:end_frame={end + 1},setpts=PTS-STARTPTS",
                        "-an", "-r", "30", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
                        "-movflags", "+faststart", str(out)], check=True)
        probe = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
                                "-show_entries", "stream=nb_read_frames,width,height,duration", "-of", "json", str(out)],
                               capture_output=True, text=True, check=True)
        stream = json.loads(probe.stdout)["streams"][0]
        diff = float(np.abs(frame_at(out, 0) - frame_at(args.video, start)).mean())
        result = comparison[vial]
        clips.append({
            "file": out.name, "vial": vial, "recipe": row["recipe_id"], "source_frames": [start, end],
            "frames": int(stream["nb_read_frames"]), "expected_frames": end - start + 1,
            "width": stream["width"], "height": stream["height"], "duration_s": float(stream["duration"]),
            "first_frame_mean_abs_diff": round(diff, 3),
            "sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
            "expected_from_offline_analysis": {
                "note": "expected from offline analysis (browser decode may differ slightly)",
                **{key: result.get(key) for key in ("dE_to_recipe", "dE_to_previous", "toward_previous_cos", "diverges")}},
        })
        print(f"{out.name}: {stream['nb_read_frames']}/{end - start + 1} frames, first-frame diff {diff:.2f}")
    (args.clips / "clips.json").write_text(json.dumps({
        "schema": "run5-vial-clips/v1", "source_video": str(args.video), "labels": str(args.labels),
        "clips": clips}, indent=1))

    run5 = json.loads(args.run5.read_text())
    references = {str(v["recipe"]): v for v in run5["vials"] if v.get("reference_role") == "reference"}
    recipes = []
    for recipe_id in range(5, 15):
        row = next(r for r in labels.values() if int(r["recipe_id"]) == recipe_id)
        volume = lambda key: float(row[key]) if row[key] else 0.0
        ref = references.get(str(recipe_id))
        recipes.append({
            "id": str(recipe_id), "name": f"Recipe {recipe_id}",
            "t100_ml": volume("t100_ml"), "t200_ml": volume("t200_ml"), "t400_ml": volume("t400_ml"),
            "order": row["dispense_order"], "note": row["color_purpose"] or None,
            "reference": {"lab": ref["lab"], "from": f"Run 5 vial {ref['vial']} endpoint (proxy)"} if ref else None,
        })
    args.library.write_text(json.dumps({
        "schema": "recipe-library/v1",
        "source": ("Organizer dataset README §4 (Run 5 recipes); reference colors are Run 5 proxy endpoints "
                   "from the frozen pipeline (3rd vial of each recipe, 2nd as fallback), not certified"),
        "band_de2000": 5, "direction_cos": 0.8,
        "endpoint_rule": "median Lab of QA-valid samples in the last 6 s of the vial clip",
        "recipes": recipes}, indent=1))
    print("references:", [r["id"] for r in recipes if r["reference"]])


if __name__ == "__main__":
    main()
