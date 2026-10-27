"""Export the Run5 retrospective recipe check as static data for the operator UI.

Reads the frozen comparison/endpoints produced by replay.run5_endpoints and the organizer
labels, writes run5.json plus one JPEG crop per vial. No measurement is recomputed beyond
reading those outputs; endpoint frame = middle QA-valid frame in the last 6 s of the fill.
"""
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from skimage.color import lab2rgb

WINDOW = 180  # last 6 s of the labelled fill (rule C, chosen on Runs 1-4)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    comparison = json.loads((args.results / "run5-recipe-comparison.json").read_text())
    endpoints = {v["vial"]: v for v in json.loads((args.results / "endpoints-v1/run5-endpoints.json").read_text())["vials"]}
    labels = {int(r["vial"]): r for r in csv.DictReader(args.labels.open())}
    rows = {r["vial"]: r for r in comparison["vials"]}

    frames = {}
    for vial, ep in endpoints.items():
        valid = [r["frame"] for r in ep["frames"] if ep["end_frame"] - WINDOW <= r["frame"] < ep["end_frame"] and r["qa_valid"]]
        frames[vial] = (valid[len(valid) // 2], True) if valid else (ep["end_frame"] - 30, False)
    wanted = {f for f, _ in frames.values()}
    capture, images, index = cv2.VideoCapture(str(args.video)), {}, 0
    while index <= max(wanted):
        ok, bgr = capture.read()
        if not ok:
            break
        if index in wanted:
            images[index] = bgr
        index += 1
    capture.release()

    def reason(vial):
        flags = [f for r in endpoints[vial]["frames"]
                 if endpoints[vial]["end_frame"] - WINDOW <= r["frame"] < endpoints[vial]["end_frame"] for f in r["flags"]]
        if not flags:
            return "no frames in endpoint window"
        top = max(set(flags), key=flags.count)
        return top.replace("_", " ")

    vials = []
    for vial in sorted(labels):
        row, label = rows[vial], labels[vial]
        frame, measured = frames[vial]
        crop = f"vial-{vial:02d}.jpg"
        cv2.imwrite(str(args.out / crop), images[frame][150:470, 230:410], [cv2.IMWRITE_JPEG_QUALITY, 80])
        lab = row["lab"]
        swatch = (np.clip(lab2rgb(np.asarray(lab, float).reshape(1, 1, 3)), 0, 1)[0, 0] * 255).round().astype(int).tolist() if lab else None
        reference = comparison["references"].get(str(row["recipe"]))
        role = ("reference" if (reference == "vial3" and row["position"] == 3) or (reference == "vial2_fallback" and row["position"] == 2)
                else "compared" if "dE_to_recipe" in row else "not_measurable" if not lab else "no_reference")
        vials.append({
            "vial": vial, "recipe": row["recipe"], "position": row["position"], "endpoint_frames": row["endpoint_frames"],
            "endpoint_frame": frame, "frame_has_valid_endpoint": measured, "lab": lab, "swatch_rgb": swatch,
            "dE_to_recipe": row.get("dE_to_recipe"), "dE_to_previous": row.get("dE_to_previous"),
            "toward_previous_cos": row.get("toward_previous_cos"), "diverges": row.get("diverges"),
            "reference_role": role, "reason": None if lab else reason(vial), "crop": crop,
            "label_vial_id": label.get("vial"),
        })

    recipes = {}
    for label in labels.values():
        rid = int(label["recipe_id"])
        recipes.setdefault(rid, {"recipe": rid, "t100_ml": label["t100_ml"] or None, "t200_ml": label["t200_ml"] or None,
                                 "t400_ml": label["t400_ml"] or None, "total_ml": label["total_ml"],
                                 "dispense_order": label["dispense_order"], "note": label["color_purpose"] or None,
                                 "reference": comparison["references"].get(str(rid))})

    compared = [v for v in vials if v["reference_role"] == "compared"]
    first = [v for v in compared if v["position"] == 1]
    second = [v for v in compared if v["position"] == 2]
    summary = {
        "vials_with_endpoint": sum(v["lab"] is not None for v in vials), "vials": len(vials),
        "recipes_with_reference": sum(r["reference"] is not None for r in recipes.values()), "recipes": len(recipes),
        "first_vials_comparable": len(first), "first_vials_total": sum(v["position"] == 1 for v in vials),
        "first_vials_diverging": sum(bool(v["diverges"]) for v in first),
        "second_vials_comparable": len(second), "second_vials_within_band": sum(not v["diverges"] for v in second),
    }
    (args.out / "run5.json").write_text(json.dumps({
        "schema": "run5-recipe-check/v1",
        "source": "Organizer video 5 vial_fill (640x480, 30 fps) and video5_vial_recipes.csv labels",
        "policy": {"endpoint_rule": comparison["rule"], "band_dE2000": comparison["band_dE2000"],
                   "direction_cos": comparison["direction_cos"], "reference_policy": comparison["reference_policy"],
                   "models": "Released YOLO11n + temporal U-Net + unchanged MeasurementQA; nothing fitted or tuned on Run 5"},
        "summary": summary, "recipes": sorted(recipes.values(), key=lambda r: r["recipe"]), "vials": vials,
    }, indent=1))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
