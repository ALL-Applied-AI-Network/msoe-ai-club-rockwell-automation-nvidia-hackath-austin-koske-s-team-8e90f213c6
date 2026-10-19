"""Validate saved RGB/ID/mask/box linkage and create local PNG review sheets."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from box_annotations import CLASSES, COLORS, instance_mask, mask_box, sha256, write_json, yolo_line


def review(out, real_frame=None):
    out = Path(out).resolve()
    run = json.loads((out / "run.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in (out / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    if not run["complete"] or len(rows) != run["captured"] or run["captured"] != run["requested"]:
        raise ValueError("Incomplete capture")
    if run["run5_used"] or run["training_eligible"] or not run["camera_parameters_unchanged"] or not run["source_assets_unchanged"]:
        raise ValueError("Unexpected provenance/eligibility")
    names, rgb_hashes = set(), set()
    totals = {name: 0 for name in CLASSES}
    unavailable = []
    masks_by_name = {}
    case_checks = []
    for row in rows:
        name = row["sample_id"]
        if name in names or row["training_eligible"]:
            raise ValueError("Duplicate case or automatic training eligibility")
        names.add(name)
        rgb_file = out / row["image_file"]
        with Image.open(rgb_file) as image:
            image_size = image.size
        if sha256(rgb_file) != row["image_sha256"] or image_size != (640, 480):
            raise ValueError("RGB integrity failure")
        rgb_hashes.add(row["image_sha256"])
        lines = []
        composite = np.zeros((480, 640, 3), dtype=np.uint8)
        for obj in row["annotations"]:
            mask_file = out / obj["mask_file"]
            if sha256(mask_file) != obj["mask_sha256"]:
                raise ValueError("Mask integrity failure")
            with Image.open(mask_file) as image:
                pixels = np.asarray(image).copy()
            if pixels.shape != (480, 640) or not set(np.unique(pixels)).issubset({0, 255}):
                raise ValueError("Expected full-size binary mask")
            mask = pixels > 0
            pass_name = obj["label_pass"]
            id_file = out / "label-passes" / f"{name}-{pass_name}-ids.npy"
            mapping_file = out / "label-passes" / f"{name}-{pass_name}-mapping.json"
            expected_mask = instance_mask(dict(data=np.load(id_file, allow_pickle=False), info=dict(
                idToLabels=json.loads(mapping_file.read_text(encoding="utf-8")))), obj["prim_path"])
            if not np.array_equal(mask, expected_mask):
                raise ValueError("Saved mask differs from renderer instance IDs")
            box = mask_box(mask)
            if box != obj["bbox_xyxy"] or int(mask.sum()) != obj["visible_pixels"]:
                raise ValueError("Mask/box/pixel-count mismatch")
            if not obj["present"] and box is not None:
                raise ValueError("Absent object has annotation")
            projected = obj["projected_geometry_bbox_xyxy"]
            if box and (box[0] < projected[0]-2 or box[1] < projected[1]-2 or
                        box[2] > projected[2]+2 or box[3] > projected[3]+2):
                raise ValueError(f"Mask exceeds independently projected geometry: {name} {obj['class_name']}: {box} vs {projected}")
            if box:
                totals[obj["class_name"]] += 1
                lines.append(yolo_line(obj["class_name"], box))
                composite[mask] = COLORS[obj["class_name"]]
            elif obj["present"]:
                unavailable.append(dict(sample_id=name, instance_id=obj["instance_id"], status=obj["status"]))
            masks_by_name[(name, obj["instance_id"])] = mask
        actual_lines = (out / "labels" / (name + ".txt")).read_text(encoding="utf-8").splitlines()
        if actual_lines != lines:
            raise ValueError("YOLO export differs from manifest boxes")
        Image.fromarray(composite).save(out / "masks" / (name + "-composite.png"))
    if len(rgb_hashes) != len(rows):
        raise ValueError("Repeated RGB capture; inspect for stale images")
    if "00-empty" in names:
        for instance in [201, 301]:
            if masks_by_name[("00-empty", instance)].any():
                raise ValueError("Empty negative has content mask")
        case_checks.append("empty case")
    if "01-first-stream" in names and (masks_by_name[("01-first-stream", 201)].any() or
                                       not masks_by_name[("01-first-stream", 301)].any()):
        raise ValueError("Stream-only negative/positive boundary failed")
    if "01-first-stream" in names:
        case_checks.append("stream-only case")
    if {"02-tiny-blue", "09-high-blue-stopped"}.issubset(names):
        if masks_by_name[("02-tiny-blue", 201)].sum() >= masks_by_name[("09-high-blue-stopped", 201)].sum():
            raise ValueError("Liquid-height mask progression failed")
        case_checks.append("liquid height progression")
    appearance_checks = []
    # File hashes alone cannot distinguish scene changes from Monte Carlo noise.
    # Check a deliberate large material change inside the shared liquid mask.
    if {"07-mid-blue", "08-mid-pale"}.issubset(names):
        region = masks_by_name[("07-mid-blue", 201)] & masks_by_name[("08-mid-pale", 201)]
        if not region.any():
            raise ValueError("No shared liquid region for appearance check")
        with Image.open(out/"rgb/07-mid-blue.png") as image:
            blue = np.asarray(image).astype(float)
        with Image.open(out/"rgb/08-mid-pale.png") as image:
            pale = np.asarray(image).astype(float)
        difference = float(np.abs(blue[region]-pale[region]).mean())
        if difference < 3:
            raise ValueError("Liquid material change did not visibly change RGB; inspect stale capture/materials")
        appearance_checks.append(dict(cases=["07-mid-blue", "08-mid-pale"],
            shared_liquid_pixels=int(region.sum()), mean_absolute_rgb_difference=difference,
            minimum_required=3, scope="stale/material-change capture check, not optical calibration"))

    def sheet(folder, filename):
        cols, thumb_w, thumb_h, caption = 3, 480, 360, 28
        height = ((len(rows)+cols-1)//cols) * (thumb_h+caption)
        canvas = Image.new("RGB", (cols*thumb_w, height), "#181c21")
        draw = ImageDraw.Draw(canvas)
        for index, row in enumerate(rows):
            image_path = out / folder / (row["sample_id"] + ("-composite.png" if folder == "masks" else ".png"))
            with Image.open(image_path) as source:
                tile = source.convert("RGB").resize((thumb_w, thumb_h))
            x, y = index % cols * thumb_w, index // cols * (thumb_h+caption)
            canvas.paste(tile, (x, y))
            draw.text((x+6, y+thumb_h+6), row["sample_id"], fill="white")
        canvas.save(out / filename)
    sheet("rgb", "contact-rgb.png")
    sheet("overlays", "contact-boxes.png")
    sheet("masks", "contact-masks.png")
    for row in rows:
        name = row["sample_id"]
        canvas = Image.new("RGB", (1920, 510), "#181c21")
        draw = ImageDraw.Draw(canvas)
        for i, (folder, caption) in enumerate([( "rgb", "RENDERED RGB"), ("overlays", "BOXES FROM GEOMETRIC MASKS"),
                                              ("masks", "MASKS: vial green / liquid orange / stream cyan")]):
            filename = name + ("-composite.png" if folder == "masks" else ".png")
            with Image.open(out / folder / filename) as image:
                canvas.paste(image.convert("RGB"), (i*640, 30))
            draw.text((i*640+6, 8), caption, fill="white")
        canvas.save(out / "overlays" / (name + "-review.png"))
    if real_frame:
        real_frame = Path(real_frame).resolve()
        with Image.open(real_frame) as real:
            if real.size != (640, 480):
                raise ValueError("Real alignment frame must be native 640x480")
            real_rgb = real.convert("RGB")
        chosen = next((r for r in rows if r["sample_id"] == "07-mid-blue"), rows[0])
        with Image.open(out / chosen["image_file"]) as rendered:
            rendered = rendered.convert("RGB")
            comparison = Image.new("RGB", (1920, 520), "#181c21")
            for i, im in enumerate([real_rgb, rendered, Image.blend(real_rgb, rendered, 0.5)]):
                comparison.paste(im, (i*640, 40))
            draw = ImageDraw.Draw(comparison)
            for i, text in enumerate(["RUN 1 / frame 2100 / 70.000 s MP4", "SUPPLIED FillCam / unchanged pose + intrinsics", "50% overlay / visual alignment review"]):
                draw.text((i*640+6, 10), text, fill="white")
            comparison.save(out / "camera-alignment.png")
        write_json(out / "alignment-reference.json", dict(source_run=1, frame_index_zero_based=2100,
            mp4_timestamp_seconds=70.0, image_sha256=sha256(real_frame), image_file=str(real_frame),
            usage="visual alignment reference only; no optimization, label fitting, or Run 5 input"))
    result = dict(passed=True, samples=len(rows), unique_rgb_images=len(rgb_hashes), visible_boxes=totals,
        present_but_unavailable=unavailable, mask_source="Replicator instance_segmentation_fast from documented label passes",
        appearance_checks=appearance_checks,
        checks=["RGB/mask SHA256", "native image dimensions", "binary mask equals saved renderer IDs",
                "box equals mask bounds", "box within independent camera projection +/- 2px",
                "YOLO rows equal manifest", "absent object has no mask",
                "RGB captures unique", "source/camera preservation"] + case_checks,
        optical_accuracy_validated=False, real_detection_accuracy_measured=False, training_eligible=False)
    write_json(out / "verification.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("--real-frame", type=Path)
    args = parser.parse_args()
    print(json.dumps(review(args.out, args.real_frame), indent=2))
