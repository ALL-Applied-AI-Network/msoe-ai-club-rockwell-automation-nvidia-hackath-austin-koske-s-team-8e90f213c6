"""Commission the tray profile's cap positions and occupancy threshold from a quiet stretch of tray video.

A tray mounted at an angle to the camera shows each seated vial's cap away from its hole centre
(on the CSI station's Bad tray, about 13 px right and 4-16 px down, growing with the row), so a
yellow check at the hole centre calls every capped vial uncapped. This reads only a window before
any vial of the run arrives (--until), with the station profile's hole centres and empty-hole grey:

  tray_cap_slots:          per slot, hole centre + the cap offset fitted per tray from the vials
                           already seated in the window (dx constant, dy linear in the row);
                           a tray with no seated capped vial keeps its hole centres
  tray_slot_occupied_diff: midpoint between the largest empty-hole difference and the smallest
                           occupied-hole difference seen in the window

    python -m replay.tray_commission --video tray.mp4 --station setup/csi-station.json --until 220 --out profile.json
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from replay.tray import TRAYS, baseline_inventory, caps, profile_slots, validate_empty_slot_gray, validate_tray_slots

MATCH_PX = 25     # a cap belongs to a seated slot when its centroid is this close


def commission(frames, tray_slots, empty_slot_gray):
    """frames: BGR images from the quiet window. Returns (cap_slots, occupied_diff, evidence)."""
    occupied = baseline_inventory(frames[0], tray_slots, empty_slot_gray)
    empty_max, occupied_min = 0.0, float("inf")
    offsets = {tray: [] for tray in TRAYS}
    for bgr in frames:
        found = caps(bgr)
        for tray, centers in tray_slots.items():
            for k, (diff, _) in enumerate(profile_slots(bgr, centers, empty_slot_gray[tray])):
                if occupied[tray][k]:
                    occupied_min = min(occupied_min, diff)
                else:
                    empty_max = max(empty_max, diff)
            # Caps right of / below a seated slot; the nearest seated slot owns each cap.
            for cx, cy in found[tray][1]:
                near = [(np.hypot(cx - x, cy - y), k) for k, (x, y) in enumerate(centers) if occupied[tray][k]]
                if near and min(near)[0] <= MATCH_PX:
                    k = min(near)[1]
                    offsets[tray].append((k // 6, cx - centers[k][0], cy - centers[k][1]))
    cap_slots, fits = {}, {}
    for tray, centers in tray_slots.items():
        rows = np.array(offsets[tray])
        if not len(rows):
            cap_slots[tray], fits[tray] = list(centers), None
            continue
        dx = float(np.median(rows[:, 1]))
        if len(set(rows[:, 0])) > 1:
            slope, intercept = np.polyfit(rows[:, 0], rows[:, 2], 1)
        else:
            slope, intercept = 0.0, float(np.median(rows[:, 2]))
        cap_slots[tray] = [[int(round(x + dx)), int(round(y + intercept + slope * (k // 6)))]
                           for k, (x, y) in enumerate(centers)]
        fits[tray] = dict(dx=round(dx, 1), dy_row0=round(float(intercept), 1), dy_per_row=round(float(slope), 2),
                          samples=len(rows), seated=[k for k, v in enumerate(occupied[tray]) if v])
    threshold = (empty_max + occupied_min) / 2
    evidence = dict(frames=len(frames), empty_max=round(empty_max, 1), occupied_min=round(occupied_min, 1),
                    cap_offset=fits)
    return validate_tray_slots(cap_slots, "tray_cap_slots"), round(threshold, 1), evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--station", type=Path, required=True, help="station profile with tray_slots and tray_empty_slot_gray")
    parser.add_argument("--from", dest="start", type=float, default=0.0, help="window start, s")
    parser.add_argument("--until", type=float, required=True, help="window end, s: before the run's first arrival")
    parser.add_argument("--every", type=float, default=1.0, help="seconds between frames read")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    station = json.loads(args.station.read_text(encoding="utf-8"))
    tray_slots = validate_tray_slots(station.get("tray_slots"))
    empty_slot_gray = validate_empty_slot_gray(station.get("tray_empty_slot_gray"))
    if empty_slot_gray is None:
        parser.error("the station profile needs tray_empty_slot_gray")
    capture = cv2.VideoCapture(str(args.video))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    frames, index = [], 0
    while index / fps < args.until:
        ok, bgr = capture.read()
        if not ok:
            break
        if index / fps >= args.start and index % max(1, round(args.every * fps)) == 0:
            frames.append(bgr)
        index += 1
    capture.release()
    cap_slots, threshold, evidence = commission(frames, tray_slots, empty_slot_gray)
    result = dict(tray_cap_slots={t: [list(p) for p in v] for t, v in cap_slots.items()},
                  tray_slot_occupied_diff=threshold,
                  commissioning=dict(video=str(args.video), window_s=[args.start, args.until], **evidence))
    args.out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps(dict(tray_slot_occupied_diff=threshold, **evidence)))


if __name__ == "__main__":
    main()
