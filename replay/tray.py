"""Finished-vial tracking on the cap_seal camera, reconciled with the fill-station ledger.

    python -m replay.tray --video cap_seal_video_5.mp4 --run DIR [--out DIR/tray]

The cap_seal camera sees the robot place capped vials into a Good tray (left) and a Bad tray
(right). Arrivals come from the 30 independently observed slots in each tray. The first clear
robot-free view is the baseline; occupied slots are reserved from the station empty-slot profile,
and a change from that baseline
in any other slot is committed after STABLE arm-free samples. The vial is "capped" when the patch
is yellow. An occupied slot with no yellow is a vial with a MISSING CAP.
Yellow caps are also counted per tray as a cross-check: a count is committed when stable with no
arm-sized yellow blob over the trays; a new high is an arrival and drops below the high are
occlusion dips.

Arrivals are aligned in order to the fill-station vials in DIR/ledger.json, allowing a vial that
never arrives or an arrival with no vial (the cameras share a clock in the CSI dataset; Run 5
burned-in clocks agree within ~20 ms). The reconciliation
reports counts in vs out, transit time, vials still in transit at the end of the video, and every
vial whose tray disagrees with our action (e.g. we said SCRAP and it went to Good).
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

TRAYS = {"good": (210, 140, 395, 330), "bad": (400, 230, 570, 415)}   # x0, y0, x1, y1 at 640x480
YELLOW = ((18, 110, 120), (38, 255, 255))                              # OpenCV HSV
CAP_PX = 300           # area of one cap blob; touching caps are counted as round(area / CAP_PX)
MIN_BLOB = 120
ARM_BLOB = 1000        # connected changed pixels vs the clear baseline: robot arm occlusion
STEP = 10              # frames between samples (1/3 s at 30 fps)
STABLE = 6             # samples with the same counts and no arm before a count is committed
MATCH_PX = 10          # cap centroids closer than this are the same cap (for arrival positions)
GOOD_SLOTS = [
    (228, 165), (257, 165), (286, 163), (314, 161), (344, 160), (372, 160),
    (229, 199), (257, 197), (287, 197), (316, 196), (345, 194), (373, 196),
    (230, 233), (259, 232), (288, 231), (317, 230), (346, 229), (375, 229),
    (231, 267), (261, 266), (289, 266), (318, 264), (347, 264), (376, 263),
    (233, 301), (262, 300), (290, 300), (319, 299), (347, 298), (377, 297),
]
# Current CSI camera profile candidate. Perspective makes this grid slightly trapezoidal, so the
# measured hole centres are explicit rather than generated from one rectangular pitch.
BAD_SLOTS = [
    (421, 256), (449, 254), (475, 253), (502, 252), (528, 250), (553, 250),
    (423, 289), (450, 288), (476, 287), (503, 285), (528, 283), (552, 282),
    (423, 323), (450, 321), (478, 320), (504, 317), (529, 316), (552, 314),
    (423, 356), (451, 354), (478, 352), (503, 350), (529, 348), (551, 345),
    (424, 389), (450, 386), (478, 384), (504, 383), (528, 380), (551, 376),
]
TRAY_SLOTS = {"good": GOOD_SLOTS, "bad": BAD_SLOTS}
SLOT_HALF = 10         # slot patch is 20 x 20 px
OCCUPIED_DIFF = 22.0   # mean abs BGR difference to the empty tray (empty ~8-10, occupied >= 35 on Run 5)
SLOT_STABLE = 15       # 5 s: the grey gripper can sit over a slot for ~3 s without being flagged as arm
CAPPED_YELLOW = 0.15   # yellow fraction of the slot patch (uncapped 0.00-0.10, capped 0.26-0.77 on Run 5)
BASELINE_CAPPED_YELLOW = 0.55  # current station tray metal reaches 0.32 under warm lighting
CLEAR_DARK_LEVEL = 100         # median grey at the hole center when the grid is unobstructed
CLEAR_MIN_SLOTS = {"good": 26, "bad": 25}  # camera-profile clear-view signature per grid
CLEAR_THUMBNAIL_SHAPE = (12, 16)
CLEAR_THUMBNAIL_ROI = (180, 0, 580, 420)
CLEAR_THUMBNAIL_MAE = 5.0      # rejects the robot-over-tray scene; allows individual placements
BASELINE_INVENTORY_GRAY = 30.0 # center-grey change from the station's empty-slot profile


def caps(bgr):
    """Per tray: (count, centroids, arm_present)."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.morphologyEx(cv2.inRange(hsv, *YELLOW), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    out = {}
    for tray, (x0, y0, x1, y1) in TRAYS.items():
        _, _, stats, centroids = cv2.connectedComponentsWithStats(mask[y0:y1, x0:x1])
        count, points, arm = 0, [], False
        for (_, _, _, _, area), (cx, cy) in zip(stats[1:], centroids[1:]):
            if area > ARM_BLOB:
                arm = True
            elif area >= MIN_BLOB:
                count += max(1, round(area / CAP_PX))
                points.append((round(cx + x0, 1), round(cy + y0, 1)))
        out[tray] = (count, points, arm)
    return out


def slots(bgr, baseline, centers=GOOD_SLOTS):
    """Per configured tray slot: (difference to the session baseline, yellow fraction)."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    out = []
    for x, y in centers:
        window = np.s_[y - SLOT_HALF:y + SLOT_HALF, x - SLOT_HALF:x + SLOT_HALF]
        diff = float(np.abs(bgr[window].astype(np.int16) - baseline[window].astype(np.int16)).mean())
        out.append((round(diff, 1), round(float((cv2.inRange(hsv[window], *YELLOW) > 0).mean()), 2)))
    return out


class TrayScanner:
    """One sample every STEP frames. The first arm-free frame at/after empty_frame is the baseline."""

    def __init__(self, fps, empty_frame=0, tray_slots=None, clear_thumbnail=None,
                 empty_slot_gray=None, cap_slots=None, occupied_diff=None):
        self.fps, self.empty_frame, self.empty, self.next_sample = fps, empty_frame, None, None
        self.tray_slots = validate_tray_slots(tray_slots)
        # Tilted tray: where each seated vial's cap appears (station profile), else the hole centre.
        self.cap_slots = validate_tray_slots(cap_slots, "tray_cap_slots") if cap_slots else self.tray_slots
        self.occupied_diff = float(occupied_diff) if occupied_diff is not None else BASELINE_INVENTORY_GRAY
        self.clear_thumbnail = validate_clear_thumbnail(clear_thumbnail)
        self.empty_slot_gray = validate_empty_slot_gray(empty_slot_gray)
        self.baseline_occupied = None

    def push(self, index, bgr):
        if self.empty is None and index >= self.empty_frame:
            # Warm metal in this camera looks yellow, so absolute yellow area cannot distinguish
            # the robot arm. Capture only when most physical holes in both configured grids are
            # dark/visible. Existing vials can occupy up to five slots and are reserved below.
            # A station empty-slot profile makes every slot independent: open holes can become
            # references immediately while changed holes are starting inventory. Legacy profiles
            # without per-slot evidence still need the former global clear-view gate.
            if self.empty_slot_gray is None and not clear_view(bgr, self.tray_slots, self.clear_thumbnail):
                return None
            self.empty = bgr.copy()
            self.baseline_occupied = baseline_inventory(
                bgr, self.tray_slots, self.empty_slot_gray, self.occupied_diff) if self.empty_slot_gray is not None else None
            self.next_sample = index + (-index) % STEP
        # Next multiple of STEP at or after this frame, so dropped live frames do not stop sampling.
        if self.empty is None or index < self.next_sample:
            return None
        self.next_sample = index + STEP - index % STEP
        per = caps(bgr)
        measured = {
            tray: (profile_slots(bgr, centers, self.empty_slot_gray[tray], self.cap_slots[tray])
                   if self.empty_slot_gray is not None else slots(bgr, self.empty, centers))
            for tray, centers in self.tray_slots.items()
        }
        return {"frame": index, "t": index / self.fps,
                **{f"{k}_count": v[0] for k, v in per.items()},
                **{f"{k}_caps": v[1] for k, v in per.items()},
                # Per-slot stability rejects the ~3 s gripper visit. Do not globally freeze all
                # slots merely because the yellow robot remains visible elsewhere in the scene.
                "arm": False if self.empty_slot_gray is not None else arm_over_trays(bgr, self.empty),
                "slots_against_empty_profile": self.empty_slot_gray is not None,
                "slot_occupied_diff": self.occupied_diff,
                # `slots` is the legacy Good-tray field retained for old review files.
                "slots": measured["good"], "good_slots": measured["good"],
                "bad_slots": measured["bad"],
                **({f"{tray}_baseline_occupied": values
                    for tray, values in self.baseline_occupied.items()}
                   if self.baseline_occupied is not None else {})}


def validate_tray_slots(config=None, name="tray_slots"):
    """Return a validated 640x480 two-tray profile, preserving configured point order."""
    raw = config or TRAY_SLOTS
    if set(raw) != set(TRAYS):
        raise ValueError(f"Station {name} must contain good and bad grids")
    out = {}
    for tray in TRAYS:
        points = raw[tray]
        if not isinstance(points, (list, tuple)) or len(points) != 30:
            raise ValueError(f"Station {name}.{tray} must contain 30 [x, y] points")
        clean = []
        for point in points:
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                raise ValueError(f"Station {name}.{tray} points must be [x, y]")
            x, y = map(int, point)
            if not (SLOT_HALF <= x < 640 - SLOT_HALF and SLOT_HALF <= y < 480 - SLOT_HALF):
                raise ValueError(f"Station {name}.{tray} point is outside the native 640x480 profile")
            clean.append((x, y))
        out[tray] = clean
    return out


def validate_clear_thumbnail(config=None):
    if config is None:
        return None
    thumbnail = np.asarray(config, dtype=np.uint8)
    if thumbnail.shape != CLEAR_THUMBNAIL_SHAPE:
        raise ValueError("Station tray_clear_thumbnail must be a 12x16 grayscale array")
    return thumbnail


def validate_empty_slot_gray(config=None):
    if config is None:
        return None
    if set(config) != set(TRAYS):
        raise ValueError("Station tray_empty_slot_gray must contain good and bad grids")
    out = {}
    for tray in TRAYS:
        values = config[tray]
        if not isinstance(values, (list, tuple)) or len(values) != 30:
            raise ValueError(f"Station tray_empty_slot_gray.{tray} must contain 30 values")
        clean = [float(value) for value in values]
        if not all(0 <= value <= 255 for value in clean):
            raise ValueError(f"Station tray_empty_slot_gray.{tray} values must be 0..255")
        out[tray] = clean
    return out


def baseline_inventory(bgr, tray_slots, empty_slot_gray, occupied_diff=BASELINE_INVENTORY_GRAY):
    """Reserve arbitrary starting slots by comparing them with the station's empty-hole profile."""
    grey = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    out = {}
    for tray, centers in tray_slots.items():
        # Auto-exposure changes every empty hole in the same direction.  Remove that common
        # shift before deciding which individual slots differ from the commissioned profile;
        # otherwise an untouched empty grid can turn into a row of uncapped-vial detections.
        differences = profile_gray_differences(grey, centers, empty_slot_gray[tray])
        out[tray] = [bool(difference > occupied_diff) for difference in differences]
    return out


def profile_gray_differences(grey, centers, expected_gray):
    """Per-slot residual after removing the tray's common illumination/exposure shift."""
    current = np.asarray(
        [float(np.median(grey[y - 5:y + 5, x - 5:x + 5])) for x, y in centers],
        dtype=np.float32,
    )
    signed = current - np.asarray(expected_gray, dtype=np.float32)
    # Fewer than half the slots are expected to change at once during a six-vial run, making
    # the median a robust estimate even while the robot or several vials are in the grid.
    illumination_shift = float(np.median(signed))
    return np.abs(signed - illumination_shift)


def profile_slots(bgr, centers, expected_gray, cap_centers=None):
    """Per slot: direct difference from the station empty-hole signature and yellow fraction.
    Yellow is read where a seated vial's cap appears (cap_centers), which differs from the hole
    centre when the tray is tilted to the camera; by default it is the hole centre."""
    grey = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    differences = profile_gray_differences(grey, centers, expected_gray)
    out = []
    for (x, y), difference in zip(cap_centers or centers, differences):
        window = np.s_[y - SLOT_HALF:y + SLOT_HALF, x - SLOT_HALF:x + SLOT_HALF]
        yellow = float((cv2.inRange(hsv[window], *YELLOW) > 0).mean())
        out.append((round(float(difference), 1), round(yellow, 2)))
    return out


def clear_view(bgr, tray_slots, clear_thumbnail=None):
    """Whether both profiled grids are sufficiently visible for baseline capture."""
    grey = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if clear_thumbnail is not None:
        x0, y0, x1, y1 = CLEAR_THUMBNAIL_ROI
        current = cv2.resize(grey[y0:y1, x0:x1], (CLEAR_THUMBNAIL_SHAPE[1], CLEAR_THUMBNAIL_SHAPE[0]),
                             interpolation=cv2.INTER_AREA)
        return float(np.abs(current.astype(np.int16) - clear_thumbnail.astype(np.int16)).mean()) <= CLEAR_THUMBNAIL_MAE
    # Legacy profiles without a scene thumbnail require mostly visible empty holes.
    for tray, centers in tray_slots.items():
        visible = sum(float(np.median(grey[y - 5:y + 5, x - 5:x + 5])) < CLEAR_DARK_LEVEL
                      for x, y in centers)
        if visible < CLEAR_MIN_SLOTS[tray]:
            return False
    return True


def arm_over_trays(bgr, baseline):
    """Large connected scene change in either tray; individual vial changes stay below the gate."""
    difference = cv2.cvtColor(cv2.absdiff(bgr, baseline), cv2.COLOR_BGR2GRAY)
    changed = cv2.morphologyEx((difference > 35).astype(np.uint8), cv2.MORPH_OPEN,
                               np.ones((3, 3), np.uint8))
    for x0, y0, x1, y1 in TRAYS.values():
        _, _, stats, _ = cv2.connectedComponentsWithStats(changed[y0:y1, x0:x1])
        if any(int(stat[4]) > ARM_BLOB for stat in stats[1:]):
            return True
    return False


def scan(video, empty_frame=0):
    """Samples every STEP frames. The frame at empty_frame must show an empty Good tray."""
    capture = cv2.VideoCapture(str(video))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    scanner, samples, index = TrayScanner(fps, empty_frame), [], 0
    while True:
        ok, bgr = capture.read()
        if not ok:
            break
        sample = scanner.push(index, bgr)
        if sample:
            samples.append(sample)
        index += 1
    capture.release()
    return samples, fps


def _new_points(before, after):
    return [p for p in after if all(np.hypot(p[0] - q[0], p[1] - q[1]) > MATCH_PX for q in before)]


class CapTracker:
    """Committed tray states -> events, one sample at a time. An arrival is a new high in a tray's
    committed count, so a cap that is briefly occluded (count drops, then returns) is a dip, not a
    removal plus arrival."""

    def __init__(self):
        self.committed = {tray: (0, []) for tray in TRAYS}
        self.high = {tray: 0 for tray in TRAYS}
        self.run = []
        self.initialized = False

    def push(self, s):
        key = None if s["arm"] else tuple(s[f"{t}_count"] for t in TRAYS)
        if key is None:
            self.run = []
            return []
        run = self.run = self.run + [s] if self.run and key == tuple(self.run[0][f"{t}_count"] for t in TRAYS) else [s]
        if len(run) != STABLE:
            return []
        if not self.initialized:
            # Counts visible when the session starts are inventory, not arrivals from this run.
            for tray in TRAYS:
                count, points = run[-1][f"{tray}_count"], run[-1][f"{tray}_caps"]
                self.committed[tray] = (count, points)
                self.high[tray] = count
            self.initialized = True
            return []
        out = []
        for tray in TRAYS:
            count, points = run[-1][f"{tray}_count"], run[-1][f"{tray}_caps"]
            old_count, old_points = self.committed[tray]
            if count > self.high[tray]:
                # Arrival time: first sample of the stable run (the arm has just left).
                new = _new_points(old_points, points)
                for i in range(count - self.high[tray]):
                    out.append({"kind": "arrival", "tray": tray, "t": run[0]["t"], "frame": run[0]["frame"],
                                "position": new[i] if i < len(new) else None, "tray_count": self.high[tray] + i + 1})
                self.high[tray] = count
            elif count != old_count:
                out.append({"kind": "dip" if count < old_count else "recovered", "tray": tray, "t": run[0]["t"],
                            "frame": run[0]["frame"], "tray_count": count, "high": self.high[tray]})
            self.committed[tray] = (count, points)
        return out

    def counts(self):
        return {tray: self.committed[tray][0] for tray in TRAYS}


def events(samples):
    tracker = CapTracker()
    return [e for s in samples for e in tracker.push(s)], tracker.counts()


class SlotTracker:
    """One tray's slot arrivals, one sample at a time. Arm samples are skipped (the arm occludes
    slots); a slot is committed occupied (or vacated) after SLOT_STABLE consecutive arm-free
    samples in the new state."""

    def __init__(self, tray="good", centers=None):
        if tray not in TRAYS:
            raise ValueError(f"Unknown tray {tray}")
        self.tray = tray
        self.centers = list(centers or TRAY_SLOTS[tray])
        self.sample_key = "slots" if tray == "good" else f"{tray}_slots"
        self.baseline_key = f"{tray}_baseline_occupied"
        self.state = [False] * len(self.centers)
        self.baseline_occupied = [False] * len(self.centers)
        self.capped = [None] * len(self.centers)   # per occupied slot, for the operator preview
        self.runs = [[] for _ in self.centers]
        self.initialized = False

    def push(self, s):
        values = s.get(self.sample_key)
        if values is None or s["arm"]:
            return []
        if not self.initialized:
            # A capped vial already visible in the baseline reserves that slot. It must not be
            # matched to a vial that leaves the fill station after this session begins.
            declared = s.get(self.baseline_key)
            self.baseline_occupied = ([bool(value) for value in declared] if declared is not None else
                                      [yellow >= BASELINE_CAPPED_YELLOW for _, yellow in values])
            self.state = list(self.baseline_occupied)
            self.capped = [yellow >= CAPPED_YELLOW if occupied else None
                           for occupied, (_, yellow) in zip(self.state, values)]
            self.initialized = True
            return []
        out, state, runs = [], self.state, self.runs
        for k, (diff, yellow) in enumerate(values):
            # Empty baseline slots become occupied when they differ. For a slot occupied in the
            # baseline, similarity means the original vial is still there and difference means it
            # was vacated. Every slot is evaluated independently; there is no R1C1 start rule.
            if s.get("slots_against_empty_profile"):
                occupied = diff > float(s.get("slot_occupied_diff", BASELINE_INVENTORY_GRAY))
            else:
                occupied = diff <= OCCUPIED_DIFF if self.baseline_occupied[k] else diff > OCCUPIED_DIFF
            runs[k] = runs[k] + [(s, yellow)] if occupied != state[k] else []
            if len(runs[k]) == SLOT_STABLE:
                first = runs[k][0][0]
                state[k] = occupied
                row, col = divmod(k, 6)
                event = {"kind": "arrival" if occupied else "vacated", "tray": self.tray, "t": first["t"],
                         "frame": first["frame"], "slot": f"R{row + 1}C{col + 1}", "position": self.centers[k]}
                if occupied:
                    event["yellow"] = float(np.median([y for _, y in runs[k]]))
                    event["capped"] = event["yellow"] >= CAPPED_YELLOW
                self.capped[k] = event.get("capped")
                out.append(event)
                runs[k] = []
        return out


def slot_events(samples, tray="good", centers=None):
    tracker = SlotTracker(tray, centers)
    return [e for s in samples for e in tracker.push(s)], sum(tracker.state)


def render_overlay(bgr, trackers, baseline_ready, sample=None, routing=True):
    """Operator preview: the raw tray camera with a box on each occupied slot. With routing the
    box carries the tray's colour and name; without it (the station's own sorter fills the trays)
    the box shows only whether the vial has a cap."""
    rendered = bgr.copy()
    colors = {"good": (80, 220, 80), "bad": (80, 100, 255)}
    capped_color, no_cap_color = (255, 255, 255), (0, 180, 255)
    no_cap = 0
    for tray, tracker in trackers.items():
        for index, ((x, y), occupied) in enumerate(zip(tracker.centers, tracker.state)):
            if not occupied:
                continue
            row, col = divmod(index, 6)
            if routing:
                color, label = colors[tray], f"{tray[0].upper()}{row + 1}:{col + 1}"
            elif tracker.capped[index] is False:
                color, label = no_cap_color, "NO CAP"
                no_cap += 1
            else:
                color, label = capped_color, None
            cv2.rectangle(rendered, (x - 13, y - 13), (x + 13, y + 13), color, 2)
            if label:
                cv2.putText(rendered, label, (x - 16, y - 17), cv2.FONT_HERSHEY_SIMPLEX, .34, color, 1, cv2.LINE_AA)
    if not baseline_ready:
        message, color = "WAITING FOR CLEAR VIEW OF THE TRAYS", (0, 180, 255)
    elif sample and sample.get("arm"):
        message, color = "ROBOT ARM OCCLUDING TRAYS - DETECTIONS HELD", (0, 180, 255)
    elif routing:
        counts = {tray: sum(tracker.state) for tray, tracker in trackers.items()}
        message, color = f"VIAL BOXES  GOOD {counts['good']}  BAD {counts['bad']}", (255, 255, 255)
    else:
        total = sum(sum(tracker.state) for tracker in trackers.values())
        message, color = f"VIALS {total}  NO CAP {no_cap}", (255, 255, 255)
    cv2.putText(rendered, message, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, .5, color, 2, cv2.LINE_AA)
    return rendered


EXPECTED_TRAY = {"PASS": "good", "SCRAP": "bad", "HOLD": "bad"}
TRANSIT_WINDOW = (30.0, 400.0)   # seconds from leaving the fill station to the tray
TRANSIT_SD = 30.0
SKIP_COST = 4.0                  # in squared standard deviations


def align(departures, arrivals, video_end_s):
    """Order-preserving match of vials to arrivals; vials or arrivals may be skipped.
    Typical transit is the median of the first FIFO pairs. Returns ({vial: arrival}, transit)."""
    first = [a - d for d, a in zip(departures[:8], arrivals[:8]) if TRANSIT_WINDOW[0] < a - d < TRANSIT_WINDOW[1]]
    typical = float(np.median(first)) if first else 140.0
    n, m = len(departures), len(arrivals)
    cost = np.full((n + 1, m + 1), np.inf)
    move = {}
    cost[0, :] = np.arange(m + 1) * SKIP_COST
    cost[:, 0] = np.arange(n + 1) * SKIP_COST
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            options = [(cost[i - 1, j] + SKIP_COST, "skip_vial"), (cost[i, j - 1] + SKIP_COST, "skip_arrival")]
            transit = arrivals[j - 1] - departures[i - 1]
            if TRANSIT_WINDOW[0] < transit < TRANSIT_WINDOW[1]:
                options.append((cost[i - 1, j - 1] + ((transit - typical) / TRANSIT_SD) ** 2, "match"))
            cost[i, j], move[i, j] = min(options)
    # Trailing vials that could still be in transit when the video ends may be unmatched for free.
    in_transit = sum(video_end_s - d < typical + 3 * TRANSIT_SD for d in departures)
    i, j = min(range(n - in_transit, n + 1), key=lambda r: cost[r, m]), m
    pairs = {}
    while i > 0 and j > 0:
        step = move[i, j]
        if step == "match":
            pairs[i - 1] = j - 1
            i, j = i - 1, j - 1
        elif step == "skip_vial":
            i -= 1
        else:
            j -= 1
    return pairs, typical


def reconcile(arrivals, ledger, video_end_s):
    """Which tray each fill-station vial reached, and whether that agrees with our decision."""
    vials = sorted(ledger["vials"], key=lambda v: v["seq"])
    pairs, typical = align([v["departure_s"] for v in vials], [a["t"] for a in arrivals], video_end_s)
    last_matched = max(pairs, default=-1)
    rows = []
    for i, vial in enumerate(vials):
        kinds = [a["kind"] for a in vial["actions"]]
        decision = next((k for k in ("SCRAP", "HOLD", "PASS") if k in kinds), None)
        arrival = arrivals[pairs[i]] if i in pairs else None
        row = {"seq": vial["seq"], "recipe_id": vial["recipe_id"], "verdict": vial["verdict"], "decision": decision,
               "left_fill_s": round(vial["departure_s"], 1), "tray": arrival and arrival["tray"],
               "tray_arrival_s": arrival and round(arrival["t"], 1), "position": arrival and arrival["position"],
               "transit_s": arrival and round(arrival["t"] - vial["departure_s"], 1)}
        if arrival:
            row["slot"], row["capped"] = arrival.get("slot"), arrival.get("capped")
            expected = EXPECTED_TRAY.get(decision)
            row["status"] = "routed_as_decided" if expected == arrival["tray"] else \
                "escaped" if decision in ("SCRAP", "HOLD") else "good_vial_in_bad_tray"
        elif i > last_matched and video_end_s - vial["departure_s"] < typical + 3 * TRANSIT_SD:
            row["status"] = "in_transit_at_end"
        else:
            row["status"] = "not_seen_arriving"
        rows.append(row)
    transit = [r["transit_s"] for r in rows if r["transit_s"] is not None]
    by = lambda status: [r["seq"] for r in rows if r["status"] == status]
    summary = {
        "vials_left_fill_station": len(vials), "tray_arrivals": len(arrivals), "matched": len(pairs),
        "typical_transit_s": round(typical, 1),
        "transit_s_median": round(float(np.median(transit)), 1) if transit else None,
        "transit_s_range": [min(transit), max(transit)] if transit else None,
        "unmatched_arrivals": [round(a["t"], 1) for j, a in enumerate(arrivals) if j not in pairs.values()],
        "not_seen_arriving": by("not_seen_arriving"),
        "in_transit_at_end": by("in_transit_at_end"),
        "routed_as_decided": by("routed_as_decided"),
        "escaped_to_good": {k: [r["seq"] for r in rows if r["status"] == "escaped" and r["decision"] == k]
                            for k in ("SCRAP", "HOLD")},
        "good_vial_in_bad_tray": by("good_vial_in_bad_tray"),
        "missing_cap": [r["seq"] for r in rows if r.get("capped") is False],
    }
    return summary, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", type=Path, required=True, help="cap_seal video, same clock as the fill video")
    parser.add_argument("--run", type=Path, required=True, help="output directory of python -m replay.line")
    parser.add_argument("--out", type=Path, help="default: RUN/tray")
    parser.add_argument("--samples", type=Path, help="tray-samples.jsonl from an earlier scan")
    parser.add_argument("--empty-frame", type=int, default=0,
                        help="first candidate baseline frame; the scanner waits until the arm is clear")
    args = parser.parse_args()
    out = args.out or args.run / "tray"
    out.mkdir(parents=True, exist_ok=True)

    if args.samples:
        samples = [json.loads(line) for line in args.samples.read_text().splitlines()]
        fps = 30.0
    else:
        samples, fps = scan(args.video, args.empty_frame)
        with (out / "tray-samples.jsonl").open("w") as handle:
            for s in samples:
                handle.write(json.dumps(s) + "\n")
    found, final = events(samples)
    good_found, good_slots = slot_events(samples, "good")
    has_bad_grid = any("bad_slots" in sample for sample in samples)
    bad_found, bad_slots = slot_events(samples, "bad") if has_bad_grid else ([], 0)
    # New observations use slot occupancy for both trays, including uncapped vials. Historical
    # samples have no Bad grid and retain the former yellow-cap arrival fallback.
    arrivals = sorted([e for e in good_found + bad_found if e["kind"] == "arrival"] +
                      ([] if has_bad_grid else
                       [e for e in found if e["kind"] == "arrival" and e["tray"] == "bad"]), key=lambda e: e["t"])
    with (out / "tray-events.jsonl").open("w") as handle:
        for e in sorted(found + good_found + bad_found, key=lambda e: e["t"]):
            handle.write(json.dumps(e) + "\n")
    ledger = json.loads((args.run / "ledger.json").read_text())
    summary, rows = reconcile(arrivals, ledger, samples[-1]["t"] if samples else 0)
    summary["final_cap_counts"] = final
    summary["final_good_slots_occupied"] = good_slots
    summary["final_bad_slots_occupied"] = bad_slots if has_bad_grid else None
    summary["bad_slot_grid_available"] = has_bad_grid
    summary["good_tray_cap_arrivals"] = sum(e["kind"] == "arrival" and e["tray"] == "good" for e in found)
    summary["cap_count_dips"] = sum(e["kind"] == "dip" for e in found)
    summary["slots_vacated"] = [{"tray": e["tray"], "slot": e["slot"]}
                                 for e in good_found + bad_found if e["kind"] == "vacated"]
    (out / "reconciliation.json").write_text(json.dumps({"summary": summary, "vials": rows}, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
