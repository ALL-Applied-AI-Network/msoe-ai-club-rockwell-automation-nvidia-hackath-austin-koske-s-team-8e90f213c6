"""Finished-vial tracking on the cap_seal camera, reconciled with the fill-station ledger.

    python -m replay.tray --video cap_seal_video_5.mp4 --run DIR [--out DIR/tray]

The cap_seal camera sees the robot place capped vials into a Good tray (left) and a Bad tray
(right). Good-tray arrivals come from its 30 slots: a slot is occupied when its patch differs from
the empty tray at the start of the video, committed after STABLE arm-free samples, and the vial is
"capped" when the patch is yellow. An occupied slot with no yellow is a vial with a MISSING CAP.
Yellow caps are also counted per tray (the Bad tray has no slot grid yet, and the Good-tray count
is a cross-check): a count is committed when stable with no arm-sized yellow blob over the trays;
a new high is an arrival and drops below the high are occlusion dips.

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
ARM_BLOB = 1000        # any larger yellow blob in a tray region means the arm is over it
STEP = 10              # frames between samples (1/3 s at 30 fps)
STABLE = 6             # samples with the same counts and no arm before a count is committed
MATCH_PX = 10          # cap centroids closer than this are the same cap (for arrival positions)
GOOD_SLOTS = [(221 + 32 * c, y) for y in (157, 194, 236, 271, 309) for c in range(6)]   # 5 x 6 slot centres
SLOT_HALF = 10         # slot patch is 20 x 20 px
OCCUPIED_DIFF = 22.0   # mean abs BGR difference to the empty tray (empty ~8-10, occupied >= 35 on Run 5)
SLOT_STABLE = 15       # 5 s: the grey gripper can sit over a slot for ~3 s without being flagged as arm
CAPPED_YELLOW = 0.15   # yellow fraction of the slot patch (uncapped 0.00-0.10, capped 0.26-0.77 on Run 5)


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


def slots(bgr, empty):
    """Per Good-tray slot: (difference to the empty tray, yellow fraction)."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    out = []
    for x, y in GOOD_SLOTS:
        window = np.s_[y - SLOT_HALF:y + SLOT_HALF, x - SLOT_HALF:x + SLOT_HALF]
        diff = float(np.abs(bgr[window].astype(np.int16) - empty[window].astype(np.int16)).mean())
        out.append((round(diff, 1), round(float((cv2.inRange(hsv[window], *YELLOW) > 0).mean()), 2)))
    return out


class TrayScanner:
    """One sample every STEP frames. The frame at empty_frame must show an empty Good tray."""

    def __init__(self, fps, empty_frame=0):
        self.fps, self.empty_frame, self.empty = fps, empty_frame, None

    def push(self, index, bgr):
        if index == self.empty_frame:
            self.empty = bgr.copy()
        if index % STEP or self.empty is None:
            return None
        per = caps(bgr)
        return {"frame": index, "t": index / self.fps,
                **{f"{k}_count": v[0] for k, v in per.items()},
                **{f"{k}_caps": v[1] for k, v in per.items()},
                "arm": any(v[2] for v in per.values()),
                "slots": slots(bgr, self.empty)}


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

    def push(self, s):
        key = None if s["arm"] else tuple(s[f"{t}_count"] for t in TRAYS)
        if key is None:
            self.run = []
            return []
        run = self.run = self.run + [s] if self.run and key == tuple(self.run[0][f"{t}_count"] for t in TRAYS) else [s]
        if len(run) != STABLE:
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
    """Good-tray slot arrivals, one sample at a time. Arm samples are skipped (the arm occludes
    slots); a slot is committed occupied (or vacated) after SLOT_STABLE consecutive arm-free
    samples in the new state."""

    def __init__(self):
        self.state = [False] * len(GOOD_SLOTS)
        self.runs = [[] for _ in GOOD_SLOTS]

    def push(self, s):
        if s["arm"]:
            return []
        out, state, runs = [], self.state, self.runs
        for k, (diff, yellow) in enumerate(s["slots"]):
            occupied = diff > OCCUPIED_DIFF
            runs[k] = runs[k] + [(s, yellow)] if occupied != state[k] else []
            if len(runs[k]) == SLOT_STABLE:
                first = runs[k][0][0]
                state[k] = occupied
                row, col = divmod(k, 6)
                event = {"kind": "arrival" if occupied else "vacated", "tray": "good", "t": first["t"],
                         "frame": first["frame"], "slot": f"R{row + 1}C{col + 1}", "position": GOOD_SLOTS[k]}
                if occupied:
                    event["yellow"] = float(np.median([y for _, y in runs[k]]))
                    event["capped"] = event["yellow"] >= CAPPED_YELLOW
                out.append(event)
                runs[k] = []
        return out


def slot_events(samples):
    tracker = SlotTracker()
    return [e for s in samples for e in tracker.push(s)], sum(tracker.state)


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
    parser.add_argument("--empty-frame", type=int, default=0, help="a frame where the Good tray is empty")
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
    slot_found, good_slots = slot_events(samples)
    # Good-tray arrivals come from slots (they see uncapped vials); Bad-tray arrivals from caps.
    arrivals = sorted([e for e in slot_found if e["kind"] == "arrival"] +
                      [e for e in found if e["kind"] == "arrival" and e["tray"] == "bad"], key=lambda e: e["t"])
    with (out / "tray-events.jsonl").open("w") as handle:
        for e in sorted(found + slot_found, key=lambda e: e["t"]):
            handle.write(json.dumps(e) + "\n")
    ledger = json.loads((args.run / "ledger.json").read_text())
    summary, rows = reconcile(arrivals, ledger, samples[-1]["t"] if samples else 0)
    summary["final_cap_counts"] = final
    summary["final_good_slots_occupied"] = good_slots
    summary["good_tray_cap_arrivals"] = sum(e["kind"] == "arrival" and e["tray"] == "good" for e in found)
    summary["cap_count_dips"] = sum(e["kind"] == "dip" for e in found)
    summary["slots_vacated"] = [e["slot"] for e in slot_found if e["kind"] == "vacated"]
    (out / "reconciliation.json").write_text(json.dumps({"summary": summary, "vials": rows}, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
