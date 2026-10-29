"""Line runner: video + recipe schedule -> per-vial verdict, action, deadline and evidence.

    python -m replay.line --video V --schedule S --library L --assets A --out DIR [--labels GT]

Stage 2 is causal over the stream: a vial dwell is a still period of the carrier (motion in a
grey thumbnail, as validated in analysis/scripts/p9_unlabeled.py); the frame shortly after the
carrier settles is the empty-vial temporal reference; the models run every STEP frames in the
dwell; when the carrier moves again the vial is finalized and decided. Times are container
seconds (frame / fps). The n-th detected filled vial is schedule row n.

`--samples` re-runs stages 2b-5 from a previous frames.jsonl without the models.
"""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from replay.decide import as_dict, decide, rule_cause

STEP = 15                 # frames between model samples (0.5 s at 30 fps)
MOTION_THR = 1.0          # mean abs thumbnail difference in the carrier area
MOTION_WIN = 5            # frames: any motion in the last WIN frames = moving
SETTLE = 8                # frames after the carrier stops before taking the reference
MIN_DWELL_S = 8.0         # shorter still periods are not vial dwells
MIN_FILL_PX = 300         # final liquid proposal pixels for a dwell to count as a fill
END_FRACTION = 0.97       # fill end: first sample reaching this fraction of the final area
START_FRACTION = 0.10     # fill start: first sample above this fraction, sustained
ENDPOINT_S = 6.0          # endpoint = median Lab of QA-valid samples in the last 6 s of the fill


def thumbnail(bgr):
    grey = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return cv2.resize(grey[:430], (32, 24), interpolation=cv2.INTER_AREA)[10:23, 11:21].astype(np.int16)


class FillTracker:
    """Causal stage 2, one frame at a time. push() returns ("start" | "sample" | "end", payload)
    events as they happen: a dwell starts when the carrier stops and ends when it moves again."""

    def __init__(self, runtime, fps):
        self.runtime, self.fps = runtime, fps
        self.prev, self.recent, self.dwell = None, [], None

    def push(self, index, bgr):
        thumb = thumbnail(bgr)
        motion = 0.0 if self.prev is None else float(np.abs(thumb - self.prev).mean())
        self.prev = thumb
        self.recent = (self.recent + [motion])[-MOTION_WIN:]
        if max(self.recent) > MOTION_THR:
            return self.close()
        out = []
        if self.dwell is None:
            self.dwell = {"start_frame": index, "end_frame": index, "reference_frame": None, "samples": []}
            self.qa = self.runtime.new_measurement() if self.runtime else None
            self.reference = None
            self.next_sample = SETTLE + STEP
            out.append(("start", self.dwell))
        dwell = self.dwell
        dwell["end_frame"] = index
        k = index - dwell["start_frame"]
        if self.runtime is None:
            return out
        # ">=" rather than "==" so a live camera that drops a frame still gets its reference
        # and samples; with every frame present this is the same schedule.
        if self.reference is None and k >= SETTLE:
            self.reference = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            dwell["reference_frame"] = index
            return out
        if self.reference is not None and k >= self.next_sample:
            self.next_sample = k + STEP - (k - SETTLE) % STEP
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            result, measured, boxes = self.runtime.infer(rgb, self.reference, self.qa)
            sample = {
                "frame": index, "t": index / self.fps,
                "presence": float(result["presence"]),
                "liquid_px": int(np.asarray(result["proposal_mask"]).sum()),
                "vial_boxes": int(len(boxes)),
                "qa_valid": bool(measured["measurement_valid"]),
                "flags": list(measured["quality_flags"]),
                "lab": measured["lab"], "rgb": measured["rgb"],
            }
            dwell["samples"].append(sample)
            out.append(("sample", sample))
        return out

    def close(self):
        """End the open dwell (carrier moved, or the stream ended)."""
        if self.dwell is None:
            return []
        dwell, self.dwell = self.dwell, None
        return [("end", dwell)]


def stream(video, runtime, fps_override=None):
    """Yield (dwell, fps) per still period. Each sample is one model evaluation."""
    capture = cv2.VideoCapture(str(video))
    fps = fps_override or capture.get(cv2.CAP_PROP_FPS) or 30.0
    tracker, index = FillTracker(runtime, fps), -1
    while True:
        ok, bgr = capture.read()
        index += 1
        if not ok:
            break
        for kind, dwell in tracker.push(index, bgr):
            if kind == "end":
                yield dwell, fps
    capture.release()
    for _, dwell in tracker.close():
        yield dwell, fps


def segment(dwell, fps):
    """Stage 2b/3: fill start/end and endpoint from one dwell's samples, or None if no fill."""
    duration = (dwell["end_frame"] - dwell["start_frame"]) / fps
    samples = dwell["samples"]
    if duration < MIN_DWELL_S or len(samples) < 4:
        return None
    area = np.array([s["liquid_px"] for s in samples], float)
    final = float(np.median(area[-6:]))
    if final < MIN_FILL_PX:
        return None
    above = area >= START_FRACTION * final
    start = next((i for i in range(len(area) - 2) if above[i:i + 3].all()), 0)
    end = next((i for i in range(start, len(area)) if area[i] >= END_FRACTION * final), len(area) - 1)
    fill_start, fill_end = samples[start]["t"], samples[end]["t"]
    window = [s for s in samples if fill_end - ENDPOINT_S <= s["t"] <= fill_end and s["qa_valid"] and s["lab"]]
    lab = np.median(np.array([s["lab"] for s in window]), axis=0).tolist() if window else None
    return {
        "arrival_s": dwell["start_frame"] / fps, "departure_s": dwell["end_frame"] / fps,
        "reference_frame": dwell["reference_frame"],
        "fill_start_s": fill_start, "fill_end_s": fill_end,
        "fill_start_frame": samples[start]["frame"], "fill_end_frame": samples[end]["frame"],
        "final_liquid_px": final, "samples": len(samples), "qa_valid_samples": sum(s["qa_valid"] for s in samples),
        "endpoint_samples": len(window), "endpoint_lab": lab,
        "mid_fill_frame": samples[(start + end) // 2]["frame"],
        "qa_flags": dict(Counter(f for s in samples[start:] for f in s.get("flags", [])).most_common()),
        # Causal decision time: the vial is decided when the carrier moves it on.
        "decision_s": dwell["end_frame"] / fps,
    }


def load_schedule(path):
    with open(path) as handle:
        rows = list(csv.DictReader(handle))
    return [str(r["recipe_id"]) for r in sorted(rows, key=lambda r: int(r["seq"]))]


def run(args):
    out = args.out
    out.mkdir(parents=True, exist_ok=False)
    schedule = load_schedule(args.schedule)
    library = json.loads(args.library.read_text())

    if args.samples:
        dwells = [json.loads(line) for line in args.samples.read_text().splitlines()]
        fps = dwells[0]["fps"] if dwells else 30.0
    else:
        from integration.runtime import ModelRuntime
        runtime = ModelRuntime(args.assets, args.device)
        dwells = []
        with (out / "frames.jsonl").open("w") as handle:
            for dwell, fps in stream(args.video, runtime):
                dwell["fps"] = fps
                handle.write(json.dumps(dwell) + "\n")
                handle.flush()
                dwells.append(dwell)
                print(f"dwell {dwell['start_frame']}-{dwell['end_frame']}: {len(dwell['samples'])} samples", flush=True)
        if args.samples is None and dwells:
            fps = dwells[0]["fps"]

    vials = []
    for dwell in dwells:
        vial = segment(dwell, dwell.get("fps", fps))
        if vial:
            vial["seq"] = len(vials) + 1
            vials.append(vial)
    for current, nxt in zip(vials, vials[1:]):
        current["next_fill_start_s"] = nxt["fill_start_s"]

    verdicts, actions = decide(vials, schedule, library, json.loads(args.line.read_text()) if args.line else None)
    write_outputs(out, vials, verdicts, actions, schedule, dwells, args)
    if args.labels:
        from replay.line_metrics import score
        metrics = score(vials, verdicts, actions, args.labels, schedule)
        (out / "metrics.json").write_text(json.dumps(metrics, indent=1))
        print(json.dumps(metrics["summary"], indent=1))


def ledger_row(vial, v):
    kinds = [a.kind for a in v.actions]
    return {
        "seq": vial["seq"], "recipe_id": v.recipe_id, "previous_recipe_id": v.previous_recipe_id,
        "arrival_s": round(vial["arrival_s"], 2), "fill_start_s": round(vial["fill_start_s"], 2),
        "fill_end_s": round(vial["fill_end_s"], 2), "decision_s": round(vial["decision_s"], 2),
        "endpoint_samples": vial["endpoint_samples"],
        "L": v.endpoint_lab and round(v.endpoint_lab[0], 2), "a": v.endpoint_lab and round(v.endpoint_lab[1], 2),
        "b": v.endpoint_lab and round(v.endpoint_lab[2], 2),
        "de_intended": v.de_intended and round(v.de_intended, 2), "de_previous": v.de_previous and round(v.de_previous, 2),
        "direction_cos": v.direction_cos and round(v.direction_cos, 2),
        "verdict": v.verdict, "pattern": v.pattern, "actions": "+".join(kinds),
        "scrap_deadline_s": round(vial["fill_end_s"] + 120, 2) if "SCRAP" in kinds else None,
        "rule_cause": rule_cause(v),
    }


def write_outputs(out, vials, verdicts, actions, schedule, dwells, args):
    rows = [ledger_row(vial, v) for vial, v in zip(vials, verdicts)]
    with (out / "ledger.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["seq"])
        writer.writeheader()
        writer.writerows(rows)
    (out / "ledger.json").write_text(json.dumps({
        "schema": "linewatch-ledger/v1", "video": str(args.video), "schedule": str(args.schedule),
        "library": str(args.library), "scheduled_vials": len(schedule), "detected_vials": len(vials),
        "dwells": len(dwells), "vials": [{**vial, **as_dict(v)} for vial, v in zip(vials, verdicts)],
    }, indent=1, default=str))
    with (out / "actions.jsonl").open("w") as handle:
        for action in sorted(actions, key=lambda a: a.issued_s):
            handle.write(json.dumps(as_dict(action)) + "\n")
    rca = out / "rca"
    rca.mkdir()
    for vial, v in zip(vials, verdicts):
        if v.verdict == "IN_SPEC":
            continue
        neighbours = [{"seq": o["seq"], "recipe_id": ov.recipe_id, "verdict": ov.verdict, "de_intended": ov.de_intended}
                      for o, ov in zip(vials, verdicts) if abs(o["seq"] - vial["seq"]) <= 2 and o is not vial]
        (rca / f"vial-{vial['seq']:02d}.json").write_text(json.dumps({
            "schema": "linewatch-evidence-packet/v1",
            "context": {"seq": vial["seq"], "intended_recipe": v.recipe_id, "previous_recipe": v.previous_recipe_id,
                        "next_recipe": schedule[vial["seq"]] if vial["seq"] < len(schedule) else None},
            "measurement": {k: vial[k] for k in ("endpoint_lab", "endpoint_samples", "qa_valid_samples", "samples",
                                                 "fill_start_s", "fill_end_s", "final_liquid_px", "qa_flags")},
            "comparison": {"de_intended": v.de_intended, "de_previous": v.de_previous, "direction_cos": v.direction_cos,
                           "verdict": v.verdict, "pattern": v.pattern},
            "neighbours": neighbours,
            "keyframes": {"reference_frame": vial["reference_frame"], "fill_start_frame": vial["fill_start_frame"],
                          "mid_fill_frame": vial["mid_fill_frame"], "fill_end_frame": vial["fill_end_frame"]},
            "actions": [as_dict(a) for a in v.actions],
            "rule_baseline": {"cause": rule_cause(v)},
            "explanation": None,
        }, indent=1))
    print(f"{len(vials)} vials from {len(dwells)} dwells; schedule has {len(schedule)}. Ledger: {out / 'ledger.csv'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--schedule", type=Path, required=True, help="CSV seq,recipe_id")
    parser.add_argument("--library", type=Path, required=True, help="recipe-library/v1 JSON")
    parser.add_argument("--assets", type=Path, help="released temporal-cnn assets (not needed with --samples)")
    parser.add_argument("--line", type=Path, help="JSON overrides of replay.decide.DEFAULT_LINE")
    parser.add_argument("--labels", type=Path, help="ground truth CSV (vial,recipe_id,start_frame,end_frame) for metrics")
    parser.add_argument("--samples", type=Path, help="frames.jsonl from an earlier run: skip the models")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if not args.samples and not args.assets:
        parser.error("--assets is required unless --samples is given")
    run(args)


if __name__ == "__main__":
    main()
