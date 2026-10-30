"""Live line runner: the line runner's stages, causal and event by event.

    python -m replay.live --samples DIR/frames.jsonl --tray-samples TRAY/tray-samples.jsonl \
        --schedule S --library L --out events.jsonl

Feed fill-camera frames (the models run) or stored per-sample model outputs, and tray-camera
frames or stored tray samples, in time order. Each call returns the events that became known at
that moment, so a UI can show them as they happen. Every event has "t": the video time at which
it is known.

Decisions use the same segment() and decide() as `python -m replay.line`. Three differences
follow from not knowing the future:
- a FLUSH has no deadline (the next fill has not started);
- FLUSH_RECOMMENDED is issued when the carrier stops, before we know the dwell is a fill;
- tray arrivals are matched first-in-first-out within the transit window, not by the
  after-the-fact alignment in replay.tray.
"""
import argparse
import json
from pathlib import Path

import cv2

from replay.decide import DEFAULT_LINE, as_dict, decide, proactive_flush
from replay.early import EarlyWarning, trajectory
from replay.line import FillTracker, ledger_row, load_schedule, segment
from replay.tray import (EXPECTED_TRAY, TRANSIT_WINDOW, TRAYS, CapTracker, SlotTracker,
                         TrayScanner, render_overlay, validate_tray_slots)

ROI = (256, 240, 368, 432)          # x0, y0, x1, y1: the fill-station vial region


class LiveLine:
    def __init__(self, schedule, library, line=None, runtime=None, fps=30.0, fallback_vial_box=None,
                 tray_slots=None, tray_clear_thumbnail=None, tray_empty_slot_gray=None):
        self.schedule, self.library = list(schedule), library
        self.line = {**DEFAULT_LINE, **(line or {})}
        self.fps = fps
        self.fill = FillTracker(runtime, fps, fallback_vial_box=fallback_vial_box)
        self.tray_slots = validate_tray_slots(tray_slots)
        self.tray_clear_thumbnail = tray_clear_thumbnail
        self.tray_empty_slot_gray = tray_empty_slot_gray
        self.scanner = None
        self.caps = CapTracker()
        self.slots = {tray: SlotTracker(tray, centers) for tray, centers in self.tray_slots.items()}
        self.tray_preview_bgr = None
        self.vials, self.flushed, self.waiting = [], set(), []
        # Operator CREATE mode: the recipe id whose reference the next decided vial sets.
        self.capture = None
        self.early = EarlyWarning(self.line["band_de2000"])
        # What the engine saw from frames, so a live session can be saved and reviewed later.
        self.dwells, self.tray_samples = [], []

    def meta(self):
        references = {str(r["id"]): r.get("reference") and r["reference"]["lab"] for r in self.library["recipes"]}
        return {"type": "meta", "t": 0.0, "schedule": self.schedule, "scheduled_vials": len(self.schedule),
                "references": references, "line": self.line, "roi": ROI,
                "good_slots": self.tray_slots["good"], "tray_slots": self.tray_slots,
                "trays": TRAYS, "transit_window_s": TRANSIT_WINDOW}

    def set_schedule(self, schedule, t):
        """Replace the recipe order (e.g. the operator added vials); returns a schedule event if it changed."""
        schedule = [str(r) for r in schedule]
        if schedule == self.schedule:
            return []
        self.schedule[:] = schedule
        return [{"type": "schedule", "t": t, "schedule": list(schedule), "scheduled_vials": len(schedule)}]

    # Fill camera.
    def fill_frame(self, index, bgr):
        items = self.fill.push(index, bgr)
        self.dwells.extend({**d, "fps": self.fps} for kind, d in items if kind == "end")
        return self._fill(items)

    def fill_item(self, kind, payload):
        """A stored stage-2 event: ("start", dwell), ("sample", sample) or ("end", full dwell)."""
        return self._fill([(kind, payload)])

    def finish(self):
        items = self.fill.close()
        self.dwells.extend({**d, "fps": self.fps} for kind, d in items if kind == "end")
        return self._fill(items)

    def _fill(self, items):
        out = []
        for kind, payload in items:
            if kind == "start":
                t = payload["start_frame"] / self.fps
                out.append({"type": "dwell_start", "t": t, "frame": payload["start_frame"]})
                self.early.reset()
                seq = len(self.vials) + 1
                flush = seq not in self.flushed and proactive_flush(seq, self.schedule, self.library, self.line, t, None)
                if flush:
                    self.flushed.add(seq)
                    out.append({"type": "action", "t": t, **as_dict(flush)})
            elif kind == "sample":
                out.append({"type": "sample", "t": payload["t"], "frame": payload["frame"],
                            "liquid_px": payload["liquid_px"], "qa_valid": payload["qa_valid"],
                            "lab": payload["lab"], "presence": payload["presence"],
                            "vial_boxes": payload.get("vial_boxes", 0),
                            "yolo_vial_boxes": payload.get("yolo_vial_boxes", payload.get("vial_boxes", 0)),
                            "vial_geometry_source": payload.get("vial_geometry_source", "yolo"),
                            "flags": payload.get("flags", [])})
                seq = len(self.vials) + 1
                warning = self.early.sample(payload, seq, self._recipe(seq))
                if warning:
                    out.append({"type": "action", "t": payload["t"], **as_dict(warning)})
            else:
                out.extend(self._dwell_end(payload))
        return out

    def _dwell_end(self, dwell):
        t = dwell["end_frame"] / self.fps
        out = [{"type": "dwell_end", "t": t, "frame": dwell["end_frame"]}]
        vial = segment(dwell, self.fps)
        if not vial:
            return out
        vial["seq"] = len(self.vials) + 1
        self.vials.append(vial)
        verdict = decide(self.vials, self.schedule, self.library, self.line)[0][-1]
        row = ledger_row(vial, verdict)
        out.append({"type": "vial", "t": t, "vial": {
            **row, "departure_s": round(vial["departure_s"], 2), "endpoint_lab": verdict.endpoint_lab,
            "final_liquid_px": vial["final_liquid_px"], "reference_frame": vial["reference_frame"]}})
        out.extend({"type": "action", "t": t, **as_dict(a)} for a in verdict.actions)
        if self.capture:
            out.append(self._capture(vial["seq"], verdict.endpoint_lab, t,
                                     trajectory(dwell["samples"], vial["fill_end_s"])))
        decision = next((k for k in ("SCRAP", "HOLD", "PASS") if k in [a.kind for a in verdict.actions]), None)
        self.waiting.append({"seq": vial["seq"], "departure_s": vial["departure_s"], "decision": decision,
                             "verdict": verdict.verdict, "recipe_id": verdict.recipe_id})
        return out

    def _recipe(self, seq):
        recipe_id = self.schedule[seq - 1] if 0 < seq <= len(self.schedule) else None
        return next((r for r in self.library["recipes"] if str(r["id"]) == str(recipe_id)), None)

    def _capture(self, seq, endpoint_lab, t, fill_trajectory=None):
        """CREATE mode: this vial's measured endpoint becomes the armed recipe's reference, used
        for every later vial. Without a QA-valid endpoint the mode stays armed for the next vial."""
        recipe_id = self.capture
        if endpoint_lab is None:
            return {"type": "reference_failed", "t": t, "recipe_id": recipe_id, "seq": seq,
                    "reason": "no QA-valid endpoint colour; the next vial will be used"}
        self.capture = None
        lab = [round(float(v), 3) for v in endpoint_lab]
        source = f"vial {seq} endpoint, operator CREATE mode"
        self.set_reference(recipe_id, lab, source, fill_trajectory)
        return {"type": "reference", "t": t, "recipe_id": recipe_id, "seq": seq, "lab": lab, "from": source,
                "trajectory": fill_trajectory or []}

    def set_reference(self, recipe_id, lab, source, fill_trajectory=None):
        reference = {"lab": lab, "from": source, **({"trajectory": fill_trajectory} if fill_trajectory else {})}
        for recipe in self.library["recipes"]:
            if str(recipe["id"]) == str(recipe_id):
                recipe["reference"] = reference
                return
        self.library["recipes"].append({"id": str(recipe_id), "name": str(recipe_id), "reference": reference})

    # Tray camera.
    def tray_frame(self, index, bgr):
        if self.scanner is None:
            self.scanner = TrayScanner(self.fps, empty_frame=index, tray_slots=self.tray_slots,
                                       clear_thumbnail=self.tray_clear_thumbnail,
                                       empty_slot_gray=self.tray_empty_slot_gray)
        sample = self.scanner.push(index, bgr)
        if sample:
            self.tray_samples.append(sample)
        events = self.tray_sample(sample) if sample else []
        self.tray_preview_bgr = render_overlay(
            bgr, self.slots, baseline_ready=self.scanner.empty is not None, sample=sample)
        return events

    def tray_sample(self, s):
        cap_events = self.caps.push(s)
        found = [e for e in self.slots["good"].push(s) if e["kind"] == "arrival"]
        if "bad_slots" in s:
            found += [e for e in self.slots["bad"].push(s) if e["kind"] == "arrival"]
        else:
            # Compatibility with historical samples recorded before the Bad grid existed.
            found += [e for e in cap_events if e["kind"] == "arrival" and e["tray"] == "bad"]
        out = []
        for arrival in sorted(found, key=lambda e: e["t"]):
            out.append(self._arrival(arrival, s["t"]))
        while self.waiting and s["t"] - self.waiting[0]["departure_s"] > TRANSIT_WINDOW[1]:
            vial = self.waiting.pop(0)
            out.append({"type": "tray_missed", "t": s["t"], "seq": vial["seq"], "decision": vial["decision"]})
        return out

    def _arrival(self, arrival, now):
        event = {"type": "tray_arrival", "t": now, "arrival_s": arrival["t"], "tray": arrival["tray"],
                 "slot": arrival.get("slot"), "capped": arrival.get("capped"), "position": arrival.get("position"),
                 "seq": None, "status": "unmatched_arrival"}
        for i, vial in enumerate(self.waiting):
            transit = arrival["t"] - vial["departure_s"]
            if TRANSIT_WINDOW[0] < transit < TRANSIT_WINDOW[1]:
                skipped, self.waiting = self.waiting[:i], self.waiting[i + 1:]
                expected = EXPECTED_TRAY.get(vial["decision"])
                status = "routed_as_decided" if expected == arrival["tray"] else \
                    "escaped" if vial["decision"] in ("SCRAP", "HOLD") else "good_vial_in_bad_tray"
                event.update(seq=vial["seq"], decision=vial["decision"], verdict=vial["verdict"],
                             recipe_id=vial["recipe_id"], transit_s=round(transit, 1), status=status,
                             skipped=[v["seq"] for v in skipped])
                break
        return event


def recorded_items(dwells, tray_samples):
    """Stored model and tray samples as (t, source, item), in the order they became available."""
    items = []
    for dwell in dwells:
        fps = dwell.get("fps", 30.0)
        items.append((dwell["start_frame"] / fps, 0, "fill", ("start", dwell)))
        items.extend((s["t"], 1, "fill", ("sample", s)) for s in dwell["samples"])
        items.append((dwell["end_frame"] / fps, 2, "fill", ("end", dwell)))
    items.extend((s["t"], 1, "tray", s) for s in tray_samples)
    return [(t, source, item) for t, _, source, item in sorted(items, key=lambda x: (x[0], x[1]))]


def feed(live, source, item):
    if source == "fill":
        return live.fill_item(*item)
    return live.tray_sample(item)


def video_frames(fill_video, tray_video=None, start_s=0.0):
    """Fill and tray frames in lockstep from recordings on the same clock: (t, source, (index, bgr))."""
    fill = cv2.VideoCapture(str(fill_video))
    tray = cv2.VideoCapture(str(tray_video)) if tray_video else None
    fps = fill.get(cv2.CAP_PROP_FPS) or 30.0
    index = int(start_s * fps)
    for capture in (fill, tray):
        if capture is not None and index:
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
    try:
        while True:
            ok, bgr = fill.read()
            if not ok:
                break
            yield index / fps, "fill", (index, bgr)
            if tray is not None:
                ok, tbgr = tray.read()
                if ok:
                    yield index / fps, "tray", (index, tbgr)
            index += 1
    finally:
        fill.release()
        if tray is not None:
            tray.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--samples", type=Path, required=True, help="frames.jsonl from python -m replay.line")
    parser.add_argument("--tray-samples", type=Path, help="tray-samples.jsonl from python -m replay.tray")
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="events.jsonl")
    args = parser.parse_args()
    dwells = [json.loads(line) for line in args.samples.read_text().splitlines()]
    tray = [json.loads(line) for line in args.tray_samples.read_text().splitlines()] if args.tray_samples else []
    live = LiveLine(load_schedule(args.schedule), json.loads(args.library.read_text()),
                    fps=dwells[0].get("fps", 30.0) if dwells else 30.0)
    with args.out.open("w") as handle:
        handle.write(json.dumps(live.meta()) + "\n")
        for _, source, item in recorded_items(dwells, tray):
            for event in feed(live, source, item):
                handle.write(json.dumps(event, default=str) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
