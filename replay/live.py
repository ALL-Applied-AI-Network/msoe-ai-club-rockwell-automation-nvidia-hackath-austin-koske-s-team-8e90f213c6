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
from replay.line import FillTracker, ledger_row, load_schedule, segment
from replay.tray import EXPECTED_TRAY, GOOD_SLOTS, TRANSIT_WINDOW, TRAYS, CapTracker, SlotTracker, TrayScanner

ROI = (256, 240, 368, 432)          # x0, y0, x1, y1: the fill-station vial region


class LiveLine:
    def __init__(self, schedule, library, line=None, runtime=None, fps=30.0):
        self.schedule, self.library = list(schedule), library
        self.line = {**DEFAULT_LINE, **(line or {})}
        self.fps = fps
        self.fill = FillTracker(runtime, fps)
        self.scanner = None
        self.caps, self.slots = CapTracker(), SlotTracker()
        self.vials, self.flushed, self.waiting = [], set(), []

    def meta(self):
        references = {str(r["id"]): r.get("reference") and r["reference"]["lab"] for r in self.library["recipes"]}
        return {"type": "meta", "t": 0.0, "schedule": self.schedule, "scheduled_vials": len(self.schedule),
                "references": references, "line": self.line, "roi": ROI, "good_slots": GOOD_SLOTS,
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
        return self._fill(self.fill.push(index, bgr))

    def fill_item(self, kind, payload):
        """A stored stage-2 event: ("start", dwell), ("sample", sample) or ("end", full dwell)."""
        return self._fill([(kind, payload)])

    def finish(self):
        return self._fill(self.fill.close())

    def _fill(self, items):
        out = []
        for kind, payload in items:
            if kind == "start":
                t = payload["start_frame"] / self.fps
                out.append({"type": "dwell_start", "t": t, "frame": payload["start_frame"]})
                seq = len(self.vials) + 1
                flush = seq not in self.flushed and proactive_flush(seq, self.schedule, self.library, self.line, t, None)
                if flush:
                    self.flushed.add(seq)
                    out.append({"type": "action", "t": t, **as_dict(flush)})
            elif kind == "sample":
                out.append({"type": "sample", "t": payload["t"], "frame": payload["frame"],
                            "liquid_px": payload["liquid_px"], "qa_valid": payload["qa_valid"],
                            "lab": payload["lab"], "presence": payload["presence"]})
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
        decision = next((k for k in ("SCRAP", "HOLD", "PASS") if k in [a.kind for a in verdict.actions]), None)
        self.waiting.append({"seq": vial["seq"], "departure_s": vial["departure_s"], "decision": decision,
                             "verdict": verdict.verdict, "recipe_id": verdict.recipe_id})
        return out

    # Tray camera.
    def tray_frame(self, index, bgr):
        if self.scanner is None:
            self.scanner = TrayScanner(self.fps, empty_frame=index)
        sample = self.scanner.push(index, bgr)
        return self.tray_sample(sample) if sample else []

    def tray_sample(self, s):
        found = [e for e in self.slots.push(s) if e["kind"] == "arrival"]
        found += [e for e in self.caps.push(s) if e["kind"] == "arrival" and e["tray"] == "bad"]
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
