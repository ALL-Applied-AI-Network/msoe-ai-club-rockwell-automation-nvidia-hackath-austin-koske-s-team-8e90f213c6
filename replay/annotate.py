"""Annotated demo video from a line run: original frames plus an operator panel.

    python -m replay.annotate --video V --run DIR --library L --out annotated.mp4 [--start S --end S]
        [--tray-video cap_seal.mp4 --tray DIR/tray]   # adds the finished-vial tray camera

Reads DIR/frames.jsonl, ledger.json and actions.jsonl from `python -m replay.line`; no models run.
Verdicts, actions and banners appear at their causal issue time. Fill-phase labels and the fill
bar use the finalized segmentation of the dwell, so they are a visual aid, not a live signal.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from replay.measurement import delta_e

ROI = (256, 240, 368, 432)          # x0, y0, x1, y1: the stage-1 vial region
PANEL = 400
BANNER_S = 4.0
WHITE, GREY, DIM = (240, 240, 240), (150, 150, 150), (90, 90, 90)
KIND_COLOUR = {"PASS": (90, 200, 90), "SCRAP": (60, 60, 230), "FLUSH": (230, 160, 40),
               "FLUSH_RECOMMENDED": (200, 140, 60), "STOP_CHECK_DOSING": (40, 40, 200), "HOLD": (40, 190, 230)}
VERDICT_COLOUR = {"IN_SPEC": (90, 200, 90), "OUT_OF_SPEC": (60, 60, 230), "CANT_VERIFY": (40, 190, 230)}


def ascii_text(text):
    for a, b in (("ΔE", "dE"), ("→", "->"), ("≤", "<="), ("—", "-")):
        text = text.replace(a, b)
    return text.encode("ascii", "replace").decode()


def put(img, text, xy, scale=0.5, colour=WHITE, thick=1):
    cv2.putText(img, ascii_text(text), xy, cv2.FONT_HERSHEY_SIMPLEX, scale, colour, thick, cv2.LINE_AA)


def lab_to_bgr(lab):
    pixel = np.array([[lab]], np.float32)
    rgb = cv2.cvtColor(pixel, cv2.COLOR_Lab2RGB)[0, 0]
    return tuple(int(c) for c in np.clip(rgb[::-1] * 255, 0, 255))


def wrap(text, width=46):
    words, lines, line = ascii_text(text).split(), [], ""
    for word in words:
        if len(line) + len(word) + 1 > width:
            lines.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    return lines + ([line] if line else [])


class Timeline:
    """Per-frame lookups over the saved run."""

    def __init__(self, run, library):
        self.dwells = [json.loads(line) for line in (run / "frames.jsonl").read_text().splitlines()]
        ledger = json.loads((run / "ledger.json").read_text())
        self.vials = ledger["vials"]
        self.scheduled = ledger["scheduled_vials"]
        self.actions = [json.loads(line) for line in (run / "actions.jsonl").read_text().splitlines()]
        self.fps = self.dwells[0].get("fps", 30.0) if self.dwells else 30.0
        self.by_start = {round(v["arrival_s"] * self.fps): v for v in self.vials}
        self.references = {str(r["id"]): r.get("reference") and r["reference"]["lab"] for r in library["recipes"]}

    def dwell_at(self, frame):
        for dwell in self.dwells:
            if dwell["start_frame"] <= frame <= dwell["end_frame"]:
                return dwell
        return None

    def vial_for(self, dwell):
        return dwell and self.by_start.get(dwell["start_frame"])


def draw_panel(panel, frame, t, timeline):
    panel[:] = (28, 28, 28)
    put(panel, "LINEWATCH", (16, 30), 0.75, WHITE, 2)
    put(panel, f"t = {t:7.1f} s", (250, 30), 0.5, GREY)
    dwell = timeline.dwell_at(frame)
    vial = timeline.vial_for(dwell)
    y = 62

    if dwell is None:
        put(panel, "Carrier moving", (16, y), 0.6, GREY)
        last = [v for v in timeline.vials if v["decision_s"] <= t]
        if last:
            v = last[-1]
            put(panel, f"last: vial {v['seq']} recipe {v['recipe_id']}", (16, y + 30), 0.5, GREY)
            put(panel, v["verdict"].replace("_", " "), (16, y + 58), 0.7, VERDICT_COLOUR[v["verdict"]], 2)
            if v["de_intended"] is not None:
                put(panel, f"dE {v['de_intended']:.1f}  {v['pattern'].replace('_', ' ')}", (220, y + 58), 0.5, GREY)
    elif vial is None:
        put(panel, "Station still - no fill detected", (16, y), 0.55, GREY)
    else:
        decided = t >= vial["decision_s"]
        put(panel, f"Vial {vial['seq']}/{timeline.scheduled}", (16, y), 0.7, WHITE, 2)
        put(panel, f"recipe {vial['recipe_id']}", (170, y), 0.6, WHITE)
        y += 22
        put(panel, f"previous recipe {vial['previous_recipe_id'] or '-'}", (16, y), 0.45, GREY)
        y += 24
        samples = [s for s in dwell["samples"] if s["frame"] <= frame]
        if t < vial["fill_start_s"]:
            phase = "Waiting for fill"
        elif t < vial["fill_end_s"]:
            phase = "Filling"
        elif not decided:
            phase = "Fill complete - measuring"
        else:
            phase = "Decided"
        put(panel, phase, (16, y), 0.55, (230, 200, 120))
        y += 12
        fill = min(1.0, samples[-1]["liquid_px"] / max(vial["final_liquid_px"], 1)) if samples else 0.0
        cv2.rectangle(panel, (16, y), (PANEL - 16, y + 10), DIM, 1)
        cv2.rectangle(panel, (16, y), (16 + int((PANEL - 32) * fill), y + 10), (230, 200, 120), -1)
        y += 30

        # Live colour vs the recipe reference.
        valid = [s for s in samples if s["qa_valid"] and s["lab"]]
        ref = timeline.references.get(str(vial["recipe_id"]))
        put(panel, "measured", (16, y), 0.45, GREY)
        put(panel, "reference", (150, y), 0.45, GREY)
        y += 6
        if valid:
            cv2.rectangle(panel, (16, y), (116, y + 50), lab_to_bgr(valid[-1]["lab"]), -1)
        else:
            put(panel, "no QA-valid", (20, y + 30), 0.4, DIM)
        if ref:
            cv2.rectangle(panel, (150, y), (250, y + 50), lab_to_bgr(ref), -1)
        else:
            put(panel, "none", (170, y + 30), 0.4, DIM)
        if valid and ref:
            put(panel, f"dE {delta_e(valid[-1]['lab'], ref):.1f}", (268, y + 32), 0.7, WHITE, 2)
        y += 72

        if decided:
            colour = VERDICT_COLOUR[vial["verdict"]]
            put(panel, vial["verdict"].replace("_", " "), (16, y), 0.75, colour, 2)
            if vial["pattern"] not in ("nominal",):
                put(panel, vial["pattern"].replace("_", " "), (220, y), 0.5, colour)
            y += 22
            if vial["de_intended"] is not None:
                detail = f"endpoint dE {vial['de_intended']:.1f} to recipe"
                if vial["direction_cos"] is not None:
                    detail += f", cos {vial['direction_cos']:.2f} to previous"
                put(panel, detail, (16, y), 0.42, GREY)
        y += 10

    # Action log: newest first, with live deadline countdowns.
    y = max(y, 300) + 14
    put(panel, "Actions", (16, y), 0.55, WHITE)
    y += 8
    issued = [a for a in timeline.actions if a["issued_s"] <= t][::-1][:5]
    for a in issued:
        y += 20
        colour = KIND_COLOUR.get(a["kind"], WHITE)
        put(panel, f"{a['issued_s']:6.0f}s", (16, y), 0.42, GREY)
        put(panel, a["kind"], (78, y), 0.45, colour, 1)
        put(panel, f"vial {a['seq']}" if a["seq"] else "line", (230, y), 0.42, GREY)
        if a["deadline_s"] is not None and a["deadline_s"] - t >= 1 and a["kind"] != "PASS":
            put(panel, f"due {int(a['deadline_s'] - t):3d}s", (300, y), 0.42, (120, 200, 255))

    # Vial strip: one cell per scheduled vial, coloured once decided.
    n = max(timeline.scheduled, 1)
    cell = (PANEL - 32) / n
    top = panel.shape[0] - 34
    put(panel, "Run", (16, top - 6), 0.42, GREY)
    decided_seq = {v["seq"]: v for v in timeline.vials if v["decision_s"] <= t}
    current = vial["seq"] if vial else None
    for i in range(n):
        x0, x1 = int(16 + i * cell) + 1, int(16 + (i + 1) * cell) - 1
        v = decided_seq.get(i + 1)
        cv2.rectangle(panel, (x0, top), (x1, top + 18), VERDICT_COLOUR[v["verdict"]] if v else DIM, -1)
        if current == i + 1:
            cv2.rectangle(panel, (x0 - 1, top - 2), (x1 + 1, top + 20), WHITE, 1)


def draw_video(bgr, frame, t, timeline):
    dwell = timeline.dwell_at(frame)
    vial = timeline.vial_for(dwell)
    colour = (230, 200, 120)
    if vial and t >= vial["decision_s"]:
        colour = VERDICT_COLOUR[vial["verdict"]]
    if dwell is not None:
        cv2.rectangle(bgr, ROI[:2], ROI[2:], colour, 2)
    recent = [a for a in timeline.actions if 0 <= t - a["issued_s"] < BANNER_S and a["kind"] != "PASS"]
    if recent:
        seq = recent[-1]["seq"]
        kinds = " + ".join(dict.fromkeys(a["kind"].replace("_", " ") for a in recent if a["seq"] == seq))
        lines = wrap(recent[-1]["reason"], 78)[:2]
        cv2.rectangle(bgr, (0, 0), (bgr.shape[1], 40 + 18 * len(lines)), (20, 20, 20), -1)
        put(bgr, f"VIAL {seq}: {kinds}", (14, 28), 0.75, KIND_COLOUR.get(recent[-1]["kind"], WHITE), 2)
        for i, line in enumerate(lines):
            put(bgr, line, (14, 52 + 18 * i), 0.45, WHITE)


NO_CAP = (255, 0, 255)
STATUS_TEXT = {"escaped": "we said {decision}: would divert", "routed_as_decided": "as decided",
               "good_vial_in_bad_tray": "good vial in Bad tray"}


def draw_tray(bgr, t, rows):
    """Finished-vial tray camera: slots coloured by our verdict once the vial lands."""
    from replay.tray import GOOD_SLOTS
    slot_xy = {f"R{i // 6 + 1}C{i % 6 + 1}": xy for i, xy in enumerate(GOOD_SLOTS)}
    landed = [r for r in rows if r["tray_arrival_s"] is not None and r["tray_arrival_s"] <= t]
    for r in landed:
        if r.get("slot") not in slot_xy:
            continue
        x, y = slot_xy[r["slot"]]
        colour = NO_CAP if r.get("capped") is False else VERDICT_COLOUR.get(r["verdict"], WHITE)
        cv2.rectangle(bgr, (x - 14, y - 14), (x + 14, y + 14), colour, 2)
        put(bgr, str(r["seq"]), (x - 9, y + 5), 0.42, (0, 0, 0), 3)
        put(bgr, str(r["seq"]), (x - 9, y + 5), 0.42, WHITE, 1)
        if r.get("capped") is False:
            put(bgr, "NO CAP", (x - 22, y + 27), 0.4, NO_CAP, 1)
    transit = [r for r in rows if r["left_fill_s"] <= t and (r["tray_arrival_s"] is None or r["tray_arrival_s"] > t)]
    escaped = [r for r in landed if r["status"] == "escaped"]
    cv2.rectangle(bgr, (0, 0), (bgr.shape[1], 50), (20, 20, 20), -1)
    put(bgr, "FINISHED TRAY", (12, 20), 0.55, WHITE, 2)
    put(bgr, f"in tray {len(landed)}   in transit {len(transit)}   left fill {len(landed) + len(transit)}",
        (190, 20), 0.45, WHITE)
    put(bgr, f"should have been diverted: {len(escaped)}", (12, 42), 0.45, KIND_COLOUR["SCRAP"])
    caps = [r for r in landed if r.get("capped") is False]
    if caps:
        put(bgr, f"missing/crooked cap: {', '.join(str(r['seq']) for r in caps)}", (330, 42), 0.45, NO_CAP)
    recent = [r for r in landed if t - r["tray_arrival_s"] < BANNER_S]
    if recent:
        r = recent[-1]
        text = f"Vial {r['seq']} -> {r['tray']} tray {r.get('slot') or ''} after {r['transit_s']:.0f}s"
        note = "NO CAP" if r.get("capped") is False else STATUS_TEXT.get(r["status"], "").format(decision=r["decision"])
        cv2.rectangle(bgr, (0, bgr.shape[0] - 52), (bgr.shape[1], bgr.shape[0] - 22), (20, 20, 20), -1)
        colour = NO_CAP if r.get("capped") is False else KIND_COLOUR["SCRAP"] if r["status"] == "escaped" else WHITE
        put(bgr, f"{text}  {note}", (12, bgr.shape[0] - 31), 0.5, colour, 1)


def render(args):
    library = json.loads(args.library.read_text())
    timeline = Timeline(args.run, library)
    capture = cv2.VideoCapture(str(args.video))
    fps = capture.get(cv2.CAP_PROP_FPS) or timeline.fps
    width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    first = int(args.start * fps)
    last = int(args.end * fps) if args.end else None
    capture.set(cv2.CAP_PROP_POS_FRAMES, first)
    tray_capture, rows = None, []
    if args.tray_video:
        tray_capture = cv2.VideoCapture(str(args.tray_video))
        tray_capture.set(cv2.CAP_PROP_POS_FRAMES, first)
        rows = json.loads((args.tray / "reconciliation.json").read_text())["vials"]
    size = (width + PANEL + (width if tray_capture else 0), height)
    writer = cv2.VideoWriter(str(args.out), cv2.VideoWriter_fourcc(*"mp4v"), fps * args.speed, size)
    panel = np.zeros((height, PANEL, 3), np.uint8)
    frame = first
    while last is None or frame <= last:
        ok, bgr = capture.read()
        if not ok:
            break
        t = frame / fps
        draw_video(bgr, frame, t, timeline)
        draw_panel(panel, frame, t, timeline)
        parts = [bgr, panel]
        if tray_capture:
            ok, tray = tray_capture.read()
            tray = tray if ok else np.zeros_like(bgr)
            draw_tray(tray, t, rows)
            parts.append(tray)
        writer.write(np.hstack(parts))
        frame += 1
        if frame % 3000 == 0:
            print(f"frame {frame}", flush=True)
    writer.release()
    capture.release()
    print(f"wrote {args.out}: frames {first}-{frame - 1}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True, help="output directory of python -m replay.line")
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--start", type=float, default=0.0, help="seconds")
    parser.add_argument("--end", type=float, help="seconds")
    parser.add_argument("--speed", type=float, default=1.0, help="playback speed-up written into the file")
    parser.add_argument("--tray-video", type=Path, help="cap_seal video on the same clock")
    parser.add_argument("--tray", type=Path, help="output directory of python -m replay.tray")
    args = parser.parse_args()
    if bool(args.tray_video) != bool(args.tray):
        parser.error("--tray-video and --tray go together")
    render(args)


if __name__ == "__main__":
    main()
