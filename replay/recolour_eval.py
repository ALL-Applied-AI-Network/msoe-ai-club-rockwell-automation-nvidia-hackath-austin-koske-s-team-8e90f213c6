"""Known-carryover test on real footage: recolour Run 5 liquid toward the previous recipe.

Generated video (Cosmos Transfer) changes liquid hue uncontrollably, so it cannot supply a known
carryover fraction. This recolours the real liquid instead. For each target vial (an in-spec vial
whose recipe and previous recipe both have reference colours), every model sample during its fill
is processed twice: the liquid segmenter's mask on the original frame selects the liquid pixels,
which are shifted in optical density by f * (previous reference - own reference), each channel:

    linear_rgb_new = linear_rgb * exp(-f * (D_previous - D_own)),   D = -ln(linear RGB of the reference Lab)

Then the unchanged models, measurement QA, segmentation and decision rule run on the recoloured
frame. Every vial outside the targets is untouched. The result is real glare, noise and
segmentation with a ground-truth residue fraction f. Evaluation only: nothing is fitted, and the
thresholds are the line's current ones (or --line).

    python -m replay.recolour_eval --video vial_fill_video_5.mp4 --schedule run5-schedule.csv \\
        --library recipe-library.json --ledger run5-v3/ledger.csv --assets ASSETS --out recolour.json
"""
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from skimage.color import lab2rgb

from replay.decide import DEFAULT_LINE, decide, toward_previous
from replay.line import FillTracker, load_schedule, segment


def srgb_to_linear(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(c):
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * np.power(np.clip(c, 0, None), 1 / 2.4) - 0.055)


def recolour(rgb, mask, shift):
    """Shift the masked pixels' optical density by `shift` (3 channels, linear RGB)."""
    mask = cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    out = rgb.copy()
    lin = srgb_to_linear(rgb[mask].astype(np.float64) / 255)
    out[mask] = np.clip(linear_to_srgb(np.clip(lin * np.exp(-shift), 0, 1)) * 255, 0, 255).astype(np.uint8)
    return out


def targets(ledger, schedule, library, refvials, f):
    """Windows (seconds) and density shifts for in-spec vials whose recipe and previous recipe have references."""
    refs = {str(r["id"]): r["reference"]["lab"] for r in library["recipes"] if r.get("reference")}
    out = []
    for row in ledger:
        seq, recipe = int(row["seq"]), row["recipe_id"]
        earlier = [r for r in schedule[:seq - 1] if r != recipe]
        previous = earlier[-1] if earlier else None
        if row["verdict"] != "IN_SPEC" or recipe not in refs or previous not in refs:
            continue
        shift = f * (srgb_to_linear_density(refs[previous]) - srgb_to_linear_density(refs[recipe]))
        out.append(dict(seq=seq, recipe=recipe, previous=previous, reference_vial=seq in refvials,
                        t0=float(row["fill_start_s"]) - 1, t1=float(row["decision_s"]) + 1, shift=shift))
    return out


def srgb_to_linear_density(lab):
    rgb = np.clip(lab2rgb(np.asarray(lab, float)[None, None])[0, 0], 1e-4, 1)
    return -np.log(np.clip(srgb_to_linear(rgb), 1e-4, 1))


class Recolouring:
    """Wraps the model runtime: recolours the liquid of frames inside a target window, then infers."""

    def __init__(self, runtime, windows):
        self.runtime, self.windows, self.t = runtime, windows, 0.0

    def new_measurement(self):
        return self.runtime.new_measurement()

    def render_overlay(self, *args):
        return self.runtime.render_overlay(*args)

    def infer(self, rgb, reference, qa, fallback_vial_box=None):
        window = next((w for w in self.windows if w["t0"] <= self.t <= w["t1"]), None)
        if window is not None and np.any(window["shift"]):
            mask = self.runtime.liquid.predict(rgb, reference)["component_mask"]
            if mask.any():
                rgb = recolour(rgb, mask, window["shift"])
        return self.runtime.infer(rgb, reference, qa, fallback_vial_box=fallback_vial_box)


def run_once(video, runtime, windows):
    wrapped = Recolouring(runtime, windows)
    capture = cv2.VideoCapture(str(video))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    tracker, index, dwells = FillTracker(wrapped, fps), -1, []
    while True:
        ok, bgr = capture.read()
        index += 1
        if not ok:
            break
        wrapped.t = index / fps
        dwells += [d for kind, d in tracker.push(index, bgr) if kind == "end"]
    capture.release()
    dwells += [d for _, d in tracker.close()]
    vials = []
    for dwell in dwells:
        vial = segment(dwell, fps)
        if vial:
            vial["seq"] = len(vials) + 1
            vials.append(vial)
    return vials


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True, help="ledger.csv of the unmodified run (target windows)")
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--line", type=Path, help="line settings JSON (default: the line's current thresholds)")
    parser.add_argument("--fractions", type=float, nargs="+", default=[0.0, 0.05, 0.1, 0.2, 0.3])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    from integration.runtime import ModelRuntime
    from replay.eval_freeze import reference_vials
    schedule = load_schedule(args.schedule)
    library = json.loads(args.library.read_text())
    line = {**DEFAULT_LINE, **(json.loads(args.line.read_text()) if args.line else {})}
    ledger = list(csv.DictReader(open(args.ledger)))
    refvials = set(reference_vials(library).values())
    runtime = ModelRuntime(args.assets, args.device)
    results = []
    for f in args.fractions:
        windows = targets(ledger, schedule, library, refvials, f)
        vials = run_once(args.video, runtime, windows)
        verdicts, _ = decide(vials, schedule, library, line)
        by_seq = {v.seq: v for v in verdicts}
        refs = {str(r["id"]): r["reference"]["lab"] for r in library["recipes"] if r.get("reference")}
        rows = []
        for w in windows:
            v = by_seq.get(w["seq"])
            rows.append(dict(seq=w["seq"], recipe=w["recipe"], previous=w["previous"], reference_vial=w["reference_vial"],
                             verdict=v and v.verdict, pattern=v and v.pattern,
                             de_intended=v and v.de_intended, de_previous=v and v.de_previous, direction_cos=v and v.direction_cos,
                             endpoint_lab=v and v.endpoint_lab,
                             density_cos=v and v.endpoint_lab and toward_previous(v.endpoint_lab, refs[w["recipe"]], refs[w["previous"]], "density"),
                             flush="FLUSH" in [a.kind for a in v.actions] if v else None))
        out_of_spec = sum(r["verdict"] == "OUT_OF_SPEC" for r in rows)
        carryover = sum(r["pattern"] == "carryover" for r in rows)
        results.append(dict(f=f, vials=len(vials), targets=rows, out_of_spec=out_of_spec, carryover=carryover))
        print(f"f {f}: {len(vials)} vials; targets {len(rows)}: OUT_OF_SPEC {out_of_spec}, carryover/FLUSH {carryover}; "
              + ", ".join(f"v{r['seq']} {r['verdict']} dE {r['de_intended'] and round(r['de_intended'], 1)}" for r in rows), flush=True)
    args.out.write_text(json.dumps(dict(video=str(args.video), line=line, fractions=args.fractions, results=results,
                                        note="Run 5 footage with injected carryover; evaluation only, nothing fitted."), indent=1))


if __name__ == "__main__":
    main()
