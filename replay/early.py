"""Early warning: call an off-recipe vial while it is still filling.

A recipe's reference vial gives a colour trajectory, not only an endpoint: the QA-valid Lab
samples by seconds since its fill started (tanks dispense in sequence, so a vial changes colour
as it fills). During a fill, each QA-valid sample is compared with the reference sample at the
same time since fill start; CONSECUTIVE samples in a row more than the line band (ΔE2000) away
raise one EARLY_WARNING for that vial. Only during the reference's fill duration: after it, the
endpoint decision takes over. Fill start is causal: the first of three samples in a row
with at least MIN_FILL_PX liquid pixels, the same rule for the reference and the live vial.

Settings are fixed, not tuned: the line band, 3 samples (1.5 s), MIN_FILL_PX.
Trajectories come from CREATE mode (stored with the reference) or, for the shipped Run 5
library, from reference-trajectories.json built from its reference vials:

    python -m replay.early FRAMES.jsonl LEDGER.csv LIBRARY.json --out reference-trajectories.json
"""
import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np

from replay.decide import Action
from replay.line import MIN_FILL_PX
from replay.measurement import delta_e

CONSECUTIVE = 3


def fill_start(samples):
    """Time of the first of three samples in a row with a fill-sized liquid area, or None."""
    for i in range(len(samples) - 2):
        if all(s["liquid_px"] >= MIN_FILL_PX for s in samples[i:i + 3]):
            return samples[i]["t"]
    return None


def trajectory(samples, until_s=None):
    """[[seconds since fill start, L, a, b], ...] of QA-valid samples, up to until_s."""
    start = fill_start(samples)
    if start is None:
        return []
    return [[round(s["t"] - start, 2), *[round(float(v), 3) for v in s["lab"]]] for s in samples
            if s["t"] >= start and (until_s is None or s["t"] <= until_s) and s.get("qa_valid") and s.get("lab")]


class EarlyWarning:
    """Per-dwell state: compare each new sample with the recipe's reference trajectory."""

    def __init__(self, band):
        self.band = band
        self.reset()

    def reset(self):
        self.samples, self.run, self.warned = [], 0, False

    def sample(self, sample, seq, recipe):
        """Returns an Action when this vial first leaves its reference trajectory, else None."""
        self.samples.append(sample)
        reference = ((recipe or {}).get("reference") or {}).get("trajectory")
        if self.warned or not reference or not (sample.get("qa_valid") and sample.get("lab")):
            return None
        start = fill_start(self.samples)
        if start is None:
            return None
        into = sample["t"] - start
        if into > reference[-1][0]:
            return None     # past the reference fill: the endpoint decision takes over
        times = np.array([p[0] for p in reference])
        expected = reference[int(np.argmin(np.abs(times - into)))][1:]
        de = delta_e(sample["lab"], expected)
        self.run = self.run + 1 if de > self.band else 0
        if self.run < CONSECUTIVE:
            return None
        self.warned = True
        return Action("EARLY_WARNING", seq, sample["t"], None,
                      f"Vial {seq} is leaving recipe {recipe['id']}'s reference fill: ΔE {de:.1f} at "
                      f"{into:.1f} s into the fill; check it before it reaches the sorter",
                      "FLAG_VIAL")


def from_run(frames, ledger, library):
    """Reference trajectories for a library whose references name a vial of a recorded run."""
    dwells = [json.loads(line) for line in Path(frames).read_text().splitlines() if line.strip()]
    rows = {int(r["seq"]): r for r in csv.DictReader(open(ledger))}
    out = {}
    for recipe in library["recipes"]:
        match = re.search(r"vial (\d+)", (recipe.get("reference") or {}).get("from", ""))
        row = match and rows.get(int(match.group(1)))
        if not row:
            continue
        fs, fe = float(row["fill_start_s"]), float(row["fill_end_s"])
        dwell = next((d for d in dwells if d["samples"] and d["samples"][0]["t"] <= fs <= d["samples"][-1]["t"]), None)
        points = trajectory(dwell["samples"], fe) if dwell else []
        if len(points) >= CONSECUTIVE:
            out[str(recipe["id"])] = {"vial": int(match.group(1)), "trajectory": points}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("frames", type=Path, help="line runner frames.jsonl of the run with the reference vials")
    parser.add_argument("ledger", type=Path, help="its ledger.csv (fill start/end per vial)")
    parser.add_argument("library", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    trajectories = from_run(args.frames, args.ledger, json.loads(args.library.read_text()))
    args.out.write_text(json.dumps({"schema": "linewatch-reference-trajectories/v1", "source": str(args.frames.name),
                                    "rule": f"QA-valid samples by seconds since fill start ({CONSECUTIVE} samples "
                                            f">= {MIN_FILL_PX} px), up to the vial's fill end",
                                    "recipes": trajectories}, indent=1))
    print({k: len(v["trajectory"]) for k, v in trajectories.items()})


if __name__ == "__main__":
    main()
