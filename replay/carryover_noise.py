"""FLUSH-rule thresholds under realistic measurement noise, on the synthetic carryover renders.

Synthetic clean vials sit 0.67 ΔE2000 (median) from their reference, far below real repeat-vial
variation, so thresholds selected on clean renders would flag many good real vials. This adds
independent Gaussian noise (σ per Lab axis) to every render's measured colour, for several σ and
noise seeds, and reports for each σ:

  - the median ΔE2000 of clean (f = 0) vials to their reference, to compare with real repeat vials
    (Runs 1-4: consecutive vials closer than ΔE 5, assumed to be repeats of one recipe, median 1.8);
  - thresholds selected on the train families (highest detection with false alarms <= 5%),
    and their detection by residue and false-alarm rate on the dev families.

No Run 5 value is read. Runs 1-4 only set the noise level to report against.

    python -m replay.carryover_noise RENDERS... --out noise.json [--sigmas 0 0.5 1 1.5 2 3] [--seeds 3]
"""
import argparse
import json
from pathlib import Path

import numpy as np

from replay.carryover_synthetic import changeovers, measure, summary, sweep
from replay.decide import DEFAULT_LINE

REAL_REPEAT_DE = 1.8   # Runs 1-4, consecutive vials with ΔE < 5 (assumed same recipe), median


def noisy(rows, sigma, rng):
    return [{**r, "lab": None if r["lab"] is None else (np.asarray(r["lab"]) + rng.normal(0, sigma, 3)).tolist()}
            for r in rows]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("renders", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sigmas", type=float, nargs="+", default=[0.0, 0.5, 1.0, 1.5, 2.0, 3.0])
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--direction", choices=("lab", "density"), default="lab",
                        help="space of the toward-previous cosine (replay.decide.toward_previous)")
    args = parser.parse_args()
    rows = [row for batch in args.renders for row in measure(batch)]
    line = DEFAULT_LINE
    results = []
    for sigma in args.sigmas:
        runs = []
        for seed in range(args.seeds):
            cs = changeovers(noisy(rows, sigma, np.random.default_rng(seed)))
            if args.direction == "density":
                cs = [{**c, "direction_cos": c["density_cos"]} for c in cs]
            train = [c for c in cs if c["split"] == "train"]
            dev = [c for c in cs if c["split"] == "dev"]
            clean = [c["de_intended"] for c in dev if c["f"] == 0]
            tuned = sweep(train, bands=np.arange(1.0, 12.5, 0.5))
            runs.append(dict(seed=seed, clean_median_de=float(np.median(clean)),
                             current=summary(dev, line["band_de2000"], line["direction_cos"]),
                             tuned=tuned and dict(band=tuned["band_de2000"], cos=tuned["direction_cos"],
                                                  dev=summary(dev, tuned["band_de2000"], tuned["direction_cos"]))))
        results.append(dict(sigma=sigma, runs=runs))
        clean = np.median([r["clean_median_de"] for r in runs])
        line_out = [f"σ {sigma}: clean median ΔE {clean:.2f}"]
        for name in ("current", "tuned"):
            for r in runs[:1]:
                s = r[name] if name == "current" else r[name] and r[name]["dev"]
                if s:
                    det = [round(b["flagged"] / b["n"], 2) if b["n"] else None for b in s["bins"]]
                    label = "current" if name == "current" else f"tuned band {r['tuned']['band']} cos {r['tuned']['cos']}"
                    line_out.append(f"{label}: false alarms {s['false_alarm_rate']:.3f} detection by bin {det}")
        print(" | ".join(line_out), flush=True)
    args.out.write_text(json.dumps(dict(real_repeat_de=REAL_REPEAT_DE, renders=len(rows), seeds=args.seeds, direction=args.direction,
                                        note="Synthetic only; noise level compared with Runs 1-4, never Run 5.",
                                        results=results), indent=1))


if __name__ == "__main__":
    main()
