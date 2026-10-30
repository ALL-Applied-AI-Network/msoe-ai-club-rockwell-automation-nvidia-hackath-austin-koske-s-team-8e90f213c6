"""Forecast a changeover's colour shift from the two recipes' reference colours and a residue fraction.

A scheduler needs the cost of each previous -> next changeover before it is filled: how far the
first vial of the next recipe will sit from its reference if a fraction f of the previous recipe
is left in the line. Two forecasts of the vial colour, with no fitted parameters:

  lab:        reference + f * (previous - reference) in Lab
  absorbance: per-channel optical density -log(linear RGB) mixed as (1 - f) * next + f * previous,
              the mixing the synthetic fixture uses for the dyes

Tested on the synthetic carryover renders with their true f: the forecast ΔE to the reference
against the measured one, and agreement on "out of band". Dev families report the result; nothing
is fitted, so train families are reported too. No real image or Run 5 value is read.

    python -m replay.carryover_forecast RENDERS... --out forecast.json [--band 5]
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from replay.carryover_synthetic import BINS, MIN_PIXELS
from replay.measurement import color, delta_e, rgb_to_lab


def linear(rgb):
    c = np.clip(np.asarray(rgb, dtype=float) / 255, 1e-4, 1)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def srgb(lin):
    c = np.clip(lin, 0, 1)
    return 255 * np.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1 / 2.4) - 0.055)


def forecast(ref, prev, f, how):
    """Forecast vial Lab from reference and previous (each dict with rgb and lab)."""
    if how == "lab":
        return np.asarray(ref["lab"]) + f * (np.asarray(prev["lab"]) - np.asarray(ref["lab"]))
    density = (1 - f) * -np.log(linear(ref["rgb"])) + f * -np.log(linear(prev["rgb"]))
    return np.asarray(rgb_to_lab(srgb(np.exp(-density))))


def measure(batch):
    pairs = {}
    for line in (batch / "manifest.jsonl").read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        p = record["parameters"]
        liquid = next(a for a in record["annotations"] if a["class_name"] == "liquid")
        if liquid["visible_pixels"] < MIN_PIXELS:
            continue
        rgb = np.asarray(Image.open(batch / record["image_file"]).convert("RGB"))
        mask = np.asarray(Image.open(batch / liquid["mask_file"])) > 0
        value, lab, _ = color(rgb[mask])
        if lab is not None:
            pairs.setdefault(p["pair"], dict(split=p["split"]))[p["role"]] = dict(rgb=value, lab=lab, f=p["carryover_fraction"])
    return [v for v in pairs.values() if {"reference", "previous", "vial"} <= set(v)]


def summary(rows, band):
    out = dict(changeovers=len(rows))
    for how in ("lab", "absorbance"):
        err = np.array([r[f"{how}_de"] - r["measured_de"] for r in rows])
        agree = np.mean([(r[f"{how}_de"] > band) == (r["measured_de"] > band) for r in rows])
        bins = []
        for low, high in BINS:
            sel = [r for r in rows if (r["f"] == 0 if high == 0 else low < r["f"] <= high)]
            if sel:
                e = np.array([r[f"{how}_de"] - r["measured_de"] for r in sel])
                bins.append(dict(f_range=[low, high], n=len(sel), median_measured_de=float(np.median([r["measured_de"] for r in sel])),
                                 median_abs_error=float(np.median(np.abs(e))), median_error=float(np.median(e))))
        out[how] = dict(median_abs_error_de=float(np.median(np.abs(err))), p90_abs_error_de=float(np.percentile(np.abs(err), 90)),
                        out_of_band_agreement=float(agree), by_bin=bins)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("renders", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--band", type=float, default=5.0)
    args = parser.parse_args()
    rows = []
    for batch in args.renders:
        for c in measure(batch):
            ref, prev, vial = c["reference"], c["previous"], c["vial"]
            f = vial["f"]
            rows.append(dict(split=c["split"], f=f, measured_de=delta_e(vial["lab"], ref["lab"]),
                             **{f"{how}_de": delta_e(forecast(ref, prev, f, how), ref["lab"]) for how in ("lab", "absorbance")}))
    result = dict(band_de2000=args.band, note="Synthetic renders with true residue fraction; no parameters fitted. No Run 5.",
                  **{split: summary([r for r in rows if r["split"] == split], args.band) for split in ("train", "dev")})
    args.out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    for how in ("lab", "absorbance"):
        d = result["dev"][how]
        print(f"dev {how}: median |error| {d['median_abs_error_de']:.2f} dE, p90 {d['p90_abs_error_de']:.2f}, "
              f"out-of-band agreement {d['out_of_band_agreement']:.3f}")


if __name__ == "__main__":
    main()
