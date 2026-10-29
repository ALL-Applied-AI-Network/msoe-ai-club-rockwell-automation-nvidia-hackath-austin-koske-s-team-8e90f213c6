"""The FLUSH rule and a residue estimate on synthetic carryover renders; no real input.

Reads render batches from setup/isaac/make_carryover_fixture.py (each changeover rendered as the
clean intended mix, the clean previous mix and the vial with residue fraction f). Each render's
liquid colour is measured with the measurement module's trimmed median over the renderer's
liquid mask, so this tests the decision rule and the colour geometry, not segmentation.

For every changeover it applies the line rule (replay.decide: out of the ΔE2000 band and Lab
direction cosine toward the previous recipe >= threshold) and a residue estimate: the projection
of (vial - reference) onto (previous - reference) in Lab, clipped to [0, 1]. Thresholds may be
swept on the train families; the dev families report the result. Run 5 is not read.

    python -m replay.carryover_synthetic RENDERS... --out carryover-eval.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from replay.decide import DEFAULT_LINE, toward_previous
from replay.measurement import color, delta_e

BINS = [(0.0, 0.0), (0.0, 0.05), (0.05, 0.10), (0.10, 0.20), (0.20, 0.30), (0.30, 0.50), (0.50, 1.0)]
MIN_PIXELS = 60   # the line's insufficient_final_liquid_pixels limit


def measure(batch):
    """One row per render in a batch folder: pair, role, split, f and the liquid Lab."""
    rows = []
    for line in (batch / "manifest.jsonl").read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        p = record["parameters"]
        liquid = next(a for a in record["annotations"] if a["class_name"] == "liquid")
        lab, pixels = None, liquid["visible_pixels"]
        if pixels >= MIN_PIXELS:
            rgb = np.asarray(Image.open(batch / record["image_file"]).convert("RGB"))
            mask = np.asarray(Image.open(batch / liquid["mask_file"])) > 0
            _, lab, _ = color(rgb[mask])
        rows.append(dict(id=record["sample_id"], pair=p["pair"], role=p["role"], split=p["split"],
                         family=p["family"], f=p["carryover_fraction"], level=p["level"],
                         light_scale=p.get("light_scale", 1.0), light_drift=p.get("light_drift", 0.0),
                         pixels=pixels, lab=lab))
    return rows


def changeovers(rows):
    """Reference, previous and vial measurements grouped per changeover; incomplete ones dropped."""
    pairs = {}
    for row in rows:
        pairs.setdefault(row["pair"], {})[row["role"]] = row
    out = []
    for pair, roles in sorted(pairs.items()):
        if set(roles) != {"reference", "previous", "vial"} or any(r["lab"] is None for r in roles.values()):
            continue
        ref, prev, vial = (np.asarray(roles[k]["lab"]) for k in ("reference", "previous", "vial"))
        w = prev - ref
        residue = float(np.clip((vial - ref) @ w / (w @ w + 1e-9), 0, 1))
        out.append(dict(pair=pair, split=roles["vial"]["split"], family=roles["vial"]["family"],
                        f=roles["vial"]["f"], light_drift=roles["vial"]["light_drift"], de_intended=delta_e(vial, ref), de_change=delta_e(prev, ref),
                        direction_cos=toward_previous(vial, ref, prev), residue_estimate=residue))
    return out


def flagged(c, band, cos):
    return c["de_intended"] > band and c["direction_cos"] >= cos


def summary(cs, band, cos):
    """Rule detection by residue bin, false alarms at f = 0, and residue-estimate error."""
    bins = []
    for low, high in BINS:
        sel = [c for c in cs if (c["f"] == 0 if high == 0 else low < c["f"] <= high)]
        bins.append(dict(f_range=[low, high], n=len(sel),
                         flagged=sum(flagged(c, band, cos) for c in sel),
                         out_of_band=sum(c["de_intended"] > band for c in sel),
                         median_de=float(np.median([c["de_intended"] for c in sel])) if sel else None,
                         residue_mae=float(np.mean([abs(c["residue_estimate"] - c["f"]) for c in sel])) if sel else None))
    positive = [c for c in cs if c["f"] > 0]
    errors = [abs(c["residue_estimate"] - c["f"]) for c in cs]
    large = [abs(c["residue_estimate"] - c["f"]) for c in cs if c["de_change"] > 15]
    return dict(changeovers=len(cs), band_de2000=band, direction_cos=cos, bins=bins,
                false_alarm_rate=bins[0]["flagged"] / bins[0]["n"] if bins[0]["n"] else None,
                detection_rate=sum(flagged(c, band, cos) for c in positive) / len(positive) if positive else None,
                residue_mae=float(np.mean(errors)) if errors else None,
                residue_mae_large_change=float(np.mean(large)) if large else None)


def sweep(cs, bands=np.arange(2.0, 10.5, 0.5), coss=np.arange(0.5, 0.96, 0.05), max_false_alarm=0.05):
    """Highest detection rate with false alarms <= max_false_alarm; train families only."""
    best = None
    for band in bands:
        for cos in coss:
            s = summary(cs, float(band), round(float(cos), 2))
            if s["false_alarm_rate"] is not None and s["false_alarm_rate"] <= max_false_alarm:
                if s["detection_rate"] is not None and (best is None or s["detection_rate"] > best["detection_rate"]):
                    best = s
    return best


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("renders", type=Path, nargs="+", help="render batch folders (each with manifest.jsonl)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = [row for batch in args.renders for row in measure(batch)]
    cs = changeovers(rows)
    train = [c for c in cs if c["split"] == "train"]
    dev = [c for c in cs if c["split"] == "dev"]
    line = DEFAULT_LINE
    tuned = sweep(train) if train else None
    result = dict(
        renders=len(rows), unmeasured=sum(r["lab"] is None for r in rows), changeovers=len(cs),
        current_rule=dict(train=summary(train, line["band_de2000"], line["direction_cos"]),
                          dev=summary(dev, line["band_de2000"], line["direction_cos"])),
        tuned_on_train=tuned and dict(band_de2000=tuned["band_de2000"], direction_cos=tuned["direction_cos"],
                                      train=tuned, dev=summary(dev, tuned["band_de2000"], tuned["direction_cos"])),
        dev_by_light_drift={name: summary([c for c in dev if pick(c)], line["band_de2000"], line["direction_cos"])
                            for name, pick in [("none", lambda c: c["light_drift"] == 0),
                                               ("5-15%", lambda c: c["light_drift"] != 0)]},
        note="Synthetic renders, legacy opacity liquid; labels are mixing parameters, not optical truth. No Run 5.",
        changeover_rows=cs)
    args.out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    for name, s in [("current rule, dev", result["current_rule"]["dev"]),
                    ("tuned on train, dev", result["tuned_on_train"] and result["tuned_on_train"]["dev"])]:
        if s:
            print(f"{name}: band {s['band_de2000']} cos {s['direction_cos']}: detection {s['detection_rate']}, "
                  f"false alarms {s['false_alarm_rate']}, residue MAE {s['residue_mae']}")


if __name__ == "__main__":
    main()
