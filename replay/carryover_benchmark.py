"""Carryover benchmark scored against out-of-spec truth, with confusion matrices; synthetic only.

Earlier records (carryover_synthetic, carryover_noise) count any residue f > 0 as a positive, so a
vial with 5% of a similar recipe left in it is a "miss" even though it is in spec and passing it is
correct. Here the truth is the spec: a changeover vial is positive when its true colour is out of
the in-spec band (ΔE2000 > SPEC_DE to the intended reference) because of residue. True colour is the
optical-density mix of the clean reference and previous renders at the true f
(replay.carryover_forecast, within 0.43 ΔE median of the rendered vial), so lighting drift on the vial
render cannot make a vial "truly" out of spec. Negatives are vials whose true colour is in spec,
clean (f = 0) or not.

Measured colours get independent Gaussian noise (σ per Lab axis, default 1.0: clean vials then sit
a median 1.69 ΔE from their reference, matching real repeat vials in Runs 1-4) with several seeds.
Two decisions are scored on the same dev vials:

  SCRAP (out of spec, any cause): out of band (ΔE > 5) to the ordered recipe. Baseline: a check
  with no recipe order, which accepts any colour that matches a known recipe (ΔE <= 5 to the
  ordered or the previous recipe; see any_recipe).
  Carryover call (SCRAP with cause, which drives FLUSH): also toward the previous recipe, shipped
  as Lab cosine >= 0.8.

Each also has a selected variant: thresholds chosen on the train families (highest recall with
false alarms <= MAX_FALSE_ALARM on in-spec vials; direction in Lab or in either Lab or optical
density), then reported once on the dev families.

Run 5 is not read for any choice. --recolour scores an existing replay.recolour_eval record (Run 5
footage with injected residue at known f) against the same truth, after selection, with the
thresholds frozen from seed 0; vials QA refused are counted apart (CAN'T VERIFY holds the vial).

    python -m replay.carryover_benchmark RENDERS... --out benchmark.json [--rows-cache rows.json]
        [--recolour recolour-run5.json --library recipe-library.json]
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from replay.carryover_forecast import forecast
from replay.carryover_synthetic import MIN_PIXELS
from replay.decide import DEFAULT_LINE, toward_previous
from replay.measurement import color, delta_e

SPEC_DE = 5.0              # in-spec band to the intended recipe's reference
SIGMA = 1.0                # Lab noise per axis matched to Runs 1-4 repeat vials (replay.carryover_noise)
MAX_FALSE_ALARM = 0.05
TRUE_DE_BINS = [(5, 10), (10, 20), (20, 1e9)]


def measure(batch):
    """One row per render: pair, role, split, f, measured RGB and Lab."""
    rows = []
    for line in (batch / "manifest.jsonl").read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        p = record["parameters"]
        liquid = next(a for a in record["annotations"] if a["class_name"] == "liquid")
        rgb = lab = None
        if liquid["visible_pixels"] >= MIN_PIXELS:
            image = np.asarray(Image.open(batch / record["image_file"]).convert("RGB"))
            mask = np.asarray(Image.open(batch / liquid["mask_file"])) > 0
            value, lab, _ = color(image[mask])
            rgb = None if value is None else [float(x) for x in value]
            lab = None if lab is None else [float(x) for x in lab]
        rows.append(dict(batch=f"{batch.parent.parent.name}/{batch.name}", pair=p["pair"], role=p["role"],
                         split=p["split"], family=p["family"], f=p["carryover_fraction"],
                         light_drift=p.get("light_drift", 0.0), rgb=rgb, lab=lab))
    return rows


def changeovers(rows):
    """Reference, previous and vial per changeover with the true (noise-free, mixed) ΔE."""
    pairs = {}
    for row in rows:
        pairs.setdefault((row["batch"], row["pair"]), {})[row["role"]] = row
    out = []
    for key, roles in sorted(pairs.items()):
        if set(roles) != {"reference", "previous", "vial"} or any(r["lab"] is None for r in roles.values()):
            continue
        ref, prev, vial = roles["reference"], roles["previous"], roles["vial"]
        f = vial["f"]
        true_de = 0.0 if f == 0 else delta_e(forecast(ref, prev, f, "absorbance"), ref["lab"])
        out.append(dict(key="/".join(key), split=vial["split"], family=vial["family"], f=f,
                        light_drift=vial["light_drift"], true_de=float(true_de),
                        ref=np.asarray(ref["lab"]), prev=np.asarray(prev["lab"]), vial=np.asarray(vial["lab"])))
    return out


def observe(cs, sigma, rng):
    """Measured ΔE (to the reference and to the previous recipe) and direction cosines, with Lab noise
    on all three colours."""
    out = []
    for c in cs:
        ref, prev, vial = (c[k] + rng.normal(0, sigma, 3) for k in ("ref", "prev", "vial"))
        lab_cos = float(toward_previous(vial, ref, prev))
        out.append(dict(split=c["split"], f=c["f"], true_de=c["true_de"], positive=c["true_de"] > SPEC_DE,
                        de=float(delta_e(vial, ref)), de_prev=float(delta_e(vial, prev)), cos=lab_cos,
                        cos_either=max(lab_cos, float(toward_previous(vial, ref, prev, "density")))))
    return out


def rule(band_de2000, direction_cos=None, space="lab"):
    """Out of band to the ordered recipe, optionally also toward the previous recipe (carryover call)."""
    key = "cos" if space == "lab" else "cos_either"
    return lambda o: o["de"] > band_de2000 and (direction_cos is None or o[key] >= direction_cos)


def any_recipe(band):
    """Baseline with no recipe order: accept a vial that matches any known recipe colour. Only the
    ordered and previous recipes are known per synthetic changeover, so this flatters the baseline;
    a real library has more colours to match."""
    return lambda o: o["de"] > band and o["de_prev"] > band


def confusion(obs, flag, **params):
    """TP/FN/FP/TN against out-of-spec truth, recall by true ΔE and false alarms on clean vials."""
    tp = fn = fp = tn = 0
    for o in obs:
        hit = flag(o)
        if o["positive"]:
            tp, fn = tp + hit, fn + (not hit)
        else:
            fp, tn = fp + hit, tn + (not hit)
    clean = [o for o in obs if o["f"] == 0]
    by_true_de = [dict(true_de=[lo, hi if hi < 1e9 else None], n=sum(lo < o["true_de"] <= hi for o in obs),
                       flagged=sum(lo < o["true_de"] <= hi and flag(o) for o in obs))
                  for lo, hi in TRUE_DE_BINS]
    return dict(**params, tp=tp, fn=fn, fp=fp, tn=tn,
                recall=tp / (tp + fn) if tp + fn else None,
                precision=tp / (tp + fp) if tp + fp else None,
                false_alarm_rate=fp / (fp + tn) if fp + tn else None,
                false_alarm_rate_clean=sum(flag(o) for o in clean) / len(clean) if clean else None,
                by_true_de=by_true_de)


def select(obs, candidates):
    """Highest recall with false alarms <= MAX_FALSE_ALARM (ties: fewer false alarms).
    candidates: (params, flag) pairs."""
    best = None
    for params, flag in candidates:
        m = confusion(obs, flag, **params)
        if m["false_alarm_rate"] <= MAX_FALSE_ALARM and (
                best is None or (m["recall"], -m["false_alarm_rate"]) > (best["recall"], -best["false_alarm_rate"])):
            best = m
    return best


def score_recolour(record, library, frozen, band):
    """Recolour targets against out-of-spec truth from the library references at the injected f."""
    from skimage.color import lab2rgb
    refs = {str(r["id"]): r["reference"]["lab"] for r in library["recipes"] if r.get("reference")}

    def entry(lab):
        return dict(lab=lab, rgb=(lab2rgb(np.asarray(lab, float)[None, None])[0, 0] * 255).tolist())

    obs, unmeasured = [], []
    for result in record["results"]:
        for t in result["targets"]:
            ref, prev = entry(refs[t["recipe"]]), entry(refs[t["previous"]])
            true_de = 0.0 if result["f"] == 0 else float(delta_e(forecast(ref, prev, result["f"], "absorbance"), ref["lab"]))
            row = dict(seq=t["seq"], f=result["f"], true_de=true_de, positive=true_de > SPEC_DE)
            if t["de_intended"] is None:
                unmeasured.append(row)
                continue
            obs.append(dict(row, de=t["de_intended"], de_prev=t["de_previous"], cos=t["direction_cos"],
                            cos_either=max(t["direction_cos"], t["density_cos"])))
    metrics = dict(
        scrap_shipped=confusion(obs, rule(band), band_de2000=band),
        scrap_selected=confusion(obs, rule(frozen["scrap"]["band_de2000"]), **frozen["scrap"]),
        baseline_any_recipe=confusion(obs, any_recipe(band), band_de2000=band),
        carryover_shipped=confusion(obs, rule(band, DEFAULT_LINE["direction_cos"]), band_de2000=band,
                                    direction_cos=DEFAULT_LINE["direction_cos"], space="lab"),
        carryover_selected=confusion(obs, rule(**frozen["carryover"]), **frozen["carryover"]))
    return dict(targets=len(obs) + len(unmeasured), measured=len(obs),
                cant_verify=dict(positive=sum(u["positive"] for u in unmeasured),
                                 negative=sum(not u["positive"] for u in unmeasured)),
                vials=[{k: v for k, v in o.items()} for o in obs] + [dict(u, de=None) for u in unmeasured],
                metrics=metrics)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("renders", type=Path, nargs="*")
    parser.add_argument("--rows-cache", type=Path, help="measured render rows; written if missing, read if present")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sigma", type=float, default=SIGMA)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--recolour", type=Path, help="replay.recolour_eval record to score with frozen thresholds")
    parser.add_argument("--library", type=Path, help="recipe library with the recolour references")
    args = parser.parse_args()
    if args.rows_cache and args.rows_cache.exists():
        rows = json.loads(args.rows_cache.read_text(encoding="utf-8"))
    else:
        rows = [row for batch in args.renders for row in measure(batch)]
        if args.rows_cache:
            args.rows_cache.write_text(json.dumps(rows), encoding="utf-8")
    cs = changeovers(rows)
    line = DEFAULT_LINE
    bands = [float(b) for b in np.arange(1.0, 12.5, 0.5)]
    coss = [round(float(c), 2) for c in np.arange(0.5, 0.96, 0.05)]
    band_only = [(dict(band_de2000=b), rule(b)) for b in bands]
    direction = [(dict(band_de2000=b, direction_cos=c, space=space), rule(b, c, space))
                 for space in ("lab", "either") for b in bands for c in coss]
    runs = []
    for seed in range(args.seeds):
        obs = observe(cs, args.sigma, np.random.default_rng(seed))
        train = [o for o in obs if o["split"] == "train"]
        dev = [o for o in obs if o["split"] == "dev"]
        chosen = dict(scrap=select(train, band_only), carryover=select(train, direction))
        p = {k: {n: v for n, v in m.items() if n in ("band_de2000", "direction_cos", "space")} for k, m in chosen.items()}
        band = line["band_de2000"]
        runs.append(dict(seed=seed, selected_on_train=p, metrics=dict(
            # out of spec, any cause (the SCRAP decision)
            scrap_shipped=confusion(dev, rule(band), band_de2000=band),
            scrap_selected=confusion(dev, rule(p["scrap"]["band_de2000"]), **p["scrap"]),
            baseline_any_recipe=confusion(dev, any_recipe(band), band_de2000=band),
            # out of spec and called carryover (SCRAP + FLUSH with the cause)
            carryover_shipped=confusion(dev, rule(band, line["direction_cos"]), band_de2000=band,
                                        direction_cos=line["direction_cos"], space="lab"),
            carryover_selected=confusion(dev, rule(**p["carryover"]), **p["carryover"]))))
        print(f"seed {seed}: " + " | ".join(
            f"{name} recall {m['recall']:.3f} FA {m['false_alarm_rate']:.3f} (TP {m['tp']} FN {m['fn']} FP {m['fp']} TN {m['tn']})"
            for name, m in runs[-1]["metrics"].items()) + f" | selected {p}", flush=True)
    recolour = None
    if args.recolour:
        frozen = runs[0]["selected_on_train"]
        recolour = dict(record=str(args.recolour), frozen_from_seed=0, **score_recolour(
            json.loads(args.recolour.read_text(encoding="utf-8")), json.loads(args.library.read_text(encoding="utf-8")),
            frozen, line["band_de2000"]))
        for name, m in recolour["metrics"].items():
            print(f"recolour {name}: TP {m['tp']} FN {m['fn']} FP {m['fp']} TN {m['tn']}", flush=True)
    dev = [c for c in cs if c["split"] == "dev"]
    args.out.write_text(json.dumps(dict(
        spec_de2000=SPEC_DE, sigma=args.sigma, seeds=args.seeds, max_false_alarm=MAX_FALSE_ALARM,
        changeovers=len(cs), dev_changeovers=len(dev), dev_positive=sum(c["true_de"] > SPEC_DE for c in dev),
        dev_clean=sum(c["f"] == 0 for c in dev),
        truth="positive = true colour (optical-density mix of clean reference and previous at the true f) more than "
              f"ΔE2000 {SPEC_DE} from the intended reference",
        note="Synthetic renders only; thresholds selected on train families, reported on dev families. No Run 5.",
        runs=runs, recolour=recolour), indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
