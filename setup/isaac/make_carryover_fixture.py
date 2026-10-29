"""Seeded carryover fixture: recipe colours with residue of the previous recipe. No real input.

Each scene family draws three random dyes (per-channel absorbance), so no CSI dye colour,
recipe reference or Run 5 value is used. A family holds several previous -> next changeovers;
each changeover is rendered twice under independent nuisances: the clean intended mix
(reference, carryover 0) and the actual vial, whose absorbance is
(1 - f) * intended + f * previous with residue fraction f (0 for a third of vials).
Families are split train/dev as wholes. Labels are the mixing parameters, not optical truth.
"""
import argparse
import json
import math
from pathlib import Path
import random

from box_annotations import validate_fixture, write_json

VOLUMES = [0, 5, 10, 15, 20, 30]


def colour(absorbance, path):
    """Linear diffuse RGB and opacity for a per-channel absorbance over a light path."""
    rgb = [min(1.0, max(0.0, math.exp(-a * path))) for a in absorbance]
    opacity = min(0.9, max(0.12, 0.12 + 0.78 * (1 - math.exp(-sum(absorbance) * path / 3))))
    return rgb, opacity


def mix(dyes, volumes):
    total = sum(volumes)
    return [sum(v * d[c] for v, d in zip(volumes, dyes)) / total for c in range(3)]


def recipe(rng):
    while True:
        # Most recipes leave out one or two dyes, so changeovers can shift the colour strongly.
        v = [rng.choice(VOLUMES[1:]) if rng.random() < 0.55 else 0 for _ in range(3)]
        if sum(v) >= 10:
            return v


def make(families=48, changeovers=8, seed=29, dev_fraction=0.25):
    rng = random.Random(seed)
    dev = set(rng.sample(range(families), round(families * dev_fraction)))
    rows = []
    for fam in range(families):
        # Three dyes, each absorbing mostly in one or two channels, at a family-wide strength.
        dyes = []
        for _ in range(3):
            peak = rng.randrange(3)
            dyes.append([rng.uniform(1.5, 5.0) if c == peak else rng.uniform(0.0, 1.2) for c in range(3)])
        path = rng.uniform(0.2, 0.9)
        for k in range(changeovers):
            prev, nxt = recipe(rng), recipe(rng)
            f = 0.0 if rng.random() < 1 / 3 else rng.uniform(0.01, 0.30)
            a_next, a_prev = mix(dyes, nxt), mix(dyes, prev)
            actual = [(1 - f) * n + f * p for n, p in zip(a_next, a_prev)]
            pair = f"f{fam:03d}-c{k:02d}"
            for role, absorbance, frac in [("reference", a_next, 0.0), ("vial", actual, f)]:
                rgb, opacity = colour(absorbance, path)
                level = rng.uniform(0.70, 0.98) if rng.random() < 0.8 else rng.uniform(0.30, 0.70)
                rows.append(dict(
                    id=f"{pair}-{role[:3]}", level=level,
                    stream="off" if rng.random() < 0.85 else "continuous",
                    rgb=rgb, opacity=opacity, light_scale=rng.uniform(0.6, 1.4),
                    roughness=rng.uniform(0.06, 0.30), stream_radius_m=rng.uniform(0.0008, 0.0023),
                    kind="fill", family=fam, split="dev" if fam in dev else "train", pair=pair, role=role,
                    carryover_fraction=frac, previous_ml=prev, next_ml=nxt))
    base = json.loads(Path(__file__).with_name("box_fixture.json").read_text(encoding="utf-8"))
    result = dict(base, purpose="Synthetic carryover renders for the carryover estimator and detector retraining",
                  seed=seed, families=families, changeovers_per_family=changeovers,
                  dev_families=sorted(dev), real_data_inputs=[], recipe_or_camera_fitting=False,
                  nuisance_policy="Fixed supplied FillCam; random dyes per family; vary level, light intensity, "
                                  "roughness, stream; reference and vial rendered with independent nuisances",
                  cases=rows)
    validate_fixture(result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--families", type=int, default=48)
    parser.add_argument("--changeovers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=29)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Use a fresh fixture path")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fixture = make(args.families, args.changeovers, args.seed)
    write_json(args.out, fixture)
    print(f"Prepared {len(fixture['cases'])} cases from {args.families} families; seed {args.seed}; no real input")
