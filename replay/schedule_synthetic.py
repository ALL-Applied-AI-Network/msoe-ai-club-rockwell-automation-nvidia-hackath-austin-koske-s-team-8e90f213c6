"""Benchmark replay.schedule on random synthetic recipe queues; no real input.

Dyes and recipes are drawn as in setup/isaac/make_carryover_fixture.py (three dyes per family,
most recipes leave out one or two). Each queue has 6 runs of 2-4 vials. Policies, all scored
with the schedule's own line model (so these are predicted savings, not measured ones):

  as requested, no flush | as requested, the current proactive rule (one water vial when the
  references differ by more than 15 ΔE) | as requested, best flushes | reordered within 2
  positions | reordered freely

    python -m replay.schedule_synthetic --out schedule-synthetic.json [--queues 300]
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np

from replay.decide import DEFAULT_LINE
from replay.measurement import delta_e
from replay.schedule import SETTINGS, lab_of, plan, simulate

VOLUMES = [5, 10, 15, 20, 30]


def queue(rng, runs=6):
    dyes = []
    for _ in range(3):
        peak = rng.randrange(3)
        dyes.append([rng.uniform(1.5, 5.0) if c == peak else rng.uniform(0.0, 1.2) for c in range(3)])
    path = rng.uniform(0.2, 0.9)
    refs = {}
    while len(refs) < runs:
        v = [rng.choice(VOLUMES) if rng.random() < 0.55 else 0 for _ in range(3)]
        if sum(v) >= 10:
            d = np.array([sum(x * dye[c] for x, dye in zip(v, dyes)) / sum(v) for c in range(3)]) * path
            refs[str(len(refs))] = (d, lab_of(d).tolist())
    return refs, [(r, rng.randint(2, 4)) for r in refs]


def proactive(runs, refs):
    gap = DEFAULT_LINE["proactive_flush_de"]
    return [0] + [int(delta_e(refs[a][1], refs[b][1]) > gap) for (a, _), (b, _) in zip(runs, runs[1:])]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--queues", type=int, default=300)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--fractions", type=float, nargs="+", default=[0.05, 0.1, 0.2, 0.3])
    args = parser.parse_args()
    rng = random.Random(args.seed)
    queues = [queue(rng) for _ in range(args.queues)]
    results = []
    for f in args.fractions:
        totals = {}
        for refs, runs in queues:
            policies = {"requested, no flush": simulate(runs, refs, f, dict(max_flushes=0)),
                        "requested, current proactive rule": simulate(runs, refs, f, flushes=proactive(runs, refs)),
                        "requested, best flushes": simulate(runs, refs, f),
                        "reordered within 2": plan(runs, refs, f, max_shift=2)[1],
                        "reordered freely": plan(runs, refs, f)[1]}
            for name, (cost, steps) in policies.items():
                t = totals.setdefault(name, dict(cost=0.0, scraps=0, flushes=0))
                t["cost"] += cost
                t["scraps"] += sum(s["predicted_scraps"] for s in steps)
                t["flushes"] += sum(s["flushes"] for s in steps)
        per = {k: {m: v / len(queues) for m, v in t.items()} for k, t in totals.items()}
        results.append(dict(f=f, per_queue=per))
        print(f"f {f}: " + "; ".join(f"{k} {v['scraps']:.2f} scraps {v['flushes']:.2f} flushes" for k, v in per.items()))
    vials = sum(c for _, runs in queues for _, c in runs) / len(queues)
    args.out.write_text(json.dumps(dict(queues=args.queues, runs_per_queue=6, vials_per_queue=vials, seed=args.seed,
                                        settings=SETTINGS, results=results, real_data_inputs=[],
                                        note="Scored with the schedule's own line model: predicted, not measured."),
                                   indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
