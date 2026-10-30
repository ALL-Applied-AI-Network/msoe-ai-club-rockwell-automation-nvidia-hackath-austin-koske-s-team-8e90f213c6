"""Carryover-aware recipe order: forecast each vial from what the line holds, then reorder runs.

Line model (one parameter, the residue fraction f): each fill leaves the line holding what it
dispensed, and the next vial is (1 - f) * its recipe + f * the line contents, mixed in optical
density (-log linear RGB, the forecast that tracked the synthetic renders best,
replay.carryover_forecast). A flush is one water vial (density 0), so each flush multiplies the
residue by f. A vial is a predicted scrap when its forecast sits more than the band from its
reference.

plan() orders the requested recipe runs (contiguous vials of one recipe) to minimise predicted
scrap and flush cost, choosing 0..max_flushes water vials at each changeover, and can limit how far
a run moves from its requested position. With a measured line state (the last vial's Lab) it
re-plans the remaining queue: heavy blue carryover makes an upcoming blue recipe the cheap next
run. Output is advice for the operator, as with replay.decide; nothing here changes equipment.
f is a line setting, not fitted to Run 5. A water vial costs a fill cycle but no product, so it
defaults to half a scrapped vial.

    python -m replay.schedule LIBRARY.json --queue 5x3 6x3 7x3 --f 0.15 [--max-shift 2] [--line-lab L a b]
"""
import argparse
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from skimage.color import lab2rgb

from replay.carryover_forecast import linear, srgb
from replay.decide import DEFAULT_LINE
from replay.measurement import delta_e, rgb_to_lab

SETTINGS = dict(band=DEFAULT_LINE["band_de2000"], max_flushes=2, flush_cost=0.5, scrap_cost=1.0)


def density(lab):
    rgb = lab2rgb(np.asarray(lab, dtype=float)[None, None])[0, 0] * 255
    return -np.log(linear(rgb))


def lab_of(d):
    return np.asarray(rgb_to_lab(srgb(np.exp(-d))))


def run_cost(line, recipe, count, refs, f, s, flushes=None):
    """Flushes before a run (best of 0..max_flushes, or the given number) from the line's density.

    Returns (cost, flushes, forecast ΔE per vial, line density after the run). line None is a
    clean start or a repeat of the same recipe, where no flush is considered.
    """
    target, ref = refs[recipe]
    best = None
    for k in [flushes] if flushes is not None else range(s["max_flushes"] + 1 if line is not None else 1):
        state = None if line is None else line * f ** k
        des = []
        for _ in range(count):
            state = target if state is None else (1 - f) * target + f * state
            des.append(delta_e(lab_of(state), ref))
        cost = k * s["flush_cost"] + s["scrap_cost"] * sum(de > s["band"] for de in des)
        if best is None or cost < best[0] - 1e-9:
            best = (cost, k, des, state)
    return best


def simulate(runs, refs, f, settings=None, line=None, flushes=None):
    """Cost of an order of (recipe, count) runs from line density `line` (None: clean start).

    flushes, if given, fixes the water vials before each run instead of choosing them.
    """
    s = {**SETTINGS, **(settings or {})}
    steps, total = [], 0.0
    for i, (recipe, count) in enumerate(runs):
        repeat = steps and steps[-1]["recipe"] == recipe
        cost, k, des, line = run_cost(line, recipe, count, refs, f, s,
                                      0 if repeat else flushes and flushes[i])
        total += cost
        steps.append(dict(recipe=recipe, vials=count, flushes=k, predicted_de=[round(d, 2) for d in des],
                          predicted_scraps=sum(d > s["band"] for d in des)))
    return total, steps


def plan(runs, refs, f, settings=None, max_shift=None, line=None):
    """Best run order by dynamic programming over subsets, then simulated exactly.

    The search costs each changeover from the previous run's reference colour (what the line holds
    once a run settles); the returned cost re-simulates the chosen order from `line`.
    """
    s = {**SETTINGS, **(settings or {})}
    n = len(runs)
    allowed = lambda i, pos: max_shift is None or abs(i - pos) <= max_shift

    @lru_cache(maxsize=None)
    def edge(i, j):  # cost of run j after run i (i = -1: the given line state)
        if i < 0:
            return run_cost(line, *runs[j], refs, f, s)[0]
        if runs[i][0] == runs[j][0]:
            return run_cost(None, *runs[j], refs, f, s)[0]
        return run_cost(refs[runs[i][0]][0], *runs[j], refs, f, s)[0]

    best = {(1 << j, j): (edge(-1, j), -1) for j in range(n) if allowed(j, 0)}
    for size in range(2, n + 1):
        for mask, last in [k for k in best if bin(k[0]).count("1") == size - 1]:
            for j in range(n):
                if not mask & 1 << j and allowed(j, size - 1):
                    cost = best[mask, last][0] + edge(last, j)
                    key = (mask | 1 << j, j)
                    if key not in best or cost < best[key][0] - 1e-9:
                        best[key] = (cost, last)
    ends = [k for k in best if k[0] == (1 << n) - 1]
    if not ends:
        raise ValueError("No order satisfies max_shift")
    mask, last = min(ends, key=lambda k: best[k][0])
    order = []
    while last >= 0:
        order.append(last)
        mask, last = mask & ~(1 << last), best[mask, last][1]
    order = [runs[i] for i in reversed(order)]
    return order, simulate(order, refs, f, s, line)


def references(library):
    return {str(r["id"]): (density(r["reference"]["lab"]), r["reference"]["lab"])
            for r in library["recipes"] if r.get("reference")}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("library", type=Path)
    parser.add_argument("--queue", nargs="+", required=True, help="runs as RECIPExCOUNT in requested order")
    parser.add_argument("--f", type=float, required=True, help="residue fraction a fill leaves for the next vial")
    parser.add_argument("--band", type=float, default=SETTINGS["band"])
    parser.add_argument("--max-flushes", type=int, default=SETTINGS["max_flushes"])
    parser.add_argument("--flush-cost", type=float, default=SETTINGS["flush_cost"], help="per water vial; a scrap costs 1")
    parser.add_argument("--max-shift", type=int)
    parser.add_argument("--line-lab", type=float, nargs=3, help="measured Lab of the last vial (re-plan from it)")
    args = parser.parse_args()
    refs = references(json.loads(args.library.read_text(encoding="utf-8")))
    runs = [(r, int(c)) for r, c in (q.split("x") for q in args.queue)]
    missing = sorted({r for r, _ in runs} - set(refs))
    if missing:
        parser.error(f"No reference colour for recipes {missing}")
    settings = dict(band=args.band, max_flushes=args.max_flushes, flush_cost=args.flush_cost)
    line = density(args.line_lab) if args.line_lab else None
    requested = simulate(runs, refs, args.f, settings, line)
    _, proposed = plan(runs, refs, args.f, settings, args.max_shift, line)
    print(json.dumps(dict(f=args.f, settings=settings, requested=dict(cost=requested[0], steps=requested[1]),
                          proposed=dict(cost=proposed[0], steps=proposed[1])), indent=1))


if __name__ == "__main__":
    main()
