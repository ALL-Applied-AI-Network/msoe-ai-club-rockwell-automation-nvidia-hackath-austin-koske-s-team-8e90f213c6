"""Score a line run against ground truth (vial,recipe_id,start_frame,end_frame CSV).

Segmentation, schedule alignment, coverage, repeatability and timeliness are scored directly.
Run5 has no per-vial pass/fail truth, so verdicts are reported by position after a recipe change
(where carryover is expected), not as accuracy. Run5 is also the only labelled run, so its recipe
references come from its own vials: those reference vials are reported but left out of the verdict
counts, and each first vial after a change is re-checked against every other repeat vial of its
recipe to show how much its verdict depends on which vial supplied the reference.
"""
import csv
from collections import defaultdict
from itertools import combinations

import numpy as np

from replay.measurement import delta_e

FPS = 30.0


def _overlap(a0, a1, b0, b1):
    return max(0, min(a1, b1) - max(a0, b0))


def score(vials, verdicts, actions, labels_path, schedule, reference_vials=(), band=5.0):
    """reference_vials: labelled vial numbers whose endpoints are recipe references (not scored)."""
    with open(labels_path) as handle:
        truth = [{"vial": int(r["vial"]), "recipe_id": str(r["recipe_id"]), "start": int(r["start_frame"]),
                  "end": int(r["end_frame"])} for r in csv.DictReader(handle)]
    position, run = {}, 0
    for i, t in enumerate(truth):
        run = run + 1 if i and truth[i - 1]["recipe_id"] == t["recipe_id"] else 1
        position[t["vial"]] = run

    pairs = sorted(((_overlap(v["fill_start_frame"], v["fill_end_frame"], t["start"], t["end"]), i, j)
                    for i, v in enumerate(vials) for j, t in enumerate(truth)), reverse=True)
    match, used = {}, set()
    for overlap, i, j in pairs:
        if overlap > 0 and i not in match and j not in used:
            match[i] = j
            used.add(j)

    start_err = [abs(vials[i]["fill_start_frame"] - truth[j]["start"]) / FPS for i, j in match.items()]
    end_err = [abs(vials[i]["fill_end_frame"] - truth[j]["end"]) / FPS for i, j in match.items()]
    aligned = sum(verdicts[i].recipe_id == truth[j]["recipe_id"] for i, j in match.items())

    references = {int(v) for v in reference_vials}
    rows, by_recipe = [], defaultdict(list)
    for i, (vial, v) in enumerate(zip(vials, verdicts)):
        t = truth[match[i]] if i in match else None
        rows.append({"seq": vial["seq"], "true_vial": t and t["vial"], "true_recipe": t and t["recipe_id"],
                     "scheduled_recipe": v.recipe_id, "position_after_change": t and position[t["vial"]],
                     "role": "reference" if t and t["vial"] in references else "scored",
                     "verdict": v.verdict, "pattern": v.pattern, "de_intended": v.de_intended})
        if t and vial["endpoint_lab"] and position[t["vial"]] > 1:
            by_recipe[t["recipe_id"]].append((t["vial"], vial["endpoint_lab"]))
    repeat = [delta_e(a, b) for labs in by_recipe.values() for (_, a), (_, b) in combinations(labs, 2)]

    by_position = defaultdict(lambda: defaultdict(int))
    for r in rows:
        if r["position_after_change"] and r["role"] == "scored":
            by_position[r["position_after_change"]][r["verdict"]] += 1

    # First vials after a change, re-scored against each repeat vial of the recipe as the reference.
    sensitivity = []
    for i, (vial, v) in enumerate(zip(vials, verdicts)):
        t = truth[match[i]] if i in match else None
        if not (t and position[t["vial"]] == 1 and vial["endpoint_lab"] and by_recipe[t["recipe_id"]]):
            continue
        alternatives = {str(n): round(delta_e(vial["endpoint_lab"], lab), 2) for n, lab in by_recipe[t["recipe_id"]]}
        sides = {d > band for d in alternatives.values()} | ({v.de_intended > band} if v.de_intended is not None else set())
        sensitivity.append({"true_vial": t["vial"], "recipe": t["recipe_id"],
                            "de_frozen_reference": v.de_intended and round(v.de_intended, 2),
                            "de_if_reference_were_vial": alternatives, "verdict_stable": len(sides) == 1})

    latency = [v["decision_s"] - v["fill_end_s"] for v in vials]
    scrap_on_time = [a.issued_s <= a.deadline_s for a in actions if a.kind == "SCRAP"]
    flush_on_time = [a.deadline_s is None or a.issued_s <= a.deadline_s for a in actions if a.kind == "FLUSH"]
    flush_after = {a.seq for a in actions if a.kind == "FLUSH"}
    # Counterfactual: next vial still out of spec toward the same previous recipe after a FLUSH was issued.
    preventable = [v.seq for v in verdicts if v.seq - 1 in flush_after and v.pattern == "carryover"]

    detected = len(vials)
    summary = {
        "scheduled_vials": len(schedule), "labelled_vials": len(truth), "detected_vials": detected,
        "segmentation": {"matched": len(match), "false_positives": detected - len(match),
                         "missed": len(truth) - len(match),
                         "fill_start_error_s_median": round(float(np.median(start_err)), 2) if start_err else None,
                         "fill_end_error_s_median": round(float(np.median(end_err)), 2) if end_err else None},
        "schedule_alignment": f"{aligned}/{len(match)} matched vials got the right recipe",
        "endpoint_coverage": f"{sum(v['endpoint_lab'] is not None for v in vials)}/{detected}",
        "repeatability_de_median_non_first": round(float(np.median(repeat)), 2) if repeat else None,
        "reference_vials_not_scored": sorted(r["true_vial"] for r in rows if r["role"] == "reference"),
        "verdicts_by_position_after_change": {str(k): dict(v) for k, v in sorted(by_position.items())},
        "first_vial_reference_sensitivity": {
            "checked": len(sensitivity), "verdict_stable": sum(s["verdict_stable"] for s in sensitivity),
            "band_de2000": band},
        "actions": {k: sum(a.kind == k for a in actions) for k in
                    ("PASS", "SCRAP", "FLUSH", "FLUSH_RECOMMENDED", "STOP_CHECK_DOSING", "HOLD")},
        "decision_latency_s_median": round(float(np.median(latency)), 1) if latency else None,
        "scrap_before_deadline": f"{sum(scrap_on_time)}/{len(scrap_on_time)}",
        "flush_before_next_fill": f"{sum(flush_on_time)}/{len(flush_on_time)}",
        "vials_a_flush_would_likely_have_saved": preventable,
    }
    return {"summary": summary, "vials": rows, "reference_sensitivity": sensitivity,
            "notes": "Run5 has no per-vial outcome truth; verdict counts by position after a recipe change "
                     "are descriptive. References are proxies from Run5 itself (3rd vial, 2nd as fallback); "
                     "those vials are listed and excluded from the verdict counts."}
