"""Stage 4: per-vial verdicts, line actions and deadlines from measured endpoints.

Pure functions over vial records, the recipe schedule and the recipe library; no models.
Actions are advice to the operator (and the command a PLC integration would send); nothing
here controls equipment. Thresholds are research settings, not commissioned tolerances.
"""
from dataclasses import asdict, dataclass, field

import numpy as np

from replay.measurement import delta_e, optical_density

DEFAULT_LINE = {
    "band_de2000": 5.0,          # ΔE2000 to the intended recipe's reference
    "direction_cos": 0.75,       # direction toward the previous recipe = carryover
    "direction_space": "either", # larger of the Lab and optical-density cosines (replay.carryover_benchmark)
    "scrap_deadline_s": 120.0,   # CSI: a vial reaches the sorter 2-3 min after fill end
    "persistent_count": 3,       # consecutive non-carryover deviations -> stop and check dosing
    "proactive_flush_de": 15.0,  # previous->next reference distance that warrants a flush first
}


@dataclass
class Action:
    kind: str                    # PASS, SCRAP, FLUSH, STOP_CHECK_DOSING, HOLD, FLUSH_RECOMMENDED
    seq: int | None              # vial the action concerns (None for line-level advice)
    issued_s: float              # video time the action is issued
    deadline_s: float | None     # video time by which it must be carried out
    reason: str
    plc_command: str             # what an integration would send; simulated only


@dataclass
class Verdict:
    seq: int
    recipe_id: str
    previous_recipe_id: str | None
    verdict: str                 # IN_SPEC, OUT_OF_SPEC, CANT_VERIFY
    pattern: str                 # nominal, carryover, deviation, persistent_deviation, unmeasured, no_reference
    endpoint_lab: list | None
    de_intended: float | None
    de_previous: float | None
    direction_cos: float | None
    actions: list = field(default_factory=list)


def _reference(library, recipe_id):
    for recipe in library["recipes"]:
        if str(recipe["id"]) == str(recipe_id):
            return recipe.get("reference") and recipe["reference"]["lab"]
    return None


def toward_previous(lab, reference, previous, space="lab"):
    """Cosine between (vial - reference) and (previous - reference), in Lab or in optical density.

    Dyes mix in optical density, so a carryover mix lies on the straight line reference -> previous
    in density but on a curve in Lab: for very different recipes the Lab cosine of a pure mix can
    be about 0.5 (Run 5 recipe 7 after 6). "density" measures the direction where the mix is linear;
    "either" takes the larger of the two.
    """
    if space == "either":
        return max(toward_previous(lab, reference, previous), toward_previous(lab, reference, previous, "density"))
    if space == "density":
        lab, reference, previous = (optical_density(x) for x in (lab, reference, previous))
    u = np.subtract(lab, reference)
    w = np.subtract(previous, reference)
    return float(u @ w / (np.linalg.norm(u) * np.linalg.norm(w) + 1e-9))


def proactive_flush(seq, schedule, library, line, issued_s, deadline_s):
    """FLUSH_RECOMMENDED before the first vial of a new recipe, if the change is large; else None."""
    line = {**DEFAULT_LINE, **(line or {})}
    if seq < 2 or seq > len(schedule) or schedule[seq - 2] == schedule[seq - 1]:
        return None
    recipe = schedule[seq - 1]
    earlier = [r for r in schedule[:seq - 1] if r != recipe]
    previous = earlier[-1] if earlier else None
    ref, prev_ref = _reference(library, recipe), previous and _reference(library, previous)
    if not (ref and prev_ref):
        return None
    gap = delta_e(prev_ref, ref)
    if gap <= line["proactive_flush_de"]:
        return None
    return Action("FLUSH_RECOMMENDED", seq, issued_s, deadline_s,
                  f"Changeover {previous}→{recipe}: references ΔE {gap:.1f}; flush (1 water vial) before filling",
                  "ORDER_WATER_VIAL")


def decide(vials, schedule, library, line=None):
    """vials: ordered dicts with seq, fill_end_s, decision_s, endpoint_lab (or None).
    schedule: ordered list of recipe ids, one per vial. Returns (verdicts, actions)."""
    line = {**DEFAULT_LINE, **(line or {})}
    verdicts, actions, streak = [], [], 0
    for vial in vials:
        seq = vial["seq"]
        recipe = schedule[seq - 1] if seq <= len(schedule) else None
        earlier = [r for r in schedule[:seq - 1] if r != recipe]
        previous = earlier[-1] if earlier else None
        lab, ref = vial.get("endpoint_lab"), _reference(library, recipe)
        prev_ref = _reference(library, previous) if previous else None
        v = Verdict(seq, recipe, previous, "CANT_VERIFY", "unmeasured", lab, None, None, None)
        t, deadline = vial["decision_s"], vial["fill_end_s"] + line["scrap_deadline_s"]

        flush = proactive_flush(seq, schedule, library, line, vial.get("arrival_s", t), vial.get("fill_start_s"))
        if flush:
            actions.append(flush)

        if lab is None:
            v.actions.append(Action("HOLD", seq, t, deadline, "No QA-valid colour in the endpoint window", "DIVERT_HOLD"))
            streak = 0
        elif ref is None:
            v.pattern = "no_reference"
            v.actions.append(Action("HOLD", seq, t, deadline, f"No reference colour for recipe {recipe}", "DIVERT_HOLD"))
            streak = 0
        else:
            v.de_intended = delta_e(lab, ref)
            if prev_ref:
                v.de_previous = delta_e(lab, prev_ref)
                v.direction_cos = toward_previous(lab, ref, prev_ref, line.get("direction_space", "lab"))
            if v.de_intended <= line["band_de2000"]:
                v.verdict, v.pattern, streak = "IN_SPEC", "nominal", 0
                v.actions.append(Action("PASS", seq, t, deadline, f"ΔE {v.de_intended:.1f} ≤ {line['band_de2000']}", "ROUTE_GOOD"))
            else:
                v.verdict = "OUT_OF_SPEC"
                v.actions.append(Action("SCRAP", seq, t, deadline, f"ΔE {v.de_intended:.1f} to recipe {recipe}", "DIVERT_SCRAP"))
                if v.direction_cos is not None and v.direction_cos >= line["direction_cos"]:
                    v.pattern, streak = "carryover", 0
                    v.actions.append(Action("FLUSH", seq, t, vial.get("next_fill_start_s"),
                                            f"Colour shifted toward previous recipe {previous} (cos {v.direction_cos:.2f}, "
                                            f"ΔE {v.de_previous:.1f} to it): flush 1 water vial before the next fill",
                                            "ORDER_WATER_VIAL"))
                else:
                    streak += 1
                    v.pattern = "deviation"
                    if streak >= line["persistent_count"]:
                        v.pattern = "persistent_deviation"
                        v.actions.append(Action("STOP_CHECK_DOSING", seq, t, t,
                                                f"{streak} consecutive deviations not explained by carryover: check recipe/dosing",
                                                "LINE_HOLD"))
        verdicts.append(v)
        actions.extend(v.actions)
    return verdicts, actions


def rule_cause(verdict):
    """Rule-based RCA baseline; the LLM/VLM explainers are compared against this."""
    return {
        "nominal": "none", "carryover": "carryover", "deviation": "unknown",
        "persistent_deviation": "wrong_recipe_or_dose", "unmeasured": "unknown", "no_reference": "unknown",
    }[verdict.pattern]


def as_dict(obj):
    return asdict(obj)
