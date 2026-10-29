"""Freeze the Run 5 evaluation settings before Run 5 is scored.

Run 5 is the only labelled recording (30 vials), so its recipe references are Run 5 proxies.
The record fixes, before scoring: the line thresholds (current, or tuned on the synthetic train
families by replay.carryover_synthetic), the recipe library by hash, the labelled vials that
supplied each reference (never scored), and the residue estimator. `replay.line --freeze`
refuses to score with a different library or threshold set.

    python -m replay.eval_freeze --library L --synthetic carryover-eval.json --rule tuned \
        --estimator projection [--estimator-result result.json] --out run5-freeze.json
"""
import argparse
import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from replay.decide import DEFAULT_LINE

SCHEMA = "linewatch-eval-freeze/v1"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reference_vials(library):
    """{recipe id: labelled vial number} from each reference's "from" note ("Run 5 vial 3 endpoint")."""
    out = {}
    for recipe in library["recipes"]:
        match = re.search(r"vial (\d+)", (recipe.get("reference") or {}).get("from", ""))
        if match:
            out[str(recipe["id"])] = int(match.group(1))
    return out


def line_settings(synthetic, rule):
    if rule == "current":
        return dict(DEFAULT_LINE)
    tuned = synthetic and synthetic.get("tuned_on_train")
    if not tuned:
        raise ValueError("--rule tuned needs a synthetic record with tuned_on_train")
    return {**DEFAULT_LINE, "band_de2000": tuned["band_de2000"], "direction_cos": tuned["direction_cos"]}


def make(library_path, rule, estimator, synthetic_path=None, estimator_result=None):
    library = json.loads(Path(library_path).read_text(encoding="utf-8"))
    synthetic = synthetic_path and json.loads(Path(synthetic_path).read_text(encoding="utf-8"))
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                cwd=Path(__file__).parent).stdout.strip() or None
    except OSError:
        commit = None
    synthetic_record = None
    if synthetic:
        dev = synthetic["tuned_on_train"]["dev"] if rule == "tuned" else synthetic["current_rule"]["dev"]
        synthetic_record = {"path": str(synthetic_path), "sha256": sha256(synthetic_path),
                            "dev": {k: dev.get(k) for k in ("changeovers", "detection_rate", "false_alarm_rate", "residue_mae")}}
    return {
        "schema": SCHEMA, "frozen_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "commit": commit, "run5_scored": False,
        "library": {"path": str(library_path), "sha256": sha256(library_path)},
        "rule": rule, "line": line_settings(synthetic, rule),
        "synthetic_record": synthetic_record,
        "estimator": {"name": estimator, "result": estimator_result and
                      {"path": str(estimator_result), "sha256": sha256(estimator_result)}},
        "reference_vials": reference_vials(library),
        "protocol": "Reference vials are reported as references and excluded from verdict counts. Headline: "
                    "first vials after a recipe change. Repeat vials are scored against the frozen references; "
                    "reference sensitivity re-scores first vials against each other repeat vial of the recipe. "
                    "Run 5 is scored once with these settings.",
    }


def check(freeze, library_path):
    """Raise unless the library matches the frozen one; returns the frozen line settings."""
    if freeze.get("schema") != SCHEMA:
        raise ValueError(f"Not a {SCHEMA} record")
    if sha256(library_path) != freeze["library"]["sha256"]:
        raise ValueError(f"{library_path} differs from the frozen library ({freeze['library']['path']})")
    return freeze["line"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--synthetic", type=Path, help="replay.carryover_synthetic output")
    parser.add_argument("--rule", choices=("current", "tuned"), required=True)
    parser.add_argument("--estimator", choices=("projection", "cnn"), required=True)
    parser.add_argument("--estimator-result", type=Path, help="train_carryover_estimator result.json")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("A freeze record is written once; use a new path")
    if args.estimator == "cnn" and not args.estimator_result:
        parser.error("--estimator cnn needs --estimator-result")
    record = make(args.library, args.rule, args.estimator, args.synthetic, args.estimator_result)
    args.out.write_text(json.dumps(record, indent=1), encoding="utf-8")
    print(f"Frozen: band {record['line']['band_de2000']}, cos {record['line']['direction_cos']}, "
          f"estimator {args.estimator}, reference vials {sorted(record['reference_vials'].values())}")


if __name__ == "__main__":
    main()
