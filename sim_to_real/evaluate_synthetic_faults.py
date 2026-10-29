
import csv
import json
import argparse
from pathlib import Path

NORMAL_CAUSES = {"none", "nominal"}
AMBIGUOUS_CAUSES = {
    "wrong_recipe_or_dose": {"wrong_recipe", "dose_error"}
}

# These are the known verdicts from the line runner.
FAULT_VERDICTS = {"carryover", "deviation", "persistent_deviation"}
NONFAULT_VERDICTS = {"nominal"}
ABSTAIN_VERDICTS = {"unmeasured", "no_reference"}


def load_csv(path):
    with open(path, newline="") as file:
        return list(csv.DictReader(file))


def safe_divide(a, b):
    return a / b if b else None


def evaluate(truth_path, ledger_path):
    truth = load_csv(truth_path)
    ledger = load_csv(ledger_path)

    if not truth or not {"seq", "cause"}.issubset(truth[0]):
        raise ValueError("Truth CSV must contain seq and cause")

    if not ledger or not {"seq", "rule_cause", "pattern"}.issubset(ledger[0]):
        raise ValueError(
            "Ledger CSV must contain seq, rule_cause, and verdict"
        )

    predictions = {int(row["seq"]): row for row in ledger}

    results = []
    tp = fp = tn = fn = 0
    abstentions = 0

    for row in truth:
        seq = int(row["seq"])
        actual = row["cause"].strip()
        prediction = predictions.get(seq)

        if prediction is None:
            results.append({
                "seq": seq,
                "actual": actual,
                "predicted": None,
                "status": "missing_prediction",
                "detection": "missing"
            })
            continue

        predicted_cause = prediction["rule_cause"].strip()
        verdict = prediction["pattern"].strip()

        if actual == predicted_cause or (
            actual in NORMAL_CAUSES
            and predicted_cause in NORMAL_CAUSES
        ):
            status = "correct"
        elif predicted_cause == "unknown":
            status = "unknown"
        elif actual in AMBIGUOUS_CAUSES.get(predicted_cause, set()):
            status = "ambiguous"
        else:
            status = "incorrect"

        actual_fault = actual not in NORMAL_CAUSES

        if verdict in FAULT_VERDICTS:
            predicted_fault = True
        elif verdict in NONFAULT_VERDICTS:
            predicted_fault = False
        elif verdict in ABSTAIN_VERDICTS:
            abstentions += 1
            detection = "abstained"
            predicted_fault = None
        else:
            raise ValueError(f"Unexpected verdict: {verdict}")

        if predicted_fault is not None:
            if actual_fault and predicted_fault:
                tp += 1
                detection = "TP"
            elif not actual_fault and predicted_fault:
                fp += 1
                detection = "FP"
            elif not actual_fault and not predicted_fault:
                tn += 1
                detection = "TN"
            else:
                fn += 1
                detection = "FN"

        results.append({
            "seq": seq,
            "actual": actual,
            "predicted": predicted_cause,
            "pattern": verdict,
            "status": status,
            "detection": detection
        })

    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None
        and recall is not None
        and precision + recall > 0
        else None
    )

    summary = {
        "total_truth_vials": len(truth),
        "matched_vials": sum(
            r["status"] != "missing_prediction" for r in results
        ),
        "exact_cause_matches": sum(
            r["status"] == "correct" for r in results
        ),
        "ambiguous_diagnoses": sum(
            r["status"] == "ambiguous" for r in results
        ),
        "unknown_diagnoses": sum(
            r["status"] == "unknown" for r in results
        ),
        "missing_predictions": sum(
            r["status"] == "missing_prediction" for r in results
        ),
        "detection": {
            "TP": tp,
            "FP": fp,
            "TN": tn,
            "FN": fn,
            "abstentions": abstentions,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "evaluated_vials": tp + fp + tn + fn
        }
    }

    return {"summary": summary, "vials": results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--truth", required=True)
    parser.add_argument("--ledger", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    report = evaluate(args.truth, args.ledger)

    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps(report["summary"], indent=2))
