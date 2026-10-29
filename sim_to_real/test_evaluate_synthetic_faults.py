
import csv
import tempfile
import unittest
from pathlib import Path

from evaluate_synthetic_faults import evaluate


class TestSyntheticFaultEvaluator(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)

        self.folder = Path(self.temp_dir.name)
        self.truth_path = self.folder / "truth.csv"
        self.ledger_path = self.folder / "ledger.csv"

    def write_csv(self, path, fieldnames, rows):
        with open(path, "w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def run_evaluation(self, truth, predictions):
        self.write_csv(
            self.truth_path,
            ["seq", "cause"],
            truth
        )
        self.write_csv(
            self.ledger_path,
            ["seq", "rule_cause", "pattern"],
            predictions
        )
        return evaluate(self.truth_path, self.ledger_path)

    def test_perfect_fault_detection(self):
        report = self.run_evaluation(
            [
                {"seq": 1, "cause": "none"},
                {"seq": 2, "cause": "carryover"},
                {"seq": 3, "cause": "dose_error"},
            ],
            [
                {"seq": 1, "rule_cause": "none", "pattern": "nominal"},
                {"seq": 2, "rule_cause": "carryover", "pattern": "carryover"},
                {"seq": 3, "rule_cause": "wrong_recipe_or_dose",
                 "pattern": "persistent_deviation"},
            ]
        )

        detection = report["summary"]["detection"]

        self.assertEqual(detection["TP"], 2)
        self.assertEqual(detection["TN"], 1)
        self.assertEqual(detection["FP"], 0)
        self.assertEqual(detection["FN"], 0)
        self.assertEqual(detection["f1"], 1.0)
        self.assertEqual(report["summary"]["ambiguous_diagnoses"], 1)

    def test_detection_errors_and_abstention(self):
        report = self.run_evaluation(
            [
                {"seq": 1, "cause": "none"},
                {"seq": 2, "cause": "carryover"},
                {"seq": 3, "cause": "wrong_recipe"},
                {"seq": 4, "cause": "short_fill"},
            ],
            [
                {"seq": 1, "rule_cause": "unknown", "pattern": "deviation"},
                {"seq": 2, "rule_cause": "none", "pattern": "nominal"},
                {"seq": 3, "rule_cause": "unknown", "pattern": "deviation"},
                {"seq": 4, "rule_cause": "unknown", "pattern": "no_reference"},
            ]
        )

        detection = report["summary"]["detection"]

        self.assertEqual(detection["TP"], 1)
        self.assertEqual(detection["FP"], 1)
        self.assertEqual(detection["FN"], 1)
        self.assertEqual(detection["abstentions"], 1)
        self.assertAlmostEqual(detection["precision"], 0.5)
        self.assertAlmostEqual(detection["recall"], 0.5)
        self.assertAlmostEqual(detection["f1"], 0.5)

    def test_missing_prediction(self):
        report = self.run_evaluation(
            [
                {"seq": 1, "cause": "none"},
                {"seq": 2, "cause": "carryover"},
            ],
            [
                {"seq": 1, "rule_cause": "none", "pattern": "nominal"},
            ]
        )

        self.assertEqual(report["summary"]["missing_predictions"], 1)
        self.assertEqual(report["summary"]["matched_vials"], 1)
        self.assertEqual(report["vials"][1]["status"], "missing_prediction")


if __name__ == "__main__":
    unittest.main()
