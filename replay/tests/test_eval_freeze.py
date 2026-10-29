"""Frozen Run 5 evaluation settings and reference-vial exclusion (software fixtures, no real data)."""
import json
import tempfile
import unittest
from pathlib import Path

from replay.decide import decide
from replay.eval_freeze import check, make, reference_vials
from replay.line_metrics import score

LIBRARY = {"recipes": [{"id": "1", "reference": {"lab": [50., 0., 0.], "from": "Run 5 vial 3 endpoint (proxy)"}},
                       {"id": "2", "reference": {"lab": [50., 40., 0.], "from": "Run 5 vial 5 endpoint (proxy)"}},
                       {"id": "3", "reference": None}]}
LABS = [[50., 0., 0.], [51., 0., 0.], [50., 0., 0.],      # recipe 1, vial 3 is its reference
        [50., 37., 0.], [50., 40., 0.], [50., 20., 0.]]    # recipe 2, vial 5 is its reference; vial 6 drifted


def vials():
    return [{"seq": i + 1, "fill_start_frame": 300 * i + 10, "fill_end_frame": 300 * i + 200,
             "fill_start_s": 10. * i, "fill_end_s": 10. * i + 6, "decision_s": 10. * i + 8,
             "endpoint_lab": lab} for i, lab in enumerate(LABS)]


class EvalFreezeTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.library = self.tmp / "library.json"
        self.library.write_text(json.dumps(LIBRARY))
        self.labels = self.tmp / "labels.csv"
        self.labels.write_text("vial,recipe_id,start_frame,end_frame\n" + "".join(
            f"{i + 1},{1 if i < 3 else 2},{300 * i},{300 * i + 210}\n" for i in range(6)))

    def test_reference_vials_from_library_notes(self):
        self.assertEqual(reference_vials(LIBRARY), {"1": 3, "2": 5})

    def test_tuned_rule_takes_synthetic_thresholds(self):
        synthetic = self.tmp / "synthetic.json"
        dev = {"changeovers": 10, "detection_rate": 0.5, "false_alarm_rate": 0.0, "residue_mae": 0.07}
        synthetic.write_text(json.dumps({"current_rule": {"dev": dev},
                                         "tuned_on_train": {"band_de2000": 2.0, "direction_cos": 0.5, "dev": dev}}))
        record = make(self.library, "tuned", "projection", synthetic)
        self.assertEqual((record["line"]["band_de2000"], record["line"]["direction_cos"]), (2.0, 0.5))
        self.assertFalse(record["run5_scored"])
        self.assertEqual(check(record, self.library)["band_de2000"], 2.0)

    def test_changed_library_is_refused(self):
        record = make(self.library, "current", "projection")
        self.library.write_text(json.dumps({**LIBRARY, "edited": True}))
        with self.assertRaisesRegex(ValueError, "differs"):
            check(record, self.library)

    def test_reference_vials_are_not_counted(self):
        verdicts, actions = decide(vials(), ["1", "1", "1", "2", "2", "2"], LIBRARY)
        metrics = score(vials(), verdicts, actions, self.labels, ["1"] * 3 + ["2"] * 3, [3, 5])
        summary = metrics["summary"]
        self.assertEqual(summary["reference_vials_not_scored"], [3, 5])
        counted = sum(n for by in summary["verdicts_by_position_after_change"].values() for n in by.values())
        self.assertEqual(counted, 4)
        # Vial 4 is 1.1 ΔE from the frozen reference (vial 5) but 7.7 from vial 6: its verdict depends
        # on which vial supplied the reference.
        vial4 = next(s for s in metrics["reference_sensitivity"] if s["true_vial"] == 4)
        self.assertFalse(vial4["verdict_stable"])
        vial1 = next(s for s in metrics["reference_sensitivity"] if s["true_vial"] == 1)
        self.assertTrue(vial1["verdict_stable"])


if __name__ == "__main__":
    unittest.main()
