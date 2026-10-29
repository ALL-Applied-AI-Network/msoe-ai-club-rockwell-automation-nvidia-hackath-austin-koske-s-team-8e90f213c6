"""Stage 4 decision rules on software fixtures (no models, no real data)."""
import unittest

from replay.decide import decide
from replay.line import segment

LIBRARY = {"recipes": [{"id": "A", "reference": {"lab": [55., -35., 45.]}},
                       {"id": "B", "reference": {"lab": [55., 35., -35.]}},
                       {"id": "C", "reference": None}]}


def vial(seq, lab, t=100.):
    t = t * seq
    return {"seq": seq, "arrival_s": t - 30, "fill_start_s": t - 20, "fill_end_s": t,
            "decision_s": t + 8, "next_fill_start_s": t + 80, "endpoint_lab": lab}


def kinds(verdict):
    return [a.kind for a in verdict.actions]


class DecideTest(unittest.TestCase):
    def test_in_spec_passes_before_deadline(self):
        verdicts, actions = decide([vial(1, [55., -34., 44.])], ["A"], LIBRARY)
        self.assertEqual(verdicts[0].verdict, "IN_SPEC")
        self.assertEqual(kinds(verdicts[0]), ["PASS"])
        self.assertLessEqual(actions[0].issued_s, actions[0].deadline_s)

    def test_carryover_scraps_and_flushes_before_next_fill(self):
        verdicts, actions = decide([vial(1, [55., -35., 45.]), vial(2, [55., -30., 40.])], ["A", "B"], LIBRARY)
        self.assertEqual(verdicts[1].pattern, "carryover")
        self.assertEqual(kinds(verdicts[1]), ["SCRAP", "FLUSH"])
        flush = [a for a in actions if a.kind == "FLUSH"][0]
        self.assertLess(flush.issued_s, flush.deadline_s)
        self.assertIn("FLUSH_RECOMMENDED", [a.kind for a in actions])

    def test_persistent_deviation_stops_line(self):
        off = [20., 0., 0.]
        verdicts, _ = decide([vial(i, off) for i in (1, 2, 3)], ["A", "A", "A"], LIBRARY)
        self.assertEqual([v.pattern for v in verdicts], ["deviation", "deviation", "persistent_deviation"])
        self.assertIn("STOP_CHECK_DOSING", kinds(verdicts[2]))

    def test_unmeasured_or_unreferenced_vials_hold(self):
        verdicts, _ = decide([vial(1, None), vial(2, [50., 0., 0.])], ["A", "C"], LIBRARY)
        self.assertEqual([v.verdict for v in verdicts], ["CANT_VERIFY", "CANT_VERIFY"])
        self.assertEqual([v.pattern for v in verdicts], ["unmeasured", "no_reference"])
        self.assertEqual(kinds(verdicts[0]), ["HOLD"])


class SegmentTest(unittest.TestCase):
    def test_fill_window_and_endpoint_from_liquid_growth(self):
        areas = [0, 0, 0, 0, 200, 600, 1000, 1400, 1800, 2000] + [2000] * 14
        samples = [{"frame": 100 + 15 * i, "t": (100 + 15 * i) / 30, "liquid_px": a, "qa_valid": a == 2000,
                    "lab": [50., 1., 2.] if a == 2000 else None} for i, a in enumerate(areas)]
        dwell = {"start_frame": 92, "end_frame": 100 + 15 * len(areas), "reference_frame": 100, "samples": samples}
        result = segment(dwell, 30.)
        self.assertEqual(result["fill_start_frame"], 160)
        self.assertEqual(result["fill_end_frame"], 235)
        self.assertEqual(result["endpoint_lab"], [50., 1., 2.])

    def test_short_or_dry_dwell_is_not_a_vial(self):
        samples = [{"frame": i, "t": i / 30, "liquid_px": 0, "qa_valid": False, "lab": None} for i in range(0, 600, 15)]
        self.assertIsNone(segment({"start_frame": 0, "end_frame": 600, "reference_frame": 8, "samples": samples}, 30.))


if __name__ == "__main__":
    unittest.main()
