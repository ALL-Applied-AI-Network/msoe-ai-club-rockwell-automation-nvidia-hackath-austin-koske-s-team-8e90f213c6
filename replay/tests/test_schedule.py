import unittest

from replay.schedule import density, plan, simulate

LIBRARY = {"blue": [32, 18, -58], "navy": [38, 8, -42], "yellow": [88, -4, 78], "green": [55, -48, 32]}
REFS = {k: (density(v), v) for k, v in LIBRARY.items()}


class ScheduleTest(unittest.TestCase):
    def test_same_recipe_has_no_carryover(self):
        cost, steps = simulate([("yellow", 3), ("yellow", 2)], REFS, 0.3)
        self.assertEqual(cost, 0)
        self.assertEqual(steps[1]["flushes"], 0)

    def test_blue_to_yellow_needs_flushes_or_scraps(self):
        _, steps = simulate([("blue", 3), ("yellow", 3)], REFS, 0.3, dict(max_flushes=0))
        self.assertGreater(steps[1]["predicted_scraps"], 0)
        _, steps = simulate([("blue", 3), ("yellow", 3)], REFS, 0.3)
        self.assertGreater(steps[1]["flushes"], 0)

    def test_blue_carryover_moves_blue_recipe_forward(self):
        runs = [("yellow", 3), ("navy", 3)]
        requested, _ = simulate(runs, REFS, 0.3, line=REFS["blue"][0])
        order, (cost, _) = plan(runs, REFS, 0.3, line=REFS["blue"][0])
        self.assertEqual(order[0][0], "navy")
        self.assertLess(cost, requested)

    def test_max_shift_zero_keeps_requested_order(self):
        runs = [("yellow", 2), ("blue", 2), ("green", 2), ("navy", 2)]
        order, _ = plan(runs, REFS, 0.3, max_shift=0)
        self.assertEqual(order, runs)

    def test_free_order_never_costs_more_than_requested(self):
        runs = [("yellow", 2), ("blue", 2), ("green", 2), ("navy", 2)]
        requested, _ = simulate(runs, REFS, 0.3)
        _, (cost, _) = plan(runs, REFS, 0.3)
        self.assertLessEqual(cost, requested)

    def test_lag_model_flush_dilutes_instead_of_clearing(self):
        lag = dict(model="lag", max_flushes=0)
        _, mixed = simulate([("blue", 1), ("yellow", 3)], REFS, 0.9, dict(max_flushes=0))
        _, lagged = simulate([("blue", 1), ("yellow", 3)], REFS, 0.9, lag)
        # mix: residue decays over several vials; lag: only the first vial after the change is off
        self.assertGreater(mixed[1]["predicted_scraps"], lagged[1]["predicted_scraps"])
        self.assertEqual(lagged[1]["predicted_scraps"], 1)
        _, flushed = simulate([("blue", 1), ("yellow", 3)], REFS, 0.9, dict(model="lag"), flushes=[0, 1])
        self.assertEqual(flushed[1]["predicted_scraps"], 1)   # the vial after the water flush is diluted


if __name__ == "__main__":
    unittest.main()
