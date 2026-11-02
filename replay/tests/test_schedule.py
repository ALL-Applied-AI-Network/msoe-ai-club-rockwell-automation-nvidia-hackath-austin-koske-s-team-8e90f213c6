import unittest

from replay.schedule import MAX_FREE_RUNS, SHIFT_FOR_LONG, density, plan, runs_of, simulate, suggest

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


    def test_runs_of_groups_consecutive_vials(self):
        self.assertEqual(runs_of(["a", "a", "b", "a"]), [("a", 2), ("b", 1), ("a", 1)])


class SuggestTest(unittest.TestCase):
    LIB = {"recipes": [{"id": k, "reference": {"lab": v}} for k, v in LIBRARY.items()] + [{"id": "new", "reference": None}]}

    def test_compares_entered_and_proposed(self):
        ids = ["blue"] * 2 + ["yellow"] * 2 + ["navy"] * 2
        out = suggest(ids, self.LIB, 0.3)
        self.assertEqual(out["missing"], [])
        self.assertEqual(len(out["proposed"]["order"]), len(ids))
        self.assertEqual(sorted(out["proposed"]["order"]), sorted(ids))
        self.assertLessEqual(out["proposed"]["cost"], out["entered"]["cost"])
        self.assertLessEqual(out["entered"]["cost"], out["entered_no_flush"]["cost"])
        self.assertEqual(out["entered_no_flush"]["flushes"], 0)

    def test_recipe_without_reference_is_not_planned(self):
        out = suggest(["blue", "new"], self.LIB, 0.3)
        self.assertEqual(out["missing"], ["new"])
        self.assertNotIn("proposed", out)

    def test_long_orders_limit_moves(self):
        ids = [k for _ in range(4) for k in LIBRARY]        # 16 runs
        self.assertGreater(len(runs_of(ids)), MAX_FREE_RUNS)
        out = suggest(ids, self.LIB, 0.3)
        self.assertEqual(out["max_shift"], SHIFT_FOR_LONG)
        self.assertEqual(sorted(out["proposed"]["order"]), sorted(ids))

    def test_replans_from_measured_line_colour(self):
        out = suggest(["yellow", "navy"], self.LIB, 0.3, line_lab=LIBRARY["blue"])
        self.assertEqual(out["proposed"]["order"][0], "navy")


if __name__ == "__main__":
    unittest.main()
