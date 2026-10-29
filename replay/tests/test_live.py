import unittest

from replay.live import LiveLine, recorded_items, feed
from replay.tray import GOOD_SLOTS

LIBRARY = {"recipes": [{"id": "1", "reference": {"lab": [50.0, 10.0, 10.0]}},
                       {"id": "2", "reference": {"lab": [50.0, -40.0, 30.0]}}]}
FPS = 30.0


def dwell(start_s, lab, length_s=20.0):
    """A carrier stop with a fill that ramps up over the first half and holds."""
    start, end = int(start_s * FPS), int((start_s + length_s) * FPS)
    samples = []
    for k, frame in enumerate(range(start + 23, end, 15)):
        samples.append({"frame": frame, "t": frame / FPS, "presence": 1.0, "vial_boxes": 1,
                        "liquid_px": min(5000, 400 * k), "qa_valid": True, "flags": [], "lab": lab, "rgb": None})
    return {"start_frame": start, "end_frame": end, "reference_frame": start + 8, "samples": samples, "fps": FPS}


def tray_sample(t, occupied):
    slots = [(5.0, 0.0)] * len(GOOD_SLOTS)
    for k in occupied:
        slots[k] = (60.0, 0.6)
    return {"frame": int(t * FPS), "t": t, "good_count": len(occupied), "bad_count": 0,
            "good_caps": [], "bad_caps": [], "arm": False, "slots": slots}


class LiveLineTest(unittest.TestCase):
    def run_line(self):
        dwells = [dwell(0, [50.0, 10.0, 10.0]), dwell(30, [50.0, 10.0, 10.0]), dwell(60, [50.0, 9.0, 11.0])]
        tray = [tray_sample(i / 3, [0] if i / 3 >= 150 else []) for i in range(0, 3 * 200)]
        live = LiveLine(["1", "1", "2"], LIBRARY, fps=FPS)
        events = [live.meta()]
        for _, source, item in recorded_items(dwells, tray):
            events.extend(feed(live, source, item))
        return events

    def test_events_are_causal_and_ordered(self):
        events = self.run_line()
        self.assertEqual([e["t"] for e in events], sorted(e["t"] for e in events))
        vials = [e for e in events if e["type"] == "vial"]
        self.assertEqual([v["vial"]["seq"] for v in vials], [1, 2, 3])
        for v in vials:
            self.assertAlmostEqual(v["t"], v["vial"]["decision_s"], places=2)

    def test_carryover_vial_is_scrapped_and_flush_recommended_at_stop(self):
        events = self.run_line()
        third = next(e["vial"] for e in events if e["type"] == "vial" and e["vial"]["seq"] == 3)
        self.assertEqual((third["verdict"], third["pattern"]), ("OUT_OF_SPEC", "carryover"))
        kinds = [(e["kind"], e["seq"], e["t"]) for e in events if e["type"] == "action"]
        self.assertIn(("FLUSH_RECOMMENDED", 3, 60.0), kinds)
        flush = next(e for e in events if e["type"] == "action" and e["kind"] == "FLUSH")
        self.assertIsNone(flush["deadline_s"])      # the next fill is in the future

    def test_tray_arrival_matches_first_vial_in_order(self):
        events = self.run_line()
        arrival = next(e for e in events if e["type"] == "tray_arrival")
        self.assertEqual((arrival["seq"], arrival["slot"], arrival["status"]), (1, "R1C1", "routed_as_decided"))
        self.assertGreater(arrival["t"], arrival["arrival_s"])   # known only after the slot is stable


if __name__ == "__main__":
    unittest.main()
