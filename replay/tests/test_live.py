import unittest

import numpy as np

from replay.live import LiveLine, recorded_items, feed
from replay.line import FillTracker
from replay.tray import BAD_SLOTS, GOOD_SLOTS, SLOT_STABLE

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


def tray_sample(t, occupied, bad_occupied=()):
    slots = [(5.0, 0.0)] * len(GOOD_SLOTS)
    bad_slots = [(5.0, 0.0)] * len(BAD_SLOTS)
    for k in occupied:
        slots[k] = (60.0, 0.6)
    for k in bad_occupied:
        bad_slots[k] = (60.0, 0.6)
    return {"frame": int(t * FPS), "t": t, "good_count": len(occupied), "bad_count": len(bad_occupied),
            "good_caps": [], "bad_caps": [], "arm": False, "slots": slots, "bad_slots": bad_slots}


class LiveLineTest(unittest.TestCase):
    def run_line(self, arrival_slot=0):
        dwells = [dwell(0, [50.0, 10.0, 10.0]), dwell(30, [50.0, 10.0, 10.0]), dwell(60, [50.0, 9.0, 11.0])]
        tray = [tray_sample(i / 3, [arrival_slot] if i / 3 >= 150 else []) for i in range(0, 3 * 200)]
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

    def test_first_tray_arrival_can_be_in_any_grid_slot(self):
        events = self.run_line(arrival_slot=17)
        arrival = next(e for e in events if e["type"] == "tray_arrival")
        self.assertEqual((arrival["seq"], arrival["slot"], arrival["status"]),
                         (1, "R3C6", "routed_as_decided"))

    def test_bad_tray_uses_its_own_grid_and_matches_scrap(self):
        live = LiveLine(["1"], LIBRARY, fps=FPS)
        live.waiting.append({"seq": 1, "departure_s": 0.0, "decision": "SCRAP",
                             "verdict": "OUT_OF_SPEC", "recipe_id": "1"})
        events = []
        for i in range(6):
            events.extend(live.tray_sample(tray_sample(40 + i / 3, [])))
        for i in range(SLOT_STABLE):
            events.extend(live.tray_sample(tray_sample(50 + i / 3, [], [29])))
        arrival = next(event for event in events if event["type"] == "tray_arrival")
        self.assertEqual((arrival["tray"], arrival["slot"], arrival["status"]),
                         ("bad", "R5C6", "routed_as_decided"))


class FillTrackerPreviewTest(unittest.TestCase):
    class Runtime:
        def new_measurement(self):
            return object()

        def infer(self, rgb, reference, qa, fallback_vial_box=None):
            self.fallback = fallback_vial_box
            result = {"presence": 1.0, "proposal_mask": np.ones((480, 640), dtype=bool),
                      "yolo_vial_boxes": 0, "vial_geometry_source": "station_profile_candidate"}
            measured = {"measurement_valid": True, "quality_flags": [],
                        "lab": [50.0, 0.0, 0.0], "rgb": [128, 128, 128]}
            return result, measured, np.asarray([[278, 75, 358, 420]], dtype=np.float32)

        def render_overlay(self, rgb, result, measured, boxes):
            return np.full_like(rgb, 127)

    def test_fill_tracker_passes_profile_and_exposes_overlay(self):
        runtime = self.Runtime()
        tracker = FillTracker(runtime, FPS, fallback_vial_box=[278, 75, 358, 420])
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        events = []
        for index in range(24):
            events.extend(tracker.push(index, frame))
        sample = next(payload for kind, payload in events if kind == "sample")
        self.assertEqual(runtime.fallback, [278, 75, 358, 420])
        self.assertEqual((sample["vial_boxes"], sample["yolo_vial_boxes"],
                          sample["vial_geometry_source"]),
                         (1, 0, "station_profile_candidate"))
        self.assertIsNotNone(tracker.preview_bgr)


if __name__ == "__main__":
    unittest.main()
