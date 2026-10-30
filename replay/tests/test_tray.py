import unittest

import numpy as np

from replay.tray import (BAD_SLOTS, GOOD_SLOTS, SLOT_STABLE, STABLE, TRAY_SLOTS, CapTracker,
                         TrayScanner, align, arm_over_trays, baseline_inventory, clear_view,
                         profile_slots, slot_events)


def sample(t, occupied, yellow=0.6, arm=False):
    slots = [(5.0, 0.0)] * len(GOOD_SLOTS)
    for k in occupied:
        slots[k] = (60.0, yellow)
    return {"t": t, "frame": int(t * 30), "arm": arm, "slots": slots}


class AlignTest(unittest.TestCase):
    def test_fifo_with_a_vial_that_never_arrives(self):
        departures = [0, 60, 120, 180, 240]
        arrivals = [130, 190, 370]                     # vials 3 and 4 not seen; 5 arrives
        pairs, typical = align(departures, arrivals, video_end_s=1000)
        self.assertEqual(pairs, {0: 0, 1: 1, 4: 2})

    def test_trailing_vials_in_transit_are_free(self):
        pairs, _ = align([0, 60, 120, 900], [130, 190, 250], video_end_s=950)
        self.assertEqual(pairs, {0: 0, 1: 1, 2: 2})


class SlotTest(unittest.TestCase):
    def test_arrival_needs_stable_occupancy_and_ignores_arm(self):
        samples = [sample(i / 3, []) for i in range(10)]
        samples += [sample(4 + i / 3, [0], arm=True) for i in range(30)]          # arm over the slot
        samples += [sample(20 + i / 3, [1]) for i in range(SLOT_STABLE - 1)]      # transient
        samples += [sample(30 + i / 3, [0], yellow=0.0) for i in range(SLOT_STABLE)]
        events, occupied = slot_events(samples)
        arrivals = [e for e in events if e["kind"] == "arrival"]
        self.assertEqual([(e["slot"], e["capped"]) for e in arrivals], [("R1C1", False)])
        self.assertEqual(arrivals[0]["t"], 30)
        self.assertEqual(occupied, 1)

    def test_clear_reference_rejects_wrong_robot_scene(self):
        clear = np.zeros((480, 640, 3), dtype=np.uint8)
        self.assertTrue(clear_view(clear, TRAY_SLOTS, np.zeros((12, 16), dtype=np.uint8)))
        self.assertFalse(clear_view(clear, TRAY_SLOTS, np.full((12, 16), 255, dtype=np.uint8)))

    def test_arm_is_large_change_from_clear_baseline(self):
        baseline = np.zeros((480, 640, 3), dtype=np.uint8)
        vial = baseline.copy()
        vial[150:170, 220:240] = 255
        self.assertFalse(arm_over_trays(vial, baseline))
        arm = baseline.copy()
        arm[150:230, 220:320] = 255
        self.assertTrue(arm_over_trays(arm, baseline))

    def test_profile_reserves_arbitrary_starting_inventory(self):
        frame = np.full((480, 640, 3), 100, dtype=np.uint8)
        x, y = BAD_SLOTS[29]
        frame[y - 5:y + 5, x - 5:x + 5] = 200
        expected = {tray: [100.0] * 30 for tray in TRAY_SLOTS}
        inventory = baseline_inventory(frame, TRAY_SLOTS, expected)
        self.assertEqual([index for index, occupied in enumerate(inventory["bad"]) if occupied], [29])
        self.assertFalse(any(inventory["good"]))

    def test_profile_ignores_uniform_camera_exposure_shift(self):
        frame = np.full((480, 640, 3), 90, dtype=np.uint8)
        expected = {tray: [50.0] * 30 for tray in TRAY_SLOTS}
        inventory = baseline_inventory(frame, TRAY_SLOTS, expected)
        self.assertFalse(any(inventory["good"]))
        self.assertFalse(any(inventory["bad"]))
        self.assertTrue(all(diff == 0.0 for diff, _ in profile_slots(
            frame, GOOD_SLOTS, expected["good"])))

    def test_profile_keeps_local_slot_change_after_exposure_correction(self):
        frame = np.full((480, 640, 3), 90, dtype=np.uint8)
        x, y = GOOD_SLOTS[17]
        frame[y - 5:y + 5, x - 5:x + 5] = 180
        expected = {tray: [50.0] * 30 for tray in TRAY_SLOTS}
        inventory = baseline_inventory(frame, TRAY_SLOTS, expected)
        self.assertEqual([index for index, occupied in enumerate(inventory["good"]) if occupied], [17])
        self.assertFalse(any(inventory["bad"]))

    def test_slot_profile_initializes_without_global_clear_scene(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        scanner = TrayScanner(30, tray_slots=TRAY_SLOTS,
                              clear_thumbnail=np.full((12, 16), 255, dtype=np.uint8),
                              empty_slot_gray={tray: [0.0] * 30 for tray in TRAY_SLOTS})
        sample = scanner.push(0, frame)
        self.assertIsNotNone(sample)
        self.assertTrue(sample["slots_against_empty_profile"])
        self.assertFalse(sample["arm"])

    def test_arrival_can_start_in_any_open_slot_with_existing_inventory(self):
        def arbitrary_slot_sample(t, new_vial=False):
            slots = [(5.0, 0.0)] * len(GOOD_SLOTS)
            slots[0] = (5.0, 0.6)       # capped vial present in the session baseline
            if new_vial:
                slots[17] = (60.0, 0.6)  # first new arrival is R3C6, not R1C1
            return {"t": t, "frame": int(t * 30), "arm": False, "slots": slots}

        samples = [arbitrary_slot_sample(i / 3) for i in range(6)]
        samples += [arbitrary_slot_sample(10 + i / 3, new_vial=True) for i in range(SLOT_STABLE)]
        events, occupied = slot_events(samples)
        arrivals = [event for event in events if event["kind"] == "arrival"]
        self.assertEqual([(event["slot"], event["capped"]) for event in arrivals], [("R3C6", True)])
        self.assertEqual(occupied, 2)     # baseline inventory plus the new vial

    def test_bad_tray_detects_uncapped_vial_in_any_slot(self):
        def bad_sample(t, occupied=False):
            values = [(5.0, 0.0)] * len(BAD_SLOTS)
            if occupied:
                values[29] = (60.0, 0.0)
            return {"t": t, "frame": int(t * 30), "arm": False, "bad_slots": values}

        samples = [bad_sample(i / 3) for i in range(6)]
        samples += [bad_sample(10 + i / 3, occupied=True) for i in range(SLOT_STABLE)]
        events, occupied = slot_events(samples, "bad")
        arrivals = [event for event in events if event["kind"] == "arrival"]
        self.assertEqual([(event["tray"], event["slot"], event["capped"]) for event in arrivals],
                         [("bad", "R5C6", False)])
        self.assertEqual(occupied, 1)


class CapTest(unittest.TestCase):
    def test_initial_bad_tray_caps_are_inventory_not_arrivals(self):
        def cap_sample(t, points):
            return {"t": t, "frame": int(t * 30), "arm": False,
                    "good_count": 0, "good_caps": [],
                    "bad_count": len(points), "bad_caps": points}

        tracker = CapTracker()
        initial = [(500.0, 300.0)]
        events = [event for i in range(STABLE) for event in tracker.push(cap_sample(i / 3, initial))]
        self.assertEqual(events, [])
        after = initial + [(525.0, 300.0)]
        events = [event for i in range(STABLE) for event in tracker.push(cap_sample(10 + i / 3, after))]
        arrivals = [event for event in events if event["kind"] == "arrival"]
        self.assertEqual([(event["tray"], event["tray_count"]) for event in arrivals], [("bad", 2)])


if __name__ == "__main__":
    unittest.main()
