import unittest

from replay.tray import GOOD_SLOTS, SLOT_STABLE, align, slot_events


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


if __name__ == "__main__":
    unittest.main()
