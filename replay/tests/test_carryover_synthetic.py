import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from replay.carryover_synthetic import changeovers, measure, summary, sweep

REF, PREV = np.array([200., 60., 60.]), np.array([40., 60., 200.])


def batch(root, cases):
    """A fake render batch: flat liquid colour inside a rectangular liquid mask."""
    for folder in ("rgb", "masks"):
        (root / folder).mkdir(parents=True, exist_ok=True)
    mask = np.zeros((480, 640), np.uint8)
    mask[200:400, 280:360] = 255
    with (root / "manifest.jsonl").open("w") as out:
        for i, (pair, role, split, f, rgb) in enumerate(cases):
            name = f"{pair}-{role[:4]}"
            image = np.full((480, 640, 3), 230, np.uint8)
            image[mask > 0] = np.clip(rgb, 0, 255).astype(np.uint8)
            Image.fromarray(image).save(root / "rgb" / f"{name}.png")
            Image.fromarray(mask).save(root / "masks" / f"{name}-liquid.png")
            out.write(json.dumps(dict(sample_id=name, image_file=f"rgb/{name}.png",
                parameters=dict(pair=pair, role=role, split=split, family=int(pair[1:4]), carryover_fraction=f, level=.9),
                annotations=[dict(class_name="liquid", mask_file=f"masks/{name}-liquid.png", visible_pixels=int((mask > 0).sum()))])) + "\n")


class CarryoverSyntheticTest(unittest.TestCase):
    def test_rule_and_residue(self):
        cases = []
        for k, f in enumerate([0.0, 0.0, 0.1, 0.25]):
            pair = f"f00{k}-c00"
            cases += [(pair, "reference", "train", 0.0, REF), (pair, "previous", "train", 0.0, PREV),
                      (pair, "vial", "train", f, (1 - f) * REF + f * PREV)]
        with tempfile.TemporaryDirectory() as tmp:
            batch(Path(tmp), cases)
            cs = changeovers(measure(Path(tmp)))
        self.assertEqual(len(cs), 4)
        clean = [c for c in cs if c["f"] == 0]
        self.assertTrue(all(c["de_intended"] < 1 and c["residue_estimate"] < 0.02 for c in clean))
        heavy = next(c for c in cs if c["f"] == 0.25)
        self.assertGreater(heavy["direction_cos"], 0.9)
        # Mixing in sRGB is not linear in Lab, so the estimate is approximate.
        self.assertLess(abs(heavy["residue_estimate"] - 0.25), 0.1)
        s = summary(cs, 5.0, 0.8)
        self.assertEqual(s["false_alarm_rate"], 0)
        self.assertEqual(s["detection_rate"], 1)
        self.assertIsNotNone(sweep(cs))

    def test_incomplete_changeover_dropped(self):
        with tempfile.TemporaryDirectory() as tmp:
            batch(Path(tmp), [("f000-c00", "reference", "dev", 0.0, REF), ("f000-c00", "vial", "dev", 0.1, PREV)])
            self.assertEqual(changeovers(measure(Path(tmp))), [])


if __name__ == "__main__":
    unittest.main()
