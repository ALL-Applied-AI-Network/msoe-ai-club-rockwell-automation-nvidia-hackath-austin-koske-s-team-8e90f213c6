from copy import deepcopy
import json
from pathlib import Path
import unittest
import tempfile
import subprocess
import sys

import numpy as np
from PIL import Image

from box_annotations import instance_mask, semantic_mask, lens_mm, mask_box, sha256, validate_fixture, write_json, yolo_line
from review_box_batch import review


class AnnotationContractChecks(unittest.TestCase):
    def test_render_preflight_does_not_load_numpy_before_kit(self):
        subprocess.run([sys.executable, "-c", "import sys; from box_annotations import validate_fixture; "
            "import json; from pathlib import Path; validate_fixture(json.loads(Path('box_fixture.json').read_text())); "
            "assert 'numpy' not in sys.modules"], cwd=Path(__file__).parent, check=True)

    def test_camera_optics_respect_usd_stage_units(self):
        np.testing.assert_allclose(lens_mm([0.40993, 0.36, 0.27], 1), [40.993, 36, 27])
        np.testing.assert_allclose(lens_mm([40.993, 36, 27], 0.01), [40.993, 36, 27])

    def test_half_open_pixel_bounds_and_yolo_roundtrip(self):
        mask = np.zeros((480, 640), dtype=bool)
        mask[479, 639] = True
        self.assertEqual(mask_box(mask), [639, 479, 640, 480])
        values = yolo_line("stream", mask_box(mask)).split()
        self.assertEqual(values[0], "2")
        cx, cy, w, h = map(float, values[1:])
        np.testing.assert_allclose([(cx-w/2)*640, (cy-h/2)*480, w*640, h*480], [639, 479, 1, 1], atol=1e-5)
        self.assertIsNone(mask_box(np.zeros((2, 2), bool)))
        with self.assertRaises(ValueError):
            yolo_line("liquid", [-1, 0, 2, 3])

    def test_runtime_ids_never_become_object_identity_or_match_sibling_paths(self):
        data = dict(data=np.array([[4, 8, 10, 0]]), info=dict(idToLabels={"4": "/a/vial", "8": "/a/vial/mesh", "10": "/a/vial_neighbour"}))
        np.testing.assert_array_equal(instance_mask(data, "/a/vial"), [[True, True, False, False]])
        self.assertFalse(instance_mask(data, "/absent").any())
        data["info"]["idToLabels"]["4"] = ""
        with self.assertRaisesRegex(ValueError, "empty prim-path"):
            instance_mask(data, "/absent")

    def test_unique_scene_semantics_do_not_depend_on_numeric_id_or_color(self):
        data = dict(data=np.array([[19, 33, 0, 1]]), info=dict(idToLabels={
            "19": {"class": "liquid__201"}, "33": {"class": "stream__301"}}))
        np.testing.assert_array_equal(semantic_mask(data, "liquid__201"), [[True, False, False, False]])
        self.assertFalse(semantic_mask(data, "vial__101").any())
        data["info"]["idToLabels"]["19"] = {}
        with self.assertRaisesRegex(ValueError, "explicit object tag"):
            semantic_mask(data, "liquid__201")

    def test_fixture_rejects_unsafe_names_and_invalid_parameters(self):
        fixture = json.loads(Path(__file__).with_name("box_fixture.json").read_text(encoding="utf-8"))
        validate_fixture(fixture)
        for key, value in [("id", "../escape"), ("level", -1), ("opacity", 0), ("rgb", [1, 2, 0]), ("stream", "typo")]:
            wrong = deepcopy(fixture)
            wrong["cases"][0][key] = value
            with self.assertRaises(ValueError):
                validate_fixture(wrong)

    def test_review_detects_corrupt_masks_and_yolo_coordinate_exports(self):
        with tempfile.TemporaryDirectory(prefix="isaac-box-contract-") as directory:
            root = Path(directory)
            for name in ["rgb", "overlays", "masks", "labels", "label-passes"]:
                (root/name).mkdir()
            write_json(root/"run.json", dict(complete=True, captured=1, requested=1, run5_used=False,
                training_eligible=False, camera_parameters_unchanged=True, source_assets_unchanged=True))
            rgb = np.zeros((480, 640, 3), np.uint8)
            rgb[::2, ::2] = 255
            for directory_name in ["rgb", "overlays"]:
                Image.fromarray(rgb).save(root/directory_name/"case.png")
            ids = np.zeros((480, 640), np.uint32)
            ids[100:200, 300:350] = 37
            np.save(root/"label-passes/case-vial-labels-ids.npy", ids)
            write_json(root/"label-passes/case-vial-labels-mapping.json", {"37": "/test/vial"})
            mask = (ids == 37).astype(np.uint8)*255
            Image.fromarray(mask).save(root/"masks/case-101.png")
            box = [300, 100, 350, 200]
            annotation = dict(class_name="vial", instance_id=101, prim_path="/test/vial", present=True,
                mask_file="masks/case-101.png", mask_sha256=sha256(root/"masks/case-101.png"), visible_pixels=5000,
                bbox_xyxy=box, projected_geometry_bbox_xyxy=box, label_pass="vial-labels")
            row = dict(sample_id="case", training_eligible=False, image_file="rgb/case.png",
                       image_sha256=sha256(root/"rgb/case.png"), annotations=[annotation])
            (root/"manifest.jsonl").write_text(json.dumps(row)+"\n", encoding="utf-8")
            label_file = root/"labels/case.txt"
            label_file.write_text(yolo_line("vial", box)+"\n", encoding="utf-8")
            self.assertTrue(review(root)["passed"])
            label_file.write_text(yolo_line("vial", [301, 100, 350, 200])+"\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "YOLO export"):
                review(root)
            label_file.write_text(yolo_line("vial", box)+"\n", encoding="utf-8")
            ids[0, 0] = 37  # Native renderer linkage cannot be silently changed.
            np.save(root/"label-passes/case-vial-labels-ids.npy", ids)
            with self.assertRaisesRegex(ValueError, "renderer instance IDs"):
                review(root)


if __name__ == "__main__":
    unittest.main()
