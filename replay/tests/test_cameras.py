import tempfile
import time
import unittest
from pathlib import Path

import cv2
import numpy as np

from replay.cameras import CameraSet


def clip(path, frames, size=(640, 480)):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, size)
    for k in range(frames):
        writer.write(np.full((size[1], size[0], 3), k * 4 % 255, np.uint8))
    writer.release()


class CameraSetTest(unittest.TestCase):
    def setUp(self):
        self.folder = Path(tempfile.mkdtemp())

    def test_file_camera_plays_in_real_time_on_the_frame_grid(self):
        clip(self.folder / "fill.mp4", 30)
        start = time.monotonic()
        frames = [(t, camera, index) for t, camera, (index, _) in CameraSet(f"file:{self.folder / 'fill.mp4'}")]
        elapsed = time.monotonic() - start
        self.assertEqual([index for _, _, index in frames], list(range(30)))
        self.assertEqual({camera for _, camera, _ in frames}, {"fill"})
        self.assertGreater(elapsed, 0.8)            # 30 frames at 30 fps take about a second

    def test_two_cameras_share_one_clock(self):
        clip(self.folder / "fill.mp4", 15)
        clip(self.folder / "tray.mp4", 15)
        frames = list(CameraSet(f"file:{self.folder / 'fill.mp4'}", f"file:{self.folder / 'tray.mp4'}"))
        self.assertEqual(sum(c == "fill" for _, c, _ in frames), 15)
        self.assertEqual(sum(c == "tray" for _, c, _ in frames), 15)

    def test_frames_that_are_not_640x480_are_refused(self):
        clip(self.folder / "small.mp4", 5, size=(320, 240))
        with self.assertRaisesRegex(RuntimeError, "640x480"):
            list(CameraSet(f"file:{self.folder / 'small.mp4'}"))


if __name__ == "__main__":
    unittest.main()
