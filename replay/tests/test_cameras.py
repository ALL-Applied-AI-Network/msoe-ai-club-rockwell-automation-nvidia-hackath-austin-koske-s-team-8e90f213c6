import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

from replay import cameras
from replay.cameras import CameraSet, device_index


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


class FakeCapture:
    """A USB camera that delivers `frames` frames at 30 fps, then drops out."""

    def __init__(self, frames):
        self.left = frames

    def isOpened(self):
        return True

    def read(self):
        time.sleep(1 / 30)
        self.left -= 1
        return (True, np.zeros((480, 640, 3), np.uint8)) if self.left >= 0 else (False, None)

    def release(self):
        pass


class DeviceTest(unittest.TestCase):
    def test_indices_and_by_id_links_resolve_to_the_current_video_node(self):
        self.assertEqual(device_index(1), 1)
        self.assertEqual(device_index("2"), 2)
        link = "/dev/v4l/by-id/usb-4K_USB_CAMERA_HD_USB_CAMERA_01.00.00-video-index0"
        with mock.patch("os.path.exists", return_value=True), \
                mock.patch("os.path.realpath", return_value="/dev/video4"):
            self.assertEqual(device_index(link), 4)
        with mock.patch("os.path.exists", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "re-enumerated"):
                device_index(link)

    def test_a_camera_that_drops_out_is_reopened_and_the_session_goes_on(self):
        opened = [FakeCapture(10), FakeCapture(10)]
        with mock.patch.object(cameras, "open_camera", side_effect=lambda device, warmup=0: (opened.pop(0), False)):
            camera_set = CameraSet(0)
            frames = []
            for item in camera_set:
                frames.append(item)
                if len(frames) == 15:
                    camera_set.stop()
        self.assertEqual(len(frames), 15)
        self.assertEqual(camera_set.stats()["reconnects"], 1)
        indices = [index for _, _, (index, _) in frames]
        self.assertEqual(indices, sorted(set(indices)))     # one frame per grid slot, in time order

    def test_a_camera_that_stays_gone_ends_the_session_with_its_error(self):
        def open_once(device, warmup=0):
            if warmup:
                raise RuntimeError("cannot open camera 0")
            return FakeCapture(3), False
        with mock.patch.object(cameras, "open_camera", side_effect=open_once), \
                mock.patch.object(cameras, "RECONNECT_S", 0.2):
            with self.assertRaisesRegex(RuntimeError, "stopped delivering"):
                list(CameraSet(0))


if __name__ == "__main__":
    unittest.main()
