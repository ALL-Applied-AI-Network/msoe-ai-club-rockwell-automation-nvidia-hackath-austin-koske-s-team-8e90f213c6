"""Live camera input for the line runner: frames as they arrive, on one clock.

Each camera is read on its own thread. A frame is stamped on arrival with the shared monotonic
clock and placed on one bounded queue; if the consumer falls behind, the oldest frames are
dropped (and counted) rather than delaying the line. Frame indices are on a 30 fps grid of that
clock (index = round(t * FPS)), so the line runner's frame-based timing holds with dropped or
irregular frames.

A camera is an OpenCV device index (0, 1, ...) or, on Linux, a device path. At CSI the cameras
are named by their stable /dev/v4l/by-id/ links, as in the Linewatch recorder's configs; a link
is resolved to its current /dev/videoN index at every open, because OpenCV's V4L2 backend opens
by index and the kernel renumbers cameras that drop off a shared USB hub. Cameras are opened the
way the recorder opens them (MJPG, 640x480, warm-up frames so auto-exposure settles), so frames
match the recorded runs. A camera that stops delivering is reopened for up to RECONNECT_S; the
missed slots are gaps on the frame grid.

A video file can stand in for a camera ("file:PATH"); it is played at real time through the same
path, which is how the pipeline is tested without connecting a camera. Frames must be 640x480:
they are never resized, because the models and measurement QA work on original pixels.
"""
import os
import re
import sys
import threading
import time
from collections import deque

import cv2

FPS = 30.0
SIZE = (640, 480)
WARMUP_FRAMES = 30       # the recorder's vial_fill setting: auto-exposure/white balance settle
RECONNECT_S = 5.0        # how long a camera that stopped delivering is retried before the session ends

_VIDEO_NODE = re.compile(r"^/dev/video(\d+)$")


def device_index(device):
    """An int index for an index or a /dev path (a by-id link is resolved now); else the device."""
    if isinstance(device, int) or str(device).isdigit():
        return int(device)
    if isinstance(device, str) and device.startswith("/dev/"):
        if not os.path.exists(device):
            raise RuntimeError(f"{device} does not exist: the camera is unplugged or re-enumerated under "
                               f"another name (ls -la /dev/v4l/by-id/)")
        match = _VIDEO_NODE.match(os.path.realpath(device))
        if match is None:
            raise RuntimeError(f"{device} resolves to {os.path.realpath(device)}, not a /dev/videoN node")
        return int(match.group(1))
    return device


def open_camera(device, warmup=WARMUP_FRAMES):
    """(cv2.VideoCapture, is_file) for a device index, device path or "file:PATH". A camera is
    opened MJPG 640x480 and warmed up; it raises if it opens but delivers no frames (a starved
    USB hub) or cannot be opened (unplugged, or held by another program such as the recorder)."""
    if isinstance(device, str) and device.startswith("file:"):
        return cv2.VideoCapture(device[5:]), True
    index = device_index(device)
    capture = cv2.VideoCapture(index, cv2.CAP_V4L2) if sys.platform.startswith("linux") else cv2.VideoCapture(index)
    if not capture.isOpened():
        capture = cv2.VideoCapture(index)
    if not capture.isOpened():
        raise RuntimeError(f"cannot open camera {device}: unplugged, or in use by another program "
                           f"(close the Linewatch recorder first)")
    capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, SIZE[0])
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, SIZE[1])
    capture.set(cv2.CAP_PROP_FPS, FPS)
    if warmup and not any([capture.read()[0] for _ in range(warmup)]):
        capture.release()
        raise RuntimeError(f"camera {device} opened but delivered no frames: the USB hub may lack the "
                           f"bandwidth for every camera (spread them across ports)")
    return capture, False


class FrameQueue:
    """Bounded, drop-oldest queue shared by the camera threads."""

    def __init__(self, maxlen=120):
        self._items = deque()
        self._maxlen = maxlen
        self._cond = threading.Condition()
        self.dropped = 0

    def put(self, item):
        with self._cond:
            if len(self._items) >= self._maxlen:
                self._items.popleft()
                self.dropped += 1
            self._items.append(item)
            self._cond.notify()

    def get(self, timeout):
        with self._cond:
            if not self._items:
                self._cond.wait(timeout)
            return self._items.popleft() if self._items else None


class Camera(threading.Thread):
    def __init__(self, name, device, t0, out, stop, start_s=0.0):
        super().__init__(name=f"camera-{name}", daemon=True)
        self.camera, self.device, self.t0, self.out, self.stop_event = name, device, t0, out, stop
        self.start_s, self.error, self.frames, self.last_index = start_s, None, 0, -1
        self.reconnects = 0

    def run(self):
        capture = None
        try:
            capture, is_file = open_camera(self.device)
            if not capture.isOpened():
                raise RuntimeError(f"cannot open camera {self.camera} ({self.device})")
            if is_file and self.start_s:
                capture.set(cv2.CAP_PROP_POS_FRAMES, int(self.start_s * FPS))
            file_index = int(self.start_s * FPS) if is_file else 0
            while not self.stop_event.is_set():
                ok, bgr = capture.read()
                if not ok:
                    if is_file:
                        break
                    capture.release()
                    capture = self._reopen()
                    if capture is None:
                        break
                    continue
                if (bgr.shape[1], bgr.shape[0]) != SIZE:
                    raise RuntimeError(f"camera {self.camera} delivers {bgr.shape[1]}x{bgr.shape[0]}; "
                                       f"the models need {SIZE[0]}x{SIZE[1]} original frames")
                if is_file:
                    # Play the file at real time: frame k is due k / FPS after the start.
                    due = self.t0 + (file_index - int(self.start_s * FPS)) / FPS
                    time.sleep(max(0.0, due - time.monotonic()))
                    t = self.start_s + (file_index - int(self.start_s * FPS)) / FPS
                    file_index += 1
                else:
                    t = time.monotonic() - self.t0
                index = round(t * FPS)
                if index <= self.last_index:
                    continue        # faster than the 30 fps grid: keep one frame per slot
                self.last_index = index
                self.frames += 1
                self.out.put((t, self.camera, (index, bgr)))
        except Exception as error:  # reported to the session, which ends
            self.error = str(error)
        finally:
            if capture is not None:
                capture.release()

    def _reopen(self):
        """Reopen a camera that stopped delivering (USB drop-out), re-resolving its device path."""
        give_up = time.monotonic() + RECONNECT_S
        while not self.stop_event.is_set():
            try:
                capture, _ = open_camera(self.device, warmup=5)
                self.reconnects += 1
                return capture
            except RuntimeError as error:
                if time.monotonic() > give_up:
                    raise RuntimeError(f"camera {self.camera} stopped delivering frames ({error})")
                time.sleep(0.5)
        return None


class CameraSet:
    """The fill camera (and optional tray camera) as one time-ordered frame source."""

    def __init__(self, fill, tray=None, start_s=0.0):
        self.stop_event = threading.Event()
        self.queue = FrameQueue()
        t0 = time.monotonic()
        self.cameras = [Camera("fill", fill, t0, self.queue, self.stop_event, start_s)]
        if tray is not None and tray != "":
            self.cameras.append(Camera("tray", tray, t0, self.queue, self.stop_event, start_s))

    def __iter__(self):
        for camera in self.cameras:
            camera.start()
        try:
            while not self.stop_event.is_set():
                item = self.queue.get(timeout=0.5)
                if item is not None:
                    yield item
                    continue
                failed = [c.error for c in self.cameras if c.error]
                if failed:
                    raise RuntimeError("; ".join(failed))
                if not any(c.is_alive() for c in self.cameras):
                    return
        finally:
            self.stop()

    def stop(self):
        self.stop_event.set()

    def stats(self):
        return {"dropped": self.queue.dropped, "reconnects": sum(c.reconnects for c in self.cameras),
                **{c.camera: c.frames for c in self.cameras}}
