"""Live camera input for the line runner: frames as they arrive, on one clock.

Each camera is read on its own thread. A frame is stamped on arrival with the shared monotonic
clock and placed on one bounded queue; if the consumer falls behind, the oldest frames are
dropped (and counted) rather than delaying the line. Frame indices are on a 30 fps grid of that
clock (index = round(t * FPS)), so the line runner's frame-based timing holds with dropped or
irregular frames.

A camera is an OpenCV device: an index (0, 1, ...) or a device path (/dev/video0). A video file
can stand in for a camera ("file:PATH"); it is played at real time through the same path, which
is how the pipeline is tested without connecting a camera. Frames must be 640x480: they are
never resized, because the models and measurement QA work on original pixels.
"""
import threading
import time
from collections import deque

import cv2

FPS = 30.0
SIZE = (640, 480)


def open_camera(device):
    """cv2.VideoCapture for a device index, device path or "file:PATH"."""
    if isinstance(device, str) and device.startswith("file:"):
        return cv2.VideoCapture(device[5:]), True
    source = int(device) if isinstance(device, int) or str(device).isdigit() else device
    capture = cv2.VideoCapture(source)
    capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, SIZE[0])
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, SIZE[1])
    capture.set(cv2.CAP_PROP_FPS, FPS)
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

    def run(self):
        capture, is_file = open_camera(self.device)
        try:
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
                    raise RuntimeError(f"camera {self.camera} stopped delivering frames")
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
            capture.release()


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
        return {"dropped": self.queue.dropped, **{c.camera: c.frames for c in self.cameras}}
