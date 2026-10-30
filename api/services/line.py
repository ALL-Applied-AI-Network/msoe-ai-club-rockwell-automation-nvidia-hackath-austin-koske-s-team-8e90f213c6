"""Line runs for the live line view: run manifests, paced live sessions and review events.

A run is a JSON manifest in APP__LINE_RUNS (one file per run):

    {"title": "Run 5", "fill_video": "...mp4", "tray_video": "...mp4",
     "schedule": "...csv", "library": "...json",
     "fill_samples": "...frames.jsonl", "tray_samples": "...tray-samples.jsonl"}

Live cameras: APP__LINE_STATION names a station JSON (see docs/line-view.md) with the fill
and tray camera devices; it appears as the run "live" with source "camera". The models run on
frames as they arrive; each live session is recorded and saved as a reviewable run in
APP__LINE_RECORDINGS.

Relative paths resolve against the manifest's folder. `schedule` is optional: a session can
instead follow an inspection run's recipe plan (the order copied from CSI's machine), which
is re-read while the session runs so newly ordered vials are picked up. Source "recorded" replays stored model
and tray samples through the live engine at the video clock; source "video" decodes the
recordings and runs the models as the frames arrive, as a camera would deliver them. Both
emit the same events, and nothing is known before its time.
"""
from __future__ import annotations

import json
import secrets
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

CLOCK_S = 0.2           # wall seconds between clock messages
ORDER_S = 1.0           # wall seconds between re-reads of a followed recipe plan
SPEEDS = (0.5, 1, 2, 4, 8, 16, 32)
MAX_SESSIONS = 4


LIVE = "live"


class LineRuns:
    def __init__(self, root: Path | None, assets: Path | None = None, device: str = "cpu",
                 station: Path | None = None, recordings: Path | None = None):
        self.root, self.assets, self.device = root, assets, device
        self.station, self.recordings = station, recordings
        self._runtime = None
        self._runtime_lock = threading.Lock()
        self._review = {}
        self.previews = {}      # preview key -> {camera: latest JPEG bytes}

    def _roots(self):
        return [r for r in (self.recordings, self.root) if r is not None and r.is_dir()]

    def list(self):
        runs = []
        if self.station is not None:
            station = self.manifest(LIVE)
            runs.append({"id": LIVE, "title": station.get("title", "Live cameras"), "sources": ["camera"],
                         "has_tray": station.get("tray_camera") not in (None, ""), "live": True,
                         "models": self.assets is not None})
        for root in self._roots():
            for path in sorted(root.glob("*.json"), reverse=root == self.recordings):
                manifest = self._manifest(path)
                runs.append({"id": path.stem, "title": manifest.get("title", path.stem),
                             "sources": self.sources(manifest), "has_tray": bool(manifest.get("tray_video"))})
        return runs

    def manifest(self, run_id: str):
        if not run_id.replace("-", "").replace("_", "").isalnum():
            raise KeyError(run_id)
        if run_id == LIVE:
            if self.station is None:
                raise KeyError(run_id)
            return self._manifest(self.station)
        for root in self._roots():
            path = root / f"{run_id}.json"
            if path.is_file():
                return self._manifest(path)
        raise KeyError(run_id)

    def _manifest(self, path):
        raw = json.loads(path.read_text())
        keep = ("title", "fill_camera", "tray_camera")
        return {k: (str((path.parent / v).resolve()) if k not in keep and isinstance(v, str) else v)
                for k, v in raw.items()}

    def sources(self, manifest):
        out = []
        if manifest.get("fill_samples"):
            out.append("recorded")
        if self.assets is not None and manifest.get("fill_video"):
            out.append("video")
        return out

    def video(self, run_id: str, camera: str) -> Path:
        path = self.manifest(run_id).get({"fill": "fill_video", "tray": "tray_video"}[camera])
        if not path or not Path(path).is_file():
            raise KeyError(camera)
        return Path(path)

    def library(self, manifest):
        return json.loads(Path(manifest["library"]).read_text())

    def schedule(self, manifest, plan=None):
        """Recipe ids in order: from a recipe plan (list of recipe dicts) if given, else the manifest."""
        from replay.line import load_schedule
        from replay.recipes_order import schedule_from_plan
        if plan is not None:
            return schedule_from_plan(plan, self.library(manifest))
        return load_schedule(manifest["schedule"]) if manifest.get("schedule") else []

    def engine(self, manifest, runtime=None, schedule=None):
        from replay.live import LiveLine
        return LiveLine(self.schedule(manifest) if schedule is None else schedule, self.library(manifest),
                        manifest.get("line"), runtime, fallback_vial_box=manifest.get("fill_vial_box"),
                        tray_slots=manifest.get("tray_slots"),
                        tray_clear_thumbnail=manifest.get("tray_clear_thumbnail"),
                        tray_empty_slot_gray=manifest.get("tray_empty_slot_gray"))

    def review(self, run_id: str, plan=None):
        """Every event of the recorded run, computed by the live engine without pacing."""
        manifest = self.manifest(run_id)
        if not manifest.get("fill_samples"):
            raise ValueError("Review needs stored samples (fill_samples) in the run manifest")
        schedule = self.schedule(manifest, plan)
        stamp = max(Path(manifest[k]).stat().st_mtime for k in ("fill_samples", "library"))
        key = (run_id, stamp, tuple(schedule))
        if key not in self._review:
            live = self.engine(manifest, schedule=schedule)
            items, feed = self.items(manifest, "recorded")
            events = [live.meta()]
            for _, camera, item in items:
                events.extend(feed(live, camera, item))
            events.extend(live.finish())
            self._review = {key: events}
        return self._review[key]

    def runtime(self):
        with self._runtime_lock:
            if self._runtime is None:
                import numpy as np
                from integration.runtime import ModelRuntime
                runtime = ModelRuntime(self.assets, self.device)
                # The first inference is slow (CUDA warm-up); do it now, not on the first vial.
                blank = np.zeros((480, 640, 3), np.uint8)
                runtime.infer(blank, blank, runtime.new_measurement())
                self._runtime = runtime
            return self._runtime

    def items(self, manifest, source, runtime=None):
        """(t, source, item) in time order, and the function that feeds one into the engine."""
        from replay.live import feed, recorded_items, video_frames
        if source == "recorded":
            dwells = _jsonl(manifest["fill_samples"])
            tray = _jsonl(manifest["tray_samples"]) if manifest.get("tray_samples") else []
            return recorded_items(dwells, tray), feed
        if source == "video":
            def feed_frame(live, camera, item):
                return live.fill_frame(*item) if camera == "fill" else live.tray_frame(*item)
            return video_frames(manifest["fill_video"], manifest.get("tray_video")), feed_frame
        raise ValueError(f"unknown source {source}")

    def session(self, run_id, source, clock, send, stop: threading.Event, order=None):
        """Run one live session on this thread, paced by clock (a SessionClock the client can
        pause or speed up). send(message) is called for every message. order(), if given,
        returns the followed recipe plan (list of recipe dicts) and is polled every ORDER_S."""
        manifest = self.manifest(run_id)
        runtime = self.runtime() if source == "video" else None
        live = self.engine(manifest, runtime, self.schedule(manifest, order()) if order else None)
        send(live.meta())
        last_order = time.monotonic()
        items, feed = self.items(manifest, source, runtime)
        start, pending, last_clock = clock.now(), [], 0.0

        def tick(t, lag=0.0):
            nonlocal last_clock
            send({"type": "clock", "t": round(t, 3), "lag_s": round(lag, 2), "speed": clock.speed,
                  "paused": clock.paused})
            last_clock = time.monotonic()

        for t, camera, item in items:
            if stop.is_set():
                return
            events = []
            if order and time.monotonic() - last_order >= ORDER_S:
                last_order = time.monotonic()
                events.extend(live.set_schedule(self.schedule(manifest, order()), t))
            events.extend(feed(live, camera, item))
            if t < start:
                # Catch up silently to the requested start, then pace from there.
                pending.extend(events)
                continue
            if pending:
                send({"type": "events", "events": pending, "catch_up": True})
                pending = []
            while (now := clock.now()) < t:
                if stop.is_set():
                    return
                if time.monotonic() - last_clock >= CLOCK_S:
                    tick(now)
                time.sleep(CLOCK_S if clock.paused else min(CLOCK_S, max(0.002, (t - now) / clock.speed)))
            if events:
                send({"type": "events", "events": events})
            if time.monotonic() - last_clock >= CLOCK_S:
                tick(t, clock.now() - t)
        if pending:
            send({"type": "events", "events": pending, "catch_up": True})
        tail = live.finish()
        if tail:
            send({"type": "events", "events": tail})
        send({"type": "end", "t": clock.now()})


    def camera_session(self, clock, send, stop: threading.Event, order=None, simulate=None, start_s=0.0):
        """A live session on the station cameras (or, with simulate=<run id>, that run's videos
        played at real time through the camera input). Runs the models on frames as they
        arrive, publishes previews, records both feeds and saves the session as a run."""
        from replay.cameras import FPS, CameraSet
        station = self.manifest(LIVE) if self.station is not None else {}
        library_source = station if station.get("library") else self.manifest(simulate) if simulate else station
        if simulate:
            sim = self.manifest(simulate)
            fill, tray = f"file:{sim['fill_video']}", (f"file:{sim['tray_video']}" if sim.get("tray_video") else None)
        else:
            fill, tray = station.get("fill_camera"), station.get("tray_camera")
        if fill in (None, ""):
            raise ValueError("No fill camera configured (APP__LINE_STATION)")
        if self.assets is None:
            raise ValueError("Live cameras need the models: set APP__MODEL_ASSETS")
        runtime = self.runtime()
        schedule = self.schedule(library_source, order()) if order else self.schedule(library_source)
        live = self.engine(library_source, runtime, schedule)
        start = float(start_s) if simulate else 0.0
        cameras = CameraSet(fill, tray, start_s=start)
        key = secrets.token_urlsafe(12)
        self.previews[key] = {}
        run_id = datetime.now().strftime("live-%Y%m%d-%H%M%S")
        folder = (self.recordings / run_id) if self.recordings is not None else None
        writers = {}
        send(live.meta())
        send({"type": "preview", "key": key, "cameras": [c.camera for c in cameras.cameras], "run_id": run_id})
        last_order = last_clock = time.monotonic()
        wall0 = time.monotonic()
        t = start
        try:
            for t, camera, (index, bgr) in cameras:
                if stop.is_set():
                    break
                events = []
                if order and time.monotonic() - last_order >= ORDER_S:
                    last_order = time.monotonic()
                    events.extend(live.set_schedule(self.schedule(library_source, order()), t))
                camera_events = live.fill_frame(index, bgr) if camera == "fill" else live.tray_frame(index, bgr)
                events.extend(camera_events)
                if index % 2 == 0 or camera == "tray":
                    ok, jpeg = _encode(bgr)
                    if ok:
                        self.previews[key][camera] = jpeg
                if camera == "fill" and live.fill.preview_bgr is not None and any(
                        event.get("type") == "sample" for event in camera_events):
                    ok, jpeg = _encode(live.fill.preview_bgr)
                    if ok:
                        self.previews[key]["fill_mask"] = jpeg
                if camera == "tray" and live.tray_preview_bgr is not None:
                    ok, jpeg = _encode(live.tray_preview_bgr)
                    if ok:
                        self.previews[key]["tray_detection"] = jpeg
                if folder is not None:
                    writer = writers.get(camera) or writers.setdefault(camera, _Recorder(folder / f"{camera}.mp4", FPS))
                    writer.write(index - int(start * FPS), bgr)
                if events:
                    send({"type": "events", "events": events})
                if time.monotonic() - last_clock >= CLOCK_S:
                    lag = (time.monotonic() - wall0) + start - t
                    send({"type": "clock", "t": round(t, 3), "lag_s": round(max(0.0, lag), 2), "speed": 1,
                          "paused": False, "cameras": cameras.stats()})
                    last_clock = time.monotonic()
            tail = live.finish()
            if tail:
                send({"type": "events", "events": tail})
        finally:
            cameras.stop()
            for writer in writers.values():
                writer.close()
            self.previews.pop(key, None)
            if folder is not None and live.dwells:
                self._save_live(run_id, folder, live, station, library_source, simulate, start)
                send({"type": "saved", "run_id": run_id})
        send({"type": "end", "t": t})

    def _save_live(self, run_id, folder, live, station, library_source, simulate, start):
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "frames.jsonl").write_text("".join(json.dumps(d) + "\n" for d in live.dwells))
        (folder / "tray-samples.jsonl").write_text("".join(json.dumps(s) + "\n" for s in live.tray_samples))
        (folder / "schedule.csv").write_text("seq,recipe_id\n" + "".join(f"{i},{r}\n" for i, r in enumerate(live.schedule, 1)))
        manifest = {"title": f"{station.get('title', 'Live')} · {run_id[5:]}" + (f" (simulated from {simulate})" if simulate else ""),
                    "fill_samples": f"{run_id}/frames.jsonl", "schedule": f"{run_id}/schedule.csv",
                    "library": library_source["library"], "recorded_from_s": start}
        if station.get("fill_vial_box") is not None:
            manifest["fill_vial_box"] = station["fill_vial_box"]
        if station.get("tray_slots") is not None:
            manifest["tray_slots"] = station["tray_slots"]
        if station.get("tray_clear_thumbnail") is not None:
            manifest["tray_clear_thumbnail"] = station["tray_clear_thumbnail"]
        if station.get("tray_empty_slot_gray") is not None:
            manifest["tray_empty_slot_gray"] = station["tray_empty_slot_gray"]
        if (folder / "fill.mp4").is_file():
            manifest["fill_video"] = f"{run_id}/fill.mp4"
        if live.tray_samples:
            manifest["tray_samples"] = f"{run_id}/tray-samples.jsonl"
            if (folder / "tray.mp4").is_file():
                manifest["tray_video"] = f"{run_id}/tray.mp4"
        (self.recordings / f"{run_id}.json").write_text(json.dumps(manifest, indent=1))


def _encode(bgr):
    import cv2
    ok, buffer = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 75])
    return ok, buffer.tobytes() if ok else b""


class _Recorder:
    """A camera feed written at a constant 30 fps from frame indices on that grid: a dropped
    slot repeats the previous frame, so video time equals line time. H.264 (browser playable)
    through imageio-ffmpeg when installed (CRF 12, the Linewatch recorder's near-lossless
    setting, since vial colour is the measurement), else OpenCV mp4v."""

    def __init__(self, path, fps):
        self.path, self.fps, self.next, self.last = path, fps, 0, None
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            import imageio_ffmpeg
            self._ffmpeg = imageio_ffmpeg.write_frames(str(path), (640, 480), fps=fps, codec="libx264",
                                                       pix_fmt_in="bgr24", quality=None,
                                                       output_params=["-preset", "veryfast", "-crf", "12"])
            self._ffmpeg.send(None)
            self._cv = None
        except ImportError:
            import cv2
            self._ffmpeg = None
            self._cv = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (640, 480))

    def write(self, index, bgr):
        if index < self.next:
            return
        while self.last is not None and self.next < index:
            self._put(self.last)
        self._put(bgr)
        self.last = bgr

    def _put(self, bgr):
        if self._ffmpeg is not None:
            self._ffmpeg.send(bgr.tobytes())
        else:
            self._cv.write(bgr)
        self.next += 1

    def close(self):
        if self._ffmpeg is not None:
            self._ffmpeg.close()
        elif self._cv is not None:
            self._cv.release()


class SessionClock:
    """Video time for a live session: starts at start_s, runs at speed x wall time, can pause."""

    def __init__(self, start_s=0.0, speed=1.0):
        self._lock = threading.Lock()
        self._t, self._wall, self.speed, self.paused = float(start_s), time.monotonic(), float(speed), False

    def _now(self):
        return self._t if self.paused else self._t + (time.monotonic() - self._wall) * self.speed

    def now(self):
        with self._lock:
            return self._now()

    def set(self, speed=None, paused=None):
        with self._lock:
            self._t, self._wall = self._now(), time.monotonic()
            if speed is not None:
                self.speed = float(speed)
            if paused is not None:
                self.paused = bool(paused)


def _jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
