"""Line runs for the live line view: run manifests, paced live sessions and review events.

A run is a JSON manifest in APP__LINE_RUNS (one file per run):

    {"title": "Run 5", "fill_video": "...mp4", "tray_video": "...mp4",
     "schedule": "...csv", "library": "...json",
     "fill_samples": "...frames.jsonl", "tray_samples": "...tray-samples.jsonl"}

Relative paths resolve against the manifest's folder. `schedule` is optional: a session can
instead follow an inspection run's recipe plan (the order copied from CSI's machine), which
is re-read while the session runs so newly ordered vials are picked up. Source "recorded" replays stored model
and tray samples through the live engine at the video clock; source "video" decodes the
recordings and runs the models as the frames arrive, as a camera would deliver them. Both
emit the same events, and nothing is known before its time.
"""
from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

CLOCK_S = 0.2           # wall seconds between clock messages
ORDER_S = 1.0           # wall seconds between re-reads of a followed recipe plan
SPEEDS = (0.5, 1, 2, 4, 8, 16, 32)
MAX_SESSIONS = 4


class LineRuns:
    def __init__(self, root: Path | None, assets: Path | None = None, device: str = "cpu"):
        self.root, self.assets, self.device = root, assets, device
        self._runtime = None
        self._runtime_lock = threading.Lock()
        self._review = {}

    def list(self):
        if self.root is None or not self.root.is_dir():
            return []
        runs = []
        for path in sorted(self.root.glob("*.json")):
            manifest = self._manifest(path.stem)
            runs.append({"id": path.stem, "title": manifest.get("title", path.stem),
                         "sources": self.sources(manifest), "has_tray": bool(manifest.get("tray_video"))})
        return runs

    def manifest(self, run_id: str):
        if self.root is None or not run_id.replace("-", "").replace("_", "").isalnum():
            raise KeyError(run_id)
        path = self.root / f"{run_id}.json"
        if not path.is_file():
            raise KeyError(run_id)
        return self._manifest(run_id)

    def _manifest(self, run_id):
        path = self.root / f"{run_id}.json"
        raw = json.loads(path.read_text())
        return {k: (str((path.parent / v).resolve()) if k != "title" and isinstance(v, str) else v)
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
                        manifest.get("line"), runtime)

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
                from integration.runtime import ModelRuntime
                self._runtime = ModelRuntime(self.assets, self.device)
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
