# Line view (LINE page)

The LINE page shows the fill-station camera, the finished-vial tray camera and an operator panel.
Each vial's verdict, actions and tray arrival appear when the engine produces them. It is the
annotated line-and-tray MP4, drawn by the app over the real videos instead of burned into a file.

![LINE page in live mode at 16×, Run 5](assets/line-view-live.png)

## Modes

| Mode | Clock | What runs | Use |
| --- | --- | --- | --- |
| **Live** | The server's video clock, at 1–16×, with pause | `replay.live` runs on the server and sends each event over a WebSocket when it becomes known. The videos follow the clock | Demos and the CSI session |
| **Review** | The fill video's playhead (scrub freely) | The same engine over the whole recorded run, computed once. The page shows only the events with `t` ≤ playhead | Going back over a run after the fact |

The two modes show the same events. Live sends them as time passes; Review has them all and
reveals them as the playhead reaches each one.

**Live sources:**
- **Recorded model outputs:** the stored per-sample model results (`frames.jsonl`) and tray
  samples are replayed at the video clock. No GPU is needed. You can start part-way through;
  the engine catches up silently first.
- **Video → models now:** the server decodes the recordings and runs YOLO/U-Net on the fill
  frames as they arrive, as it would from a camera. It needs `APP__MODEL_ASSETS` (and the model
  dependencies) and starts at 0. This source is wired, but has **not yet been run with the
  models**. Without the models, its frame decoding, dwell detection and tray scanning match the
  stored Run 5 results (below).

## Live cameras (real time)

Run **Live cameras** (id `live`), source **Cameras (live, models now)**:
- **Capture.** The fill and tray cameras are read as their frames arrive, on one clock
  ([cameras.py](../replay/cameras.py)).
- **Models.** YOLO/U-Net and measurement QA run on the vial under the nozzle every 0.5 s
  while the carrier is stopped.
- **Decisions.** Each vial is decided when the carrier moves it on.
- **Display.** The page shows both live feeds (MJPEG) with the same overlays.
- **Saving.** Each session records both feeds (H.264) and the model samples, and saves itself
  as a run. **Stop** shows a link to open it in Review.

**Measured on the laptop (RTX 4080 SUPER, native Windows Python, 2026-10-28):**
- Model load: 1.7 s, plus a warm-up inference at load.
- Model sample: 24 ms (p90 29 ms), against a 500 ms budget.
- Run 5 played through the camera input at real time for 130 s: 0 frames dropped, at most
  0.08 s behind. Vials 1–3 were decided at 28.0, 74.5 and 121.0 s, with the batch verdicts:
  OUT OF SPEC ΔE 12.48, IN SPEC 2.84, IN SPEC 2.05.

These runs used **simulated cameras**: the Run 5 videos played at real time through the same
camera input. Real webcams have not been connected. The repository rules leave that to the
person at the machine.

**Before relying on it:**
- **Aim the cameras to match the Run 5 framing.** The models read a fixed vial region
  (`[256, 240, 368, 432]` at 640×480), and the tray slot grid is fixed too. The overlays on the
  live feeds show where these are: aim each camera so the vial and the tray slots sit inside
  them. Per-mount calibration (`station.json` offsets) is still deploy work item 2.
- **Cameras must deliver 640×480.** Frames are never resized; a camera that cannot do
  640×480 is refused with an error.
- **Start the live session before the first vial.** Vials are numbered from the start of the
  session, and vial n takes recipe n of the recipe order.
- **Different cameras and lighting from CSI's.** Expect lower accuracy than on Run 5; the
  models were not tuned for them.

**Configure the station** with `APP__LINE_STATION=station.json`:

```json
{
  "title": "CSI line",
  "fill_camera": 0,
  "tray_camera": 1,
  "library": "/path/to/repo/client/public/data/run5/recipe-library.json"
}
```

- A camera is an OpenCV device index, or a device path such as `/dev/video0` on the GB10.
  Leave `tray_camera` out if there is no tray camera.
- The server also needs `APP__MODEL_ASSETS` (the extracted `temporal-cnn-assets-v1`) and
  `APP__DEVICE=cuda:0`.
- Sessions are saved to `APP__LINE_RECORDINGS`; the default is `APP__DATA_DIR/line-recordings`.
- With `imageio-ffmpeg` installed, recordings are H.264 and play in the browser. Without it,
  they are OpenCV `mp4v`.

**On Windows**, `pwsh -File setup/run-line-live.ps1 -Station station.json` starts it:
- it uses the `vision-windows` environment and the extracted model release;
- it installs the four API packages into a side folder, so the environment itself is not
  changed;
- it needs `client/dist` built first (`npm ci && npm run build` in `client/`).

**Where the server must run:** natively on the machine the cameras are plugged into.
- On Windows, OpenCV opens USB webcams directly; WSL and Docker on Windows cannot see them.
  Use a Python environment with torch (CUDA), OpenCV, ultralytics and scikit-image, plus
  `fastapi`, `uvicorn[standard]`, `pydantic-settings` and `imageio-ffmpeg`.
- On the GB10 (Linux), the cameras are `/dev/video*`. The user needs the `video` group.

## Recipe order

The line runner needs the intended recipe of every vial, in order. At CSI we copy each order
from their machine into the app: in the **Monitor** recipe plan (Josiah's ordering panel),
enter the same recipe and tank amounts (T100–T400 mL, dispense order) in the same order.
The LINE page's **Recipe order** picker then uses that inspection run's plan. The run's
`schedule.csv` file is only a fallback.

- **While a run is active, the Monitor saves the plan whenever it changes.** A live line
  session re-reads it every second, so vials ordered mid-run are picked up. The panel shows
  "not in the recipe order yet" for a vial beyond the end of the plan.
- **Matching to the recipe library** ([recipes_order.py](../replay/recipes_order.py)), first
  match wins:
  1. the same id;
  2. the same formula: tank volumes and dispense order;
  3. the same recipe number (`recipe-07` → `7`).
- **No match:** the vial keeps its own id and has no reference, so it is a HOLD
  (`no_reference`) until a reference exists. The plan's "expected colour" is operator intent,
  not a measured reference, and is not used for verdicts.

## What the page shows

- **Fill camera:**
  - the vial region, in cyan while measuring and in the verdict colour for 4 s after a
    decision;
  - a banner for each non-PASS action.
- **Panel:**
  - **Under the nozzle:**
    - the next scheduled vial and its recipe, and the phase;
    - the liquid area so far;
    - the latest QA-valid colour against the recipe reference, with the live ΔE2000.
  - **Last decision:** verdict, pattern, endpoint ΔE and the direction toward the previous
    recipe.
  - **Actions:** the latest six, with deadline countdowns.
  - **Run:** a strip with one cell per scheduled vial.
  - **Finished tray:** in tray, in transit, should have been diverted, missing caps.
- **Tray camera:**
  - each arrived vial boxed in its slot, coloured by our verdict; NO CAP in magenta;
  - a banner for each arrival.
- **Ledger:** the vials decided so far, with tray status.

Nothing the panel shows uses future frames. The phase and the liquid bar come from samples so
far. The vial number under the nozzle is the next schedule row; a vial gets its number only when
it is decided.

## Live versus batch

`replay.live` reuses `FillTracker`, `segment`, `decide`, `ledger_row`, `TrayScanner`,
`CapTracker` and `SlotTracker`. `python -m replay.line` and `python -m replay.tray` now run
through the same trackers. Three things differ, because the future is unknown:

| | Batch (`replay.line` / `replay.tray`) | Live (`replay.live`) |
| --- | --- | --- |
| FLUSH deadline | Start of the next fill | None ("before the next fill") |
| FLUSH_RECOMMENDED | Issued at the arrival of the vial that turns out to be a fill | Issued when the carrier stops, before we know the stop is a fill |
| Tray matching | Order-preserving alignment over the whole run | First-in-first-out within the transit window, as arrivals happen |

## Verification on Run 5 (2026-10-28)

Run 5 is evaluation-only, and nothing was tuned on it here.

- **Batch refactor:** `replay.line --samples` and `replay.tray --samples` rebuild the stored
  outputs byte for byte: `ledger.csv`, `ledger.json`, `actions.jsonl`, `rca/`,
  `reconciliation.json` and `tray-events.jsonl`.
- **Live versus batch, from stored samples:**
  - **Vials:** the same 30, with identical verdicts, patterns and decision times.
  - **Actions:** the same 39. The one difference is vial 7's FLUSH_RECOMMENDED, issued 18 s
    earlier at a non-fill stop.
  - **Tray:** the same 26 arrivals, matched to the same vials, slots and statuses.
  - **Speed:** the whole run takes 0.5 s.
- **Frame path (first 600 s, without the models):** decoding both videos found the same 14
  dwells with identical start/end frames, and the same 9 Good-tray slot arrivals at the same
  times, as the stored results. Decoding and tray scanning ran 26× faster than real time on
  the laptop CPU.
- **Browser** (headless Chromium, built UI served by the API):
  - Live at 16× advanced 158 s of video in 10 s, with both videos following the clock.
  - Pause held the clock.
  - Review at 17:20 showed the 19 vials decided by then.
  - No console errors.
- **Tests:** `replay/tests/test_live.py` and `api/tests/test_line.py` (skipped without OpenCV).

## Configure a run

Point `APP__LINE_RUNS` at a folder of manifests, one JSON file per run. The file name is the
run id. Relative paths resolve against the manifest's folder.

```json
{
  "title": "Run 5 (CSI Dataset 2, 28 min)",
  "fill_video": "/path/to/vial_fill_video_5.mp4",
  "tray_video": "/path/to/cap_seal_video_5.mp4",
  "schedule": "/path/to/repo/replay/examples/run5-schedule.csv",
  "library": "/path/to/repo/client/public/data/run5/recipe-library.json",
  "fill_samples": "../line-runner-20261027/run5-v3/frames.jsonl",
  "tray_samples": "../tray-20261028/run5-v2/tray-samples.jsonl"
}
```

`fill_samples` comes from `python -m replay.line` (its `frames.jsonl`), and `tray_samples` from
`python -m replay.tray`. Without `fill_samples`, only the "Video → models now" source is
offered. Both videos must be on the same clock and play in the browser (H.264).

## Run it

Native (one process serves the API and the built UI):

```bash
cd client && npm ci && npm run build && cd ../api
APP__LINE_RUNS=/path/to/line-runs APP__CLIENT_DIST=../client/dist \
  uvicorn main:app --host 127.0.0.1 --port 8443
# open http://127.0.0.1:8443/line
```

The Python environment needs the API requirements plus OpenCV, numpy and scikit-image (in
`requirements-model.txt`). For development, `npm run dev` in `client/` proxies `/api`, including
the WebSocket, to the API.

For Docker, mount the manifests, videos and sample files into the API container, and set
`APP__LINE_RUNS` to the mounted manifest folder. The paths in the manifests must be container
paths.

## API

| Route | Returns |
| --- | --- |
| `GET /api/v1/line/runs` | Runs with their available sources |
| `GET /api/v1/line/runs/{id}/events?order=RUN` | Every event of the recorded run (Review), optionally with an inspection run's recipe plan |
| `GET /api/v1/line/runs/{id}/video/{fill\|tray}` | The recording, with range requests |
| `GET /api/v1/line/orders` | Inspection runs with a recipe plan, newest first |
| `GET /api/v1/line/preview/{key}/{fill\|tray}` | MJPEG stream of a live camera session's feed |
| `WS /api/v1/line/runs/{id}/live?source=recorded\|video\|camera&speed=N&start=S&order=RUN&simulate=RUN` | `meta`, then `events` and `clock` messages, then `end`. Send `{"speed": N}` or `{"paused": true}` |

## Not yet

- **Real webcams.** Not yet connected; see "Live cameras" above.
- **Per-mount calibration** of the vial region and tray slots (deploy work item 2). Until
  then, cameras are aimed to the overlays.
- **The GB10 kit and Docker image** do not yet include the line modules, camera access or
  `imageio-ffmpeg` (deploy work items 3–4).
- **Operator sign-off.** Decisions on this page are not yet sent to the operator's
  confirm/override flow or stored in the inspection history.
