# Continuous inspection station

The main interface is now an operator station receiving **continuous incoming
frames**. It starts observing as soon as its source and models are ready. The
operator monitors the current image, enters recipe/vial context, reviews captured
exceptions and records a human response. There are no scenario, reference-run,
test-vial or recovery buttons on the main screen.

## Start

```powershell
pwsh -File setup/run-operator-twin.ps1
```

Open **http://127.0.0.1:8767**. The default source is a continuous native Isaac
camera producer using the existing Windows GPU environment and supplied station.
It cycles through independent simulated fills automatically, with declared initial
residue changes for development testing. Operator responses do not change this
feed. Source pause/resume is tucked into **Source & processing details** and
affects only the simulated test source.

The default is headless Isaac rendering; `-Gui` additionally requests the native
viewport. Each launch creates a fresh timestamped output directory. Set `-Out`
explicitly to choose one. Stop with Ctrl+C. Existing frozen models, per-frame QA,
camera assets and teammate renderer sources are reused without edits.

## Optional real-time pre-rendered input

Choose **Input mode → Pre-rendered · real time** beside the camera image. Switch
back to **Live Isaac rendering** with the same dropdown. Enter an operator name
first so the change is attributed. Switching preserves the review queue and
operator journal. The live renderer remains the launch default; to start cached:

```powershell
pwsh -File setup/run-operator-twin.ps1 -Source prerender
```

The prepared cache is under
`C:/Users/Austin/ai-club-tools/results/operator-prerender-20261021/cache-v1`.
Use `-PrerenderCache PATH` for another prepared cache. The option is available
when its video and sample hashes validate at launch.

This mode plays **651 actual Isaac-rendered frames at 30 fps**, a looping
21.7-second clip with three independent simulated fills. Only camera input is
cached: the existing frozen CV models run again on samples decoded from that
same encoded video. The producer targets 5 Hz and skips old samples under
backpressure. A short browser measurement observed 90 presented frames in
3.01 seconds while CV processed ten samples. This establishes 30 fps playback,
not 30 fps CV or new model-accuracy evidence.

The playing camera video advances independently of inference. **Live CV · sampled
mask** is open beside the video by default and shows the exact analyzed image,
mask pixel count, source timestamp and lag. On narrow screens it stacks below
the video. Review requests pin that analyzed sample. Masks are never overlaid on a
newer, unanalyzed video frame. Pausing the simulated source freezes playback;
resuming continues its clock. Each replay loop gets a new stream identity.
These fills are fixed input, so operator responses do not alter their outcome.

To rebuild inputs, use fresh output paths at every step. From the repository in
PowerShell, using the existing native Windows Isaac installation and data Python:

```powershell
$root = "$env:USERPROFILE/ai-club-tools/results/operator-prerender-NEW"
& C:/Apps/IsaacSim/isaacsim/_build/windows-x86_64/release/python.bat replay/operator/render_prerender.py --out "$root/render" --no-ros-env
if ($LASTEXITCODE -ne 0) { throw 'Render failed; preserve its output' }
ffmpeg -framerate 30 -i "$root/render/render-rgb/%06d.png" -c:v libx264 -preset fast -crf 18 -pix_fmt yuv420p -movflags +faststart "$root/camera.mp4"
if ($LASTEXITCODE -ne 0) { throw 'Video encoding failed' }
& "$env:USERPROFILE/ai-club-tools/envs/data-windows/Scripts/python.exe" -m replay.operator.prepare_prerender_cache --render "$root/render" --video "$root/camera.mp4" --out "$root/cache"
if ($LASTEXITCODE -ne 0) { throw 'Cache preparation failed' }
pwsh -File setup/run-operator-twin.ps1 -Source prerender -PrerenderCache "$root/cache"
```

`ffmpeg` must be on PATH; the prepared cache already exists on Austin's machine.
Preparation fully decodes the video, validates native timestamps and saves
111 lossless sample PNGs from the encoded video. No model predictions are cached.
See [executed playback verification](../../setup/verification/prerender-station-20261021.json).

### Reading deviations and unavailable samples

A QA-valid reading over the existing **5 dE development threshold** displays
**Outside development band · confirmation pending** immediately. It does not say
within band while waiting for confirmation. The existing three-consecutive-valid-
sample rule still governs confirmed warnings and automatic review creation.
Missing samples, phase changes and failed QA still interrupt confirmation.

After confirmation, **Appearance differs** stays visible for that observation
even if its next frame fails QA. The current measurement remains unavailable on
that frame; the earlier captured alert and mask remain in the review queue.
Specific failure reasons appear in plain language. If there is no usable mask,
the preview says so rather than implying that a valid mask exists.

Polling returns compact display summaries rather than repeatedly sending old
mask arrays. Exact PNGs, full model logs and full JSON exports retain their
original geometry. These display changes do not relax measurement QA or establish
recipe acceptance. See [visibility verification](../../setup/verification/station-visibility-20261021.json).

The original scripted operator experiment remains available explicitly:

```powershell
pwsh -File setup/run-operator-twin.ps1 -Walkthrough
```

Its [historical results](LIVE-TWIN.md) remain separate from continuous-station
verification. The recording-only manual operator app still runs on port 8766.

## Operator work

- **Live image and assessment:** camera pixels, sampling-mask overlay, current QA,
  eligible development comparison and signal freshness. Stale or paused sources
  are unavailable; the last image stays visibly dimmed with its own timestamp.
- **Recipe / vial context:** enter the intended recipe, batch and physical vial
  association. Source-window IDs and image candidates do not confirm physical IDs.
  A new vial record can be created at a manually observed boundary. A known source
  segment or image-derived fill candidate can also create an observation window.
- **Review queue:** a warning or manual request saves its exact source frame,
  clock, measurements and diagnostic. Review it while the current source advances.
  The queue retains that evidence after it falls out of the live-image buffer.
  A manual request pins the clicked sample when the form opens and shows that fixed
  image while notes are entered; submitting later does not substitute a newer frame.
- **Human response:** acknowledge, record an action actually performed with notes,
  then close follow-up. Acknowledgement alone does not close review. Procedure
  entries require an identifier. The response does not command a simulator or
  physical equipment, authorize release or establish independent product quality.
- **Completion:** explicit operator confirmation captures an endpoint from three
  recent stable QA-valid samples on the same track. Failed checks stay unavailable.
  Simulated phase/completion metadata never silently confirms a real fill.
- **Export:** JSON retains source identities, cases, captured evidence metadata,
  measurements and attributed operator history. A browser reload retains the active
  server session; a process restart uses a fresh workspace and preserves old runs.

For Isaac only, the default job explicitly configures the synthetic 10/10/10 mL
recipe. Its previous pinned nominal model measurements are loaded as a rendered
**development** reference. Progress and recipe metadata are explicitly simulated.
The comparison remains diagnostic and operational recipe acceptance remains
unavailable. An external frame source starts with unknown recipe, no phase/meter
telemetry and no approved reference. The interface does not invent those inputs.

## Replace the producer with an approved camera adapter

```text
Isaac producer OR future approved camera/file producer
        -> local frame input -> frozen CV + measurement QA
        -> current observation + retained exception queue
        -> human acknowledgement / response journal
```

Run the same station without launching Isaac:

```powershell
pwsh -File setup/run-operator-twin.ps1 -Source frames -SourceId vial-camera
```

The producer calls `FramePublisher.publish(metadata, png_bytes)` from
`replay.operator.frame_input`, or POSTs to **`/api/frames`**. This is a local
input boundary, not an installed RTSP, camera SDK or industrial-network connector.
No camera has been connected. Once CSI confirms the feed arrangement, its adapter
only needs to publish the image and original source clock/identity. The dashboard
and operator workflow stay the same.

The POST body is a **640×480 PNG** in the existing measurement profile. Do not
silently resize, crop or change colors to satisfy it; a different pose/resolution
needs a qualified camera/measurement profile. Headers:

- `Content-Type: image/png`
- `X-Frame-Token`: producer token read from the run's local `frame-input-token.json`
- `X-Frame-Metadata`: a JSON object matching the following contract

```json
{
  "schema_version": 1,
  "source_id": "vial-camera",
  "stream_id": "unique-source-connection-id",
  "kind": "frames",
  "sequence": 0,
  "pts": 1200000,
  "time_base": "1/1000000",
  "width": 640,
  "height": 480,
  "rgb_sha256": "SHA256_OF_THE_DECODED_RGB_BYTES",
  "segment_id": null,
  "telemetry": null
}
```

`pts * time_base` preserves the supplied capture clock; do not substitute assumed
FPS conversions or receive time. Sequence and source time must increase within a
stream. `sequence` counts published samples independently of camera FPS; source
time remains the supplied native PTS/time base. Use a new stream ID for a new source connection. Gaps reset consecutive
color evidence. `segment_id` is an optional source window, not a physical vial ID;
ordinary camera input can leave it null and use visual/manual vial boundaries.
Unknown context stays unknown.

The producer receives a source-sequence/RGB-hash acknowledgement **after actual
inference**. One in-flight frame bounds processing; concurrent submissions return
429 instead of building an unbounded stale queue. Sample the camera for the current
5 Hz measurement profile and handle backpressure. UI freshness uses receive age;
measurement/source clocks retain native PTS independently.

Producer and operator credentials are separate. The frame token is not returned
to the browser or placed in URLs/log output. The endpoint accepts pixels and
provenance, never arbitrary client claims of approved masks, reference eligibility
or product acceptance. If a future upstream service already runs CV, adapt the
internal processor to its validated outputs; the UI consumes the same station
state. The current public input runs the pinned CV models on raw frames.

No existing sensor, actuator, PLC, Teledyne or industrial network is connected.
Camera geometry, reference eligibility, physical identity and any progress signal
still need commissioning/review before real recipe decisions are available.

## Local evidence and checks

The continuous Isaac browser check used **86 newly rendered frames across three
source windows**, with no operator start/scenario command. Source RGB/ACK hashes,
72 sampling colors, diagnostics, retained exception images and original camera
asset hashes were independently checked. A separate **three-frame raw-PNG producer**
ran through actual inference with Isaac disabled, preserving native PTS and leaving
recipe/progress/reference unknown. That was a local file-input test, not a camera
connection or accuracy study.

```powershell
& "$HOME/ai-club-tools/envs/data-windows/Scripts/python.exe" -m unittest replay.tests.test_inspection_station -v
```

Seven focused checks cover input validation, unknown context, sample gaps, retained
review evidence, stale/duplicate commands and producer/operator credential separation.
`verify_inspection_station_browser.py` checks the real continuously running source,
review actions during ingestion, explicit context, paused evidence, reload and
mobile layout. `verify_inspection_station_run.py` independently recomputes the
saved source/measurement associations. Full captures and reports live under
`results/operator-station-20261021/`; production-style runs keep a bounded live-image
buffer and permanently save review captures, rather than every PNG. Structured
frame/event logs remain local. `--record-frames` enables full diagnostic captures.
