# Live digital-twin camera and inspection pilot

October 21, 2026. Experimental development on `austin-work`.

This page records the initial four-case pilot. The newer [21-case study and
camera-correction replay](TWIN-EXPERIMENTS.md) adds 803 verified live frames,
flow/camera/optics probes and current results. [Cosmos Transfer](COSMOS-TRANSFER-TRIAL.md)
is a prepared data-generation candidate, currently blocked by checkpoint access.

The supplied station can now drive the existing frozen vision models directly
from a running **native Windows Isaac Sim** process. This is local software in
the loop: render one state, send its RGB, wait for the matching inference response,
then advance the process. It does not replay a previously rendered MP4.

## Architecture and boundaries

```text
Recipe / simulated meter schedule
             |
             v
Hypothesized shared-path compartment -> procedural liquid / stream geometry
             |                                      |
       private composition                   Isaac camera RGB
             |                                      |
       evaluator only                     loopback RGB request
                                                    |
                                      YOLO11n vial detector
                                                    |
                                 Faster R-CNN ResNet50-FPN liquid
                                                    |
                                   SAM2.1 Hiera Tiny + existing QA
                                                    |
                                      matching frame acknowledgement
                                                    |
                                       simulator advances one step
```

Isaac and PyTorch run in separate processes/environments on the same RTX 4080
Super. Frames are untouched 640×480 RGB with a simulation timestamp and SHA256.
The observer receives intended recipe, ingredient phase and metered quantities;
it never receives exact composition, condition names or geometric label masks.
Opaque episode IDs prevent the condition name from revealing the expected result.
Transport binds to `127.0.0.1` only, has one in-flight frame and validates shape,
hash, order and identity. Disconnects fail the run rather than skip frames.

The live adapter binds its declared recipe/phase/meter context to the actual
synthetic session and episode, replacing the recording benchmark's placeholder
context. The operational recipe decision remains unavailable: it has no
commissioned camera/reference/identity/geometry approval. A **separate provisional
color diagnostic** compares QA-valid measured Lab with a prior nominal rendered
reference at the same phase and metered progress. Three consecutive deviations
above the existing experimental 5 dE threshold request provisional review. Phase
changes, failed QA and absent references break consecutive evidence. This diagnostic
does not authorize product release or trigger a flush.

## What the process model means

The new adapter reuses `setup/isaac/liquid_geometry.py` and the supplied station
camera/CAD without changing Cameron/Hayden's renderer or any source assets.
The station animation is frozen at the existing frame-686 geometry; filling is
driven by the new experiment's process model and procedural surface updates.
This is not a new robot-motion or CFD simulation.

A perfectly mixed **2 mL shared-path compartment** receives three ingredient
flows. Each step conserves constituent volume; its exponential displacement is
integrated exactly for a constant inlet. The vial contains the discharged mixture,
not the commanded meter quantities substituted as truth. An **8 mL simulated
flush** discharges to a separate drain and changes the residual inventory before
the next matched test. It does not simply set a quality flag to good.
This example uses constituent 0 as the purge/prime fluid; CSI's actual flush
medium and procedure are not established by this model.

The nominal sequence commands 10 mL per constituent in order 0→1→2, at 6 mL/s,
with pauses, plus prefill/settling. Initial path composition, capacity, flush volume,
height-to-volume scale and transmission/absorption are hypotheses. The nominal
rendered reference is not a CSI-approved acceptable mixture. The plan's seed is
an experiment identifier; this first deterministic pilot does not implement a
randomized scenario or appearance sweep.

## Test cases

| Case | Purpose |
| --- | --- |
| Nominal reference | Build an experimental reference from actual model measurements, never private mask pixels |
| Nominal repeat | Check the same nominal sequence without introducing a different initial residue |
| Different prior residue | Start with a different constituent in the same 2 mL path; keep commanded recipe and meter schedule identical |
| Same residue after flush | Repeat the matched initial condition after the predeclared 8 mL simulated operator flush |

The flush is a **scripted operator intervention**, not a model-triggered action.
The paired cases reset to their declared initial conditions for comparison;
they are not successive vials in one uninterrupted physical changeover. A
continuously carried station state and reviewed intervention policy are next work.

The private evaluator checks native geometric masks in an independent opaque
content pass, source RGB hashes, every acknowledgement, mass balance and complete
video decoding. Sampling-mask overlap with these geometric proxies is not human
segmentation accuracy: the final sampling mask is deliberately conservative, and
transparent/refraction appearance does not define an exact visible-liquid label.
First nonzero simulated body volume is likewise not reviewed first-visible onset.

## Run locally

Use the existing Windows GPU environment; no new installs or driver changes:

```powershell
$python = "$HOME/ai-club-tools/envs/nvidia-pilot-windows/Scripts/python.exe"
$out = "$HOME/ai-club-tools/results/live-twin-my-run"
& $python setup/vision/run_live_twin.py --out $out
```

Add `--gui` for a native Isaac window while the loop runs. The complete pilot is
headless; a separate **32-frame window-mode smoke check passed**, using the supplied
FillCam in the native viewport and `--limit-frames 8` per episode. It does not cover
complete dispensing. Each run requires a
fresh output directory. The launcher starts only its own inference/simulator
processes and retains failure logs instead of overwriting partial evidence.

Default native launcher:
`C:/Apps/IsaacSim/isaacsim/_build/windows-x86_64/release/python.bat`.
The executed runtime is **6.1.0-rc.26**, Python 3.12.13, PathTracing, 8 SPP,
two subframes. This differs from the previously pinned GB10 Isaac 6.0.0.1;
the experiment records its actual version instead of claiming GB10 parity.

After completion, independently verify with the existing analysis environment:

```powershell
& "$HOME/ai-club-tools/envs/vision-alternatives-windows/Scripts/python.exe" `
  setup/vision/verify_live_twin.py --run $out --out "$out-verification"
```

`observer/frames.jsonl` contains actual model/QA outputs; `observer/*.mp4` contains
their visualizations. `simulation/private-truth.jsonl` and `geometric-masks/` are
evaluator evidence. Input/helper/dependency hashes and separate completion markers
identify what actually ran. No model is trained or promoted, no Run 5 data is used,
and no industrial device/network/camera is connected.

## Executed results

Canonical local root:
`C:/Users/Austin/ai-club-tools/results/live-twin-20261021/`.
`full-v3/` completed all four sequences, **37 frames each / 148 frames total**.
Every RGB hash and inference acknowledgement passed the independent verifier;
all 148 private geometric masks and source/process associations were checked.
All 148 original visualization frames fully decode. Source assets are unchanged.
The [verification report](../verification/live-twin-20261021.json) pins 350 evidence
files. Helpers used by the complete run remain frozen in its `helpers/` directory;
later input-validation improvements do not rewrite those snapshots.

| Scenario | Sample QA passes / all attempted frames | Provisional comparisons | Maximum measured dE | Provisional warning |
| --- | ---: | ---: | ---: | --- |
| Nominal reference | 25/37 | Reference capture | — | Not scored |
| Nominal repeat | 25/37 | 25 | 0.567 | No |
| Different initial residue | 21/37 | 21 | 12.157 | Yes, at simulation 1.8 s |
| Same initial residue after scripted flush | 25/37 | 25 | 0.644 | No |

There is **one trajectory per condition**, not 24/72 replicated test episodes.
These counts must not be combined with the earlier analytic Stage 2 results.
Attempted-frame denominators include prefill and settling. The incoming stream
starts at 0.6 s, nonzero accumulated body is first sampled at 0.8 s, and the first
QA-valid model sample is at 1.0 s for nominal/flushed cases versus 1.4 s for the
different-residue case. These are simulation/geometry landmarks, not reviewed
first-visible-liquid accuracy. Sampling-mask/geometric-proxy median IoU ranges
from 0.326 to 0.436 across the four cases; it is not a human mask metric.

All **148 operational recipe assessments remain unavailable**. The provisional
color diagnostic is separate, and the flush was predeclared rather than triggered
by the model. No actual product-quality or intervention-efficacy result follows.
The recorded operational reason is `camera_reference_mismatch`: the declared
camera is explicitly uncommissioned. Identity/geometry and reference eligibility
also remain unapproved; this experiment does not weaken those gates.

The **image-only fill-state detector remains `waiting_for_fill` on all 148
frames**, with no image-derived fill identity. The rendered stream does not yet
exercise its real-video lifecycle successfully. Simulator-supplied recipe/phase
telemetry supports the separate provisional comparison; it is not evidence that
the image-only event detector recognizes a complete synthetic fill. Stream
appearance/motion and lifecycle transfer are concrete next work with the renderer
team. Existing stream thresholds and measurement QA were not changed to force a pass.

Median live frame cycle is **0.549 s** (about **1.82 cycles/s**), including
rendering, separate label captures, image saving and inference acknowledgement.
Median inference round trip is **0.115 s**. Simulation advances 0.2 s per cycle;
this is slower than wall-clock real time. It is not comparable to the earlier
89.48→83.22 ms recording benchmark's timing boundary.

Use `media-v2/live-digital-twin-pipeline.mp4` for the **29.6-second H.264 demo**
(148 frames). `live-twin-scorecard.png` plots the measured color comparisons and
sample-QA counts. The video overlays scenario names only after inference; they
were not inputs to the models. `encoding.json` pins the encoded copy and its
source. Earlier `smoke-v1/` and aborted `full-v1/` are retained as diagnostics for
the native label reset problem, not accepted mask evidence. `full-v2/` and its
first media/verification remain a completed preliminary per-frame experiment;
`full-v3/` supersedes it with corrected synthetic session/process provenance.
Small capture differences between fresh simulator launches are retained, not
replaced with the earlier run's more favorable sample counts.

Seven focused tests passed for mass balance, flush state, nonfinite inputs, source
clock/order/hash integrity, private-truth rejection, missing QA/reference behavior
and consecutive-evidence resets. All six new runtime/verification/media helpers
compile. Window-mode evidence is separately retained under `gui-smoke-v2/` and
`gui-verification-v2/`; it is not counted as another complete fill experiment.

## Next validation

1. Review synthetic appearance/visibility with Cameron/Hayden and the data team.
2. Vary appearance and process assumptions independently; keep nominal references
   separate from test cases. Add stale meter, missing-frame and camera-failure cases.
3. Carry path state across successive vials; test an explicit simulated operator
   response and recovery sequence without inventing physical effectiveness.
4. Fit/check optical and process parameters against separate CSI measurements and
   approved reference trajectories. Keep held-out real evidence separate.

Replicator capture follows NVIDIA's blocking
[`orchestrator.step(wait_for_render=True)` interface](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/replicator_tutorials/tutorial_replicator_getting_started.html),
which avoids knowingly matching a previous render to the current state. Runtime
checks still verify capture identity and reject absent/missing native labels.
