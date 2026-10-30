# Live Isaac operator walkthrough

Historical guided experiment. The default launcher now opens the
[continuous inspection station](STATION.md). Pass **`-Walkthrough`** to run the
guided reference/test/recovery interface documented here.

This application runs the existing station in **native Windows Isaac Sim**, sends
each new 640×480 RGB frame through the frozen YOLO11n vial detector, Faster R-CNN
ResNet50-FPN liquid detector, SAM2.1 Hiera Tiny and existing measurement QA, and
waits for the matching inference acknowledgement before the next simulation step.
The operator interface displays those current source-aligned samples. It does not
play a rendered MP4 in place of a live simulator.

## Run locally

From the repository in PowerShell:

```powershell
pwsh -File setup/run-operator-twin.ps1 -Walkthrough
```

Open **http://127.0.0.1:8767**. Allow the model and renderer startup to finish.
The launcher uses the existing `nvidia-pilot-windows` GPU environment under
`AI_CLUB_HOME` or `$HOME/ai-club-tools`, and the installed Isaac launcher at
`C:/Apps/IsaacSim/isaacsim/_build/windows-x86_64/release/python.bat`.
The supplied station assets and the previously pinned local model files are
required. There is no runtime model download, training job, cloud service or new
simulator/driver installation. The recording-only app remains on port 8766.

Each launch requires a **fresh** output directory. The default creates a timestamped
one. Set `-Out 'C:/Users/Austin/ai-club-tools/results/my-operator-twin'` to choose
it. `-Gui` additionally requests the native Isaac viewport; the browser camera
also works with headless rendering. Ctrl+C stops the application. The source
panel also has **Stop this local simulator**, which leaves the web evidence view
available but releases the simulator process.

## Full walkthrough

1. Enter your operator name. Click **Capture clean reference**. A full nominal
   fill runs, generating a phase/meter-indexed development reference from model
   measurements. The recipe is 10 mL from each of three constituents, order 0→1→2.
2. Click **Run test vial**. The declared test starts with a different initial
   residue in an assumed 2 mL mixed shared path; the commanded recipe is unchanged.
   Actual measured color is compared with the earlier rendered reference. Three
   consecutive QA-valid deviations above the existing experimental 5 dE band
   request review. Missing evidence breaks that sequence. If no warning occurs,
   the planned human review is clearly labeled and is not counted as a detection.
3. Inspect the RGB, model mask and measurements, then **Acknowledge review**.
   Acknowledgement leaves the follow-up open. Wait for the complete test fill
   before requesting the intervention.
4. Click **Simulate operator drain / prime**, enter a note and confirm its
   simulation-only scope. The process sends 8 mL of constituent 0 through the
   actual retained compartment to a drain. The journal records a request first
   and a performed-action receipt only after the simulator acknowledges it.
5. Click **Run recovery vial**. A fresh empty vial receives the discharge from
   the same path inventory left after that intervention. Newly rendered frames
   run through the models again. A button does not set the diagnostic to good.
6. Inspect QA availability and comparisons, then **Close simulated follow-up**
   with an outcome note. The earlier review stays latched throughout recovery.
   Export the session or select **Start another walkthrough** to repeat it.

The source and mask views share a source frame identity. The color chart marks
unavailable comparisons as gaps. The episode ledger reports attempted samples,
QA passes, eligible comparisons, maximum dE and observed diagnostic warnings.
There is no claim that a missing warning proves acceptance.

## What the demonstration establishes

It exercises the full software loop: new simulator RGB → actual inference →
provisional diagnostic → human acknowledgement/request → simulated process change
→ new RGB/inference → documented follow-up. Process state is carried from the
completed test through prime-to-drain into recovery. Reference capture and test
initialization are separate declared scenarios, not consecutive production vials.

The public inference frame contains source identity/clock/RGB hash and explicitly
simulated recipe, ingredient phase, meter quantities and completion. Private
constituent inventory and geometric label images are saved separately for the
evaluator. They are not detector inputs, model masks or approved labels.

The process is the existing hypothetical perfectly mixed 2 mL compartment plus
procedural liquid meshes/materials. The carrier remains fixed at the existing
station pose; the drain action is a process-model operation, not a newly animated
valve or calibrated fluid-flow simulation. **The 8 mL action is not an approved
CSI procedure or a recommended physical operating instruction.**

Operational recipe assessment stays **unavailable**: camera/reference commissioning,
reviewed geometry and approved identity/reference evidence are still missing.
Neither a within-band comparison nor closing follow-up authorizes release or
establishes independent product quality. In a physical visit, responses remain
human/operator actions; only this isolated simulator accepts software commands.
No industrial device, network, camera, sensor, actuator, PLC or Teledyne connection
is made. Existing renderer sources, assets, replay defaults and QA are preserved.
Run 5 is unused.

## Artifacts and verification

Each run retains model pins, executed helper snapshots, native runtime/camera and
asset hashes, source RGB, private geometric masks/process history, actual inference
`frames.jsonl`, displayed overlays, operator `operator-events.jsonl` and logs.
The journal attributes manual responses to the typed operator name; it is not
authenticated proof of a physical intervention or a tamper-proof audit system.
A live process restart uses a fresh output; archived runs remain evidence rather
than being silently resumed with an invented simulator state.

```powershell
& "$HOME/ai-club-tools/envs/data-windows/Scripts/python.exe" -m unittest replay.tests.test_operator_twin -v
```

`replay/tests/verify_operator_twin_browser.py` drives the real browser controls
against the running app, captures screenshots/video and checks acknowledgement,
intervention receipt, recovery, closure, export, reload and mobile layout. It uses
the separate Playwright verification environment, not an application dependency.
`replay/tests/verify_operator_twin_run.py` independently joins saved RGB hashes and
acknowledgements, recomputes diagnostics, checks the original schedule/mass balance
and retained-path transition, and verifies source assets. These checks establish
software/provenance behavior, not real defect or human mask accuracy.

## Executed October 21 walkthrough

Local root: `C:/Users/Austin/ai-club-tools/results/operator-isaac-20261021/`.
`live-v1/` completed all three 37-frame fills with **111 source-aligned live
RGB/inference acknowledgements** in native Isaac Sim **6.1.0-rc.26**. The browser
performed acknowledgement, a confirmed simulated intervention, recovery and
follow-up closure through the actual controls.

| Episode | Sample QA passes / attempts | Eligible comparisons | Peak dE | Diagnostic |
| --- | ---: | ---: | ---: | --- |
| Nominal development reference | 26/37 | Reference capture | — | Not scored |
| Different initial residue | 21/37 | 21 | 12.03 | Warning at simulation 1.8 s |
| Recovery after operator-requested prime | 26/37 | 26 | 1.01 | No warning observed |

The final recovery sample remains QA-unavailable; the preceding valid comparisons
do not become an approved completed-vial decision. All 111 operational recipe
assessments remain unavailable. This is one software walkthrough, not a replicated
intervention-efficacy or real defect-accuracy result.

`verification-v2/verification.json` independently verifies source RGB/ACK identities,
diagnostics, all private process trajectories, prime mass balance, actual retained
path continuity and source asset hashes. Four new process/workflow tests and the
nine recording-workspace tests passed. Browser checks passed for current RGB,
all three evidence views, latched review, receipt-before-recovery, closure, reload,
mobile layout and loopback-only requests, with no browser errors.

`browser-v2/` has screenshots, the session export and browser capture.
`media-v1/operator-isaac-walkthrough.mp4` is the full approximately 78-second H.264
application walkthrough. The archived test simulator was then stopped cleanly;
`interactive-v1/` is a separate fresh running session for hands-on use at port 8767.
See [the repository verification record](../../setup/verification/operator-isaac-20261021.json).
