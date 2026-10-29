# Demo script

A 5–7 minute presenter run-of-show for the operator app, with claims limited to what
was verified on 2026-10-27. Setup and launch are in [the laptop demo](demo.md).
Numbers below are integration and availability counts, not accuracy.

## Main view: the LINE page (live cameras)

The app opens on the LINE page (`/`). On the GB10 at CSI it shows the fill and tray cameras
from `setup/csi-station.json`; the single-vial flow below moved to **MONITOR** (`/monitor`).

1. Close the Linewatch recorder. Start the app (`bash START.sh` from the kit, or
   `bash setup/gb10/onboard.sh`), and check its output: each camera link must resolve to a
   `/dev/videoN`.
2. In **MONITOR**, enter the order from CSI's machine as the recipe plan; on LINE pick it under
   **RECIPE ORDER**.
3. LINE → **Live** → *Cameras (live, models now)* → *Station cameras* → **Start**. Each vial's
   verdict and action appear when the carrier moves it on; the camera stats show dropped
   frames and reconnects.
4. Fallback without cameras: *Cameras* → a recorded run (e.g. Run 5) plays its videos through
   the same camera input in real time.

Live cameras were exercised with recorded Run 5 video played through the camera input, not
yet on the CSI GB10's cameras; run step 1 well before presenting.

## Before presenting (MONITOR flow)

- App open at **http://localhost:8788/monitor**: laptop Docker, or a browser on the GB10 after
  `bash setup/gb10/onboard.sh` (USB backup: `bash START.sh`; see [GB10 deployment](../setup/gb10/README.md)).
  Onboarding must end with *Self-test PASSED*. The header shows
  **SYSTEM ONLINE** and decision badge **MODEL** once a frame is processed.
- The `run5-vial02-recipe5.mp4`, `run5-vial04-recipe6.mp4` and `run5-vial05-recipe6.mp4` clips
  from the release, plus the fallback video, ready locally (see [Fallback](#fallback)).
- Reload the monitor so the vial list starts empty.

## Run-of-show

| Time | Click / show | Say |
| --- | --- | --- |
| 0:00 | Monitor | "Today's sorter checks colour against a static database and isn't told the recipe. We add the missing link: what recipe *should* this vial be, and does it look like it?" |
| 0:30 | **Next vial recipe** → Recipe 5 → **SET RECIPE** | "In production this recipe would come from the line. Here the operator enters it before the vial fills." |
| 0:50 | **RECORDING** → drop `run5-vial02-recipe5.mp4` | "One vial from the organizers' labelled run. Every frame goes through YOLO for the vial, a temporal U-Net for the liquid, and our unchanged QA." |
| 1:30 | Point at overlay and *Latest frame* | "Green is liquid the model proposes; blue is what QA actually samples. When QA isn't confident it says why instead of guessing." |
| 1:50 | Vial result: **IN SPEC · route to GOOD** | "Finished colour is within the band of recipe 5." |
| 2:10 | Recipe 6 → **SET RECIPE** → drop `run5-vial04-recipe6.mp4` | "Now the recipe changes to 6, a pale orange." |
| 2:50 | Vial result: **OUT OF SPEC · divert to SCRAP · possible carryover** | "ΔE 25 from recipe 6, but only about 1.5 from recipe 5. The first vial after the change still looks like the old recipe. Today's sorter would likely pass it." |
| 3:20 | Recipe 6 → drop `run5-vial05-recipe6.mp4` | "Second vial: closer, still shifted toward recipe 5. The line is recovering." |
| 3:50 | Type your name → **CONFIRM** | "The action is simulated: nothing goes to the PLC or sorter. The operator confirms or overrides, and that decision is saved." |
| 4:10 | Optional: **Custom amounts…** | "A new recipe has no reference yet, so the first vial is held; the operator can accept it as the reference for the next ones." |
| 4:40 | **RECIPE CHECK** page | "Across the whole organizer run, all five measurable first-vials-after-change diverged; most second vials recovered." |
| 5:30 | Close | "Operator in the loop today; the same signal can drive a scrap divert once PLC integration is approved." |

## Verified talking points

| Claim | Value | Nature |
| --- | --- | --- |
| Browser rehearsal, Run4 excerpt | 41 sampled frames, 29 with proposals, 14 QA-available | Integration/availability count |
| Direct API pass, same excerpt (40 extracted frames) | 40 / 28 / 14 | Integration/availability count; different seek grid |
| CPU inference, x86 laptop Docker | ~300 ms/frame including HTTP | Not an FPS guarantee |
| GB10 native, CUDA (spark8) | 86 ms/frame warm, 138–145 ms cold, including HTTP, rendering and persistence | Saved Run4 frames, not camera-to-display latency |
| GB10 CUDA vs. x86 CPU, same 40 frames | Proposal 40/40 and QA 40/40 agree; ≤1 mask pixel; max ΔLab 0.59 | Cross-device consistency, not accuracy |
| API tests | 25 passed | Includes explicit mock/protocol fixtures |
| Browser checks | Reset, stale-response, offline, stopped, restart persistence passed | Software behavior only |
| Diagnosis pilot | 1,620 synthetic four-vial sequences, 10 families | Feature-model scenarios; not physical cause accuracy |
| Real completed-vial RCA cases | 0 | The Run4 probe supplied none |
| U-Net training | Random init, 144 synthetic train / 48 synthetic dev images | Real imagery is evaluation-only; Run5 excluded |

## Demonstrated vs. pending

| Demonstrated (2026-10-27) | Pending physical validation |
| --- | --- |
| Released YOLO11n + U-Net + unchanged QA running in the app (model mode) | Live camera on the deployment GB10 |
| Proposal vs. QA-available states with reasons | Independently reviewed empty reference for that camera |
| Attributed, persistent operator responses | Accepted recipe endpoint references |
| Source/rendered/mask artifacts and run review | Segmentation/presence accuracy on human-reviewed labels |
| Run-reset, stale, offline, stopped states | End-to-end latency from native camera PTS to display |
| Synthetic Investigation examples, isolated from runs | CSI station session and workstation |
| Portable Docker kit on x86; GitHub clone → `onboard.sh` → Docker GPU, self-test 40/40 (x86 GPU) | Live camera at the event GB10 (`/dev/video*` or browser-visible feed) |
| GB10 native run on spark8 (aarch64, CUDA and CPU): self-test 40/40, same 41/29/14 browser rehearsal, no page errors | Docker GPU image on aarch64 (untested; onboarding falls back automatically) and event GB10 GitHub access |

## Likely questions

**Why not SAM or Faster R-CNN?** The direct U-Net runs the full CV path at 16 ms median / 25 ms
p95 on GB10, one small model with no prompting stage. SAM and paired DINO produced more QA-eligible samples (185 and 180 vs. 105 of 264); the DINO gains coincided with taller masks that can cover glass. Visual review favored the tighter U-Net pools. That is an engineering selection, not proof of better real accuracy.

**Why train on synthetic images?** Real imagery is reserved for evaluation, so results
aren't tuned on the footage we report. The U-Net starts from random weights. YOLO uses generic COCO pretraining, which we disclose.

**What if the camera view differs?** The model uses a fixed ROI on the released
640×480 profile. The app rejects other geometry rather than resizing, and a new view needs its own camera profile and evaluation.

**Why no pass/fail?** A color sample isn't a recipe decision. That needs an accepted
reference, confirmed fill completion and physical vial identity, none of which we have yet.

**What does the RCA prove?** It generates hypotheses with alternatives. Carryover, dosing and
optical causes can produce identical color traces, so it never confirms a cause.

**How does it deploy on the GB10?** `git clone` and `bash setup/gb10/onboard.sh`. It downloads and
verifies the model release, starts Docker with the GPU (falling back to Docker CPU, then a native
launcher), and proves with a 40-frame self-test that the device reproduces our outputs. A USB kit
with its own Python and hash-locked packages is the backup. We ran the native path and full browser
rehearsal on a GB10 (spark8), and the GitHub onboarding with Docker GPU from a fresh clone on an x86 GPU.

**Is it real-time?** On GB10 CUDA, a saved frame takes about 86 ms through the whole app when sent one at a time.
Camera-to-display latency from native camera timestamps hasn't been measured.

## Fallback

If the live app fails, say so, then play `linewatch-vial-flow-demo.mp4` from the
`linewatch-v1.2.0` release: the same recipe 5 → 6 → 6 sequence recorded on this app.
Still backups: the Recipe Check page screenshots and
`C:/Users/Austin/ai-club-tools/results/run5-recipe-check-20261027/run5-recipe-evidence-board.png`.
A recording is not a physical line validation; say that it is a recorded run of the same app.
