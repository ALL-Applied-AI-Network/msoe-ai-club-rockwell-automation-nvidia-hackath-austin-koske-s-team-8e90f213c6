# Demo script

A 5–7 minute presenter run-of-show for the operator app, with claims limited to what
was verified on 2026-10-27. Setup and launch are in [the laptop demo](demo.md).
Numbers below are integration and availability counts, not accuracy.

## Before presenting

- App open at **http://localhost:8788**: laptop Docker, or a browser on the GB10 after
  `bash setup/gb10/onboard.sh` (USB backup: `bash START.sh`; see [GB10 deployment](../setup/gb10/README.md)).
  Onboarding must end with *Self-test PASSED*. The header shows
  **SYSTEM ONLINE** and decision badge **MODEL** once a frame is processed.
- `run4-demo-excerpt.mp4` and the fallback MP4 ready locally (see [Fallback](#fallback)).
- Start from **NEW RUN** so history from rehearsals is not on screen.

## Run-of-show

| Time | Click / show | Say |
| --- | --- | --- |
| 0:00 | Monitor page | "We watch the vial under the filling nozzle, propose accumulated liquid early, and measure its color only when sampling checks allow it. The operator decides; the software never commands the line." |
| 0:40 | **RECORDING** → set **FRAME INTERVAL SEC** to `0.2` → drop `run4-demo-excerpt.mp4` | "This is a re-encoded eight-second Run4 clip. Each sampled frame goes through YOLO11n vial geometry, a paired temporal U-Net against an earlier reference, then our unchanged measurement QA." |
| 1:30 | Wait for *Recording complete*; toggle **SOURCE** then **POST-PIPELINE** | "SOURCE is the original 640×480 pixels, unresized. POST-PIPELINE adds the boxes and masks." |
| 2:00 | Point at overlay | "Green is the liquid proposal. Blue is the region QA actually sampled. They differ deliberately: the sampling mask excludes walls, glare and the base." |
| 2:20 | Select an early frame in **Recent frames** showing *Measurement unavailable* | "Here the model proposes liquid, but QA withholds color. The reasons are listed, for example `vial_track_not_stable` or `insufficient_final_liquid_pixels`. We show both states instead of hiding the gap." |
| 2:50 | Select a later frame: *Liquid measurement available · review context* | "Here QA passed: the **SAMPLE LAB** tile shows the color measured from raw source pixels. Recipe stays **Unqualified**: the reference is unreviewed and no accepted recipe endpoint exists, so we do not claim agreement or a pass." |
| 3:20 | Response panel: **REVIEWER** = your name, **RESPONSE** = *Reviewed evidence*, **SAVE RESPONSE** | "The response is attributed and persisted. It records evidence review, not product approval." |
| 3:50 | **OPEN FULL FRAME REVIEW →** | "Every event keeps separate source, rendered and mask artifacts plus QA reasons. They survive an API restart." |
| 4:20 | Header **INVESTIGATION** → **THIS RUN** | "Root-cause analysis needs completed-vial records. Frames are never counted as vials automatically, so this run is unavailable: 0 completed vials, with the missing inputs listed." |
| 4:50 | **SYNTHETIC EXAMPLE** → *Transient color recovery* | "This is clearly labeled synthetic. The first vial shifts toward the previous recipe and later ones converge, which supports a carryover hypothesis. Optical and dosing alternatives stay listed. Confirmed cause: not identified." |
| 5:30 | Example → *Delivery record disagrees inside the research color band* | "Color looks normal, but an attributed delivery record disagrees, and that stays visible. Color alone would miss it." |
| 6:00 | Close | "Operator-in-the-loop: evidence, reasons and hypotheses. No PLC, sorter or valve control, and no automatic pass/fail." |

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

If the live app fails, say so, then play
`C:/Users/Austin/ai-club-tools/results/demo-rehearsal-20261027-v1/final-local/linewatch-demo-fallback-final.mp4`
and narrate the same table. The same flow recorded natively on the GB10:
`C:/Users/Austin/ai-club-tools/results/demo-rehearsal-20261027-v1/gb10/linewatch-gb10-demo.mp4`. Still backups:
`C:/Users/Austin/ai-club-tools/results/demo-rehearsal-20261027-v1/app-checks/` (`recording-monitor.png`,
`recording-review.png`, `still-qa.png`, `offline-state.png`) and
`.../investigation-checks/` (`synthetic-transient.png`, `synthetic-dose-record.png`, `run-unavailable.png`).
A recording is not a physical line validation; say that it is a recorded run of the same app.
