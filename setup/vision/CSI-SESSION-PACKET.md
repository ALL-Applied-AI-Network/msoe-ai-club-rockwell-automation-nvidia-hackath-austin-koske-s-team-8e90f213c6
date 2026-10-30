# CSI visit: first-visible liquid and recipe-aware inspection

**October24 update:** use the [Tuesday October27 visit guide](../../team_handoff/integration/CSI-VISIT-20261027.md)
for the current direct-U-Net architecture, camera/workstation tie-in plan,
90-minute run sheet, tests and field sheets. Time, assigned host and feed remain
unconfirmed. The published CSI host is Windows11 / RTX PRO5000 Blackwell; verify
the actual unit. The older R-CNN/SAM freeze and proposed dates below are historical
preparation, not the selected October27 deployment baseline.

**Updated access boundary:** Austin relayed that there is no direct communication
with existing line sensors/actuators during the hackathon and no Teledyne database
updates. Use an operator interface with attributed recipe/completion input and a
CSI-approved video/file arrangement. Log exports are optional, not a promised
live meter feed. The earlier interface questions below are superseded by the
[operator clarification](../../team_handoff/part2/OPERATOR-CLARIFICATION.md).
Prior distributed visit PDFs remain historical drafts; the updated Part 2 package
contains the clarified scope. Good/scrap destination semantics need confirmation.

Prepared October 21, 2026. Proposed October 26–28, **subject to CSI confirmation**.
Booking, attendees, feed access and recipes are not confirmed. This packet
supersedes the October 16 scheduling draft. It extends the
[validation protocol](CSI-VALIDATION-PLAN.md); no equipment operations are authorized
by this document. The organizer feedback supplied by Austin establishes the
supervision and network-approval requirements; the linked online guide could not
be retrieved during preparation.

## What we want to learn

1. Does the frozen detector/mask pipeline find the first visible pooled liquid,
   including a thin pool at the bottom, without selecting the entire vial?
2. Can we collect the intended recipe, actual per-fluid metering, physical vial
   identity and independently assessed outcomes needed for a real advisory test?
3. If an approved recipe change naturally produces a mismatch, can qualified
   evidence warn before the confirmed operator/routing deadline?

Success for this visit is a complete, usable evidence set and measured system
limitations. A demonstrated mismatch is desirable, but a real defect cannot be
guaranteed or created by relabeling a camera score. A simulator warning is not a
physical-test result.

## Freeze before prospective scoring

Use the existing raw-pixel PyTorch pipeline: YOLO vial proposals, Faster R-CNN
ResNet-50/FPN liquid proposals, localized SAM2.1 Tiny prompts/causal bridge, then
the existing geometry/color QA. Preserve the confidence, NMS and QA settings.
Record the model/config/helper hashes with every run. Keep TensorRT and upstream
camera correction as separate candidates until their differences are reviewed.
Cosmos Transfer is excluded from the training/evaluation data for this visit.

The October 21 real Run 4 correction check is complete: 1,581 samples, 1,533
camera-eligible pairs, 849 QA passes for each arm (six gains and six losses).
There are 25 startup-unavailable and 23 unhealthy-anchor samples. This provides
no aggregate QA gain on that recording; onset and mask accuracy remain unmeasured.

Existing profiles use a specific 640×480 vial station. New camera geometry,
framing or settings require explicit commissioning; resizing alone does not
establish valid ROIs or a trustworthy mask. Run 4 development thresholds are
frozen. Run 5 remains excluded from tuning. Neither real-review labels nor new
physical footage automatically enter training; clarify the synthetic-training
rule with organizers before any such use.

Keep commissioning/reference data separate from prospective validation. Mark
sequence roles before scoring, freeze the reference and thresholds, and preserve
failed or unavailable cases. If changes are needed after viewing a validation
case, label the result development and reserve a fresh independent sequence.

## Before the visit

- Austin: confirm the slot, attendee names, laptop/export arrangement and the
  equipment/operator contact. Send the prepared request only after filling in
  availability; no message has been sent by this preparation work.
- CSI: choose existing recipe pair A/B, define safe conditioning/flush procedures,
  confirm cycle times, available process exports and written acceptance criteria.
- Data team (Aleyna/Zoe/Reagan): agree on visible pooled-liquid conventions and
  return the existing six-clip Run 4 onset/mask reviews if time permits. Preserve
  individual reviews; resolve disagreement explicitly. This visit does not
  reassign their current work.
- Austin: verify the offline pipeline and file-inventory tool on supplied footage;
  bring the frozen local assets, sufficient disk space, charging/storage, and the
  CSV templates. Test recording/export and native timestamps before useful fills.
- Confirm a legal live video export/feed arrangement with CSI. **Default is
  original recording export followed by local offline processing.** The Isaac
  live loop is a simulation test; it does not establish a working plant feed.

## Proposed 90-minute run sheet

| Time | Work | Evidence/decision |
| --- | --- | --- |
| 0–15 min | CSI walkthrough, camera framing/settings, vial identity, process exports and clock landmarks | Short pilot video and log; verify decoded frames/PTS before continuing. Confirm latest actionable deadline and its clock. |
| 15–35 min | Independently verified acceptable A/B examples, including complete dispensing | Target three fills per recipe if cycle time permits: two reference/commissioning, one reserved validation, with roles fixed in advance. CSI decides acceptability independently. |
| 35–65 min | CSI-approved A→B changeover, observing up to three B vials | Record conditioning, physical vial IDs, recipe/phase/actual meters and outcomes. No assumption of defect or baseline inspection failure. |
| 65–75 min | Optional CSI-approved intervention/comparison, only if feasible | Fresh documented comparable initial condition; CSI executes approved flush. Observe up to three subsequent B vials. Otherwise prioritize missing baseline evidence. |
| 75–90 min | Export originals, completeness check, independent outcome sign-off | Video/log hashes, native clock trace, uncertainty/missing fields, inspection/routing evidence if available; confirm what can be shared. |

The upper target is six nominal fills plus up to six B observation vials and
CSI-defined conditioning. CSI must confirm feasibility. For a shorter visit,
prioritize the pilot, acceptable complete trajectories and one changeover;
defer the flush comparison. Successive vials from one condition are correlated,
not independent experimental repeats. One reserved fill per recipe is a small
commissioning check, not a performance validation study.

## What CSI should confirm on site

- How the existing flow meters report **actual** cumulative delivery, their units,
  accuracy/uncertainty, sampling rate, resets and ingredient/phase semantics.
  Preserve requested quantities separately; do not substitute setpoints for meters.
- A reliable physical vial ID across fill, inspection and routing. Visual track IDs
  are not organizer vial IDs. If association is uncertain, keep it unlinked.
- Written intended-recipe acceptance criteria and an independent assessment method.
  Teledyne's recorded recognized-color/weight result is a separate baseline;
  it does not by itself prove intended-recipe correctness or defect truth.
- Approved camera settings and whether exposure/white balance can be recorded or
  held fixed. If permitted, record a stable background/color target without
  changing equipment. A background patch alone is not a certified color reference.
- At least start/end synchronization landmarks visible in video and logged on the
  process clock, with method and uncertainty. Native PTS is not automatically a
  station timestamp; never infer alignment using assumed FPS or file creation time.
- The actual operator hold/review procedure, disposition deadline and clock, plus
  the permitted way to observe it. Our software records advice and evidence only;
  CSI performs all equipment, recipe and flush operations.

## Evidence folder and forms

Use a new session ID, e.g. `csi-20261026-session01` after the date is confirmed.
Preserve originals in `originals/video`, `originals/process`,
`originals/inspection`, `originals/outcomes`, and `originals/interventions`.
Store filled forms in `forms`, inspector outputs in `inventory-v1`, and derived
analysis/review in fresh directories. Keep filenames free of credentials.

The portable ZIP includes blank CSV templates. Empty fields mean unknown;
do not fill blanks with zero or an inferred answer. Use the actual tank IDs and
units, adapting a documented derived copy while preserving the raw station export.

| Form | Purpose |
| --- | --- |
| `session.csv` | Confirmed booking, attendees, operator, camera settings and approved access arrangement. |
| `vials.csv` | Physical vial identity, predeclared role, recipe and conditioning/sequence identity. |
| `process.csv` | Actual native station timestamps/phase/meter quantities, dispensing state and raw-log identity. |
| `clock-anchors.csv` | Video SHA/frame/PTS/time base paired with a process timestamp, evidence, reviewer and uncertainty. |
| `outcomes.csv` | Independent recipe decision, method/criterion/uncertainty, evidence and reviewer. |
| `inspection-routing.csv` | Separate existing inspection result, physical disposition and actionable deadline. |
| `interventions.csv` | Approved procedure, conditioning and request/ack/completion receipts. |
| `issues.csv` | Missing input, dropped video, identity/clock ambiguity, abort or changed configuration. |

After export, run the provided read-only inspector using the existing environment:

```powershell
$python = 'C:/Users/Austin/ai-club-tools/envs/data-windows/Scripts/python.exe'
& $python inspect_csi_capture.py --source 'D:/csi-SESSION/originals' --out 'D:/csi-SESSION/inventory-v1'
if ($LASTEXITCODE -ne 0) { throw 'Capture inspection failed; preserve originals and investigate' }
```

The inspector hashes every file, fully decodes videos and writes exact native
frame/PTS traces. It makes no station-clock associations and supplies no model
accuracy or label approval. Its completion marker means inventory completed;
check video clock usability separately. Shared hardware/network access remains
subject to CSI approval; the inspector requires only local files.

## Metrics to report after the visit

For reviewed early liquid: first-detection delay relative to reviewed first-visible
native PTS, detection before first visibility, missed/censored cases, mask IoU/Dice,
and oversized/whole-vial errors. Score missing model masks as empty on eligible
reviewed-positive frames; report unavailable measurements explicitly. Score false
presence on human-confirmed absent frames separately. The current sampled pipeline
uses 5 Hz; onset resolution is approximately 0.2 s, not native 30 Hz. Report exact
sampling and clock uncertainty. Onset labels and mask conventions are human work.

For Stage 2: total attempted vials, eligible coverage and unavailable reasons,
warned independently unacceptable vials/total independently unacceptable vials,
warned independently acceptable vials/total independently acceptable vials,
warnings before the confirmed routing deadline and lead-time uncertainty. Report
both all-attempt denominators and eligible subsets. Unknown outcomes stay unknown.
Do not claim sensitivity when there are zero independently unacceptable vials.

For interventions: report execution receipts and independent outcomes in observed
subsequent vials. One before/after sequence cannot establish flush efficacy.
Report actual hardware commands from our application as zero.

## Stop or downgrade conditions

CSI can stop equipment work at any time. If video/log export is unavailable,
framing is incompatible, vial identity is ambiguous, clocks cannot be aligned,
or acceptance criteria are missing, capture the limitation and downgrade the
associated metric to unavailable. Continue useful approved recording/workflow
observation. Never substitute synthetic truth or a model color score for missing
physical evidence. Use the final 15 minutes to preserve the evidence already collected.
