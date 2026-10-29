# Full operator workflow demo — October 23

This document describes the development app on `austin-work`. The scoped
October26 `main` handoff publishes the liquid model/example and QA contracts;
it does not include this operator server or establish live deployment.

**October 24 model decision:** [direct temporal U-Net is the leading SAM-free
integration candidate](../../setup/vision/CURRENT-ARCHITECTURE.md). This demo
retains both cached branches for comparison; its SAM option is optional
experimental refinement. No new live adapter or application-default change is
implied by the architecture decision.

## Cap display / metric correction

The initial video and `bundle-v2` colored fixed fixture-aperture coordinates as
if they localized cap tops. That presentation was wrong for leaning vials.
CPU replay of the exact CAD transforms finds273/646 present-cap observations
whose old marker centers lie outside the projected cap bounding box; maximum
anchor displacement is28.71 source pixels. CAD projection is audit geometry,
not a model localization prediction or an optical visibility annotation.

`bundle-v3` removes those colored point overlays. The current app highlights the
selected96×96 fixture crop and displays the raw model input with its aperture
crosshair, alongside the provisional class. Captured reviews retain that crop
rectangle. The model does not detect a cap center. All720 original classifications,
probabilities, warning flags, source images and checkpoints remain unchanged.

There are genuine errors: nominal608/608, missing cap36/38, crooked cap31/38,
empty36/36. Thus98.75% overall accuracy coexists with81.58% crooked-class recall.
The earlier score-only warning diagnostic is also distinct from the UI rule:
the UI excludes predictions of `no_vial` from occupied-cap warnings, so one
crooked target classified as empty is not warned without placement context.
Current displayed-warning misses are0/38 missing and1/38 crooked, rather than
the earlier score-only0/38 each. No operating threshold was changed for this fix.

Full audit and all nine exact-class errors:
`C:/Users/Austin/ai-club-tools/results/cap-overlay-audit-20261023`.
Its updated video is `First-Vial-Corrected-Slot-Display.mp4`; the original demo
is retained as history. Refresh the existing local app to load the corrected UI.
The active server uses`bundle-v3` /`operator-workspace-v2`; the previous workspace
is preserved. This is a display/interpretation correction, not improved weights.

This opt-in app combines completed filling and cap-inspection outputs with a
working local review/response API. It is separate from the existing station
defaults and the mentees' `api/` / `client/` starter. It demonstrates the proposed
composition of the tested models; it does not run new inference or connect a
camera, PLC, sorter or industrial network.

## What runs in the demo

- **Station monitor:** two independently controlled evidence streams. Filling is
  72 exact real Run4 source frames. Final inspection is 12 static Isaac-rendered
  states, with the frozen cap-state model's actual predictions for 60 positions.
- **Filling branch switch:** direct temporal output versus actual completed
  SAM2.1 refinement on the same source frame. The subset has27/72 versus53/72
  sample-QA passes. This is sampling availability, not segmentation accuracy.
- **Raw/overlay/mask views:** source-aligned images, original native filling
  frame/time, pixel counts and original-RGB sample QA. Failed QA stays unavailable
  even if a raw liquid-mask proposal is visible.
- **Cap target selection:** actual nominal/missing/crooked/empty proposals, frozen
  warning scores and source identity. Body condition is always unassessed. Empty
  positions cannot become missed placements without progress context.
- **Operator context:** attributed demonstration recipe/job input. It does not
  qualify a recipe reference, identify a physical vial or approve a product.
- **Review workflow:** capture source evidence → acknowledge → record response
  → close follow-up. The backend refuses closure without the prior steps.
  Pinned reviews retain their own image when playback advances.
- **Persistence/export:** append-only, hash-chained demonstration events; restart
  identity checks; revision checks; idempotent commands; JSON export. Every export
  remains demonstration-only and ineligible as human annotation truth.
- **Pipeline view:** the two vision paths, offline synthetic training and shared
  operator workflow in one rendered architecture screen.

This playback is sampled-frame replay. It does not establish native video speed,
live inference throughput, end-to-end camera latency or real defect accuracy.
The two source streams are neither synchronized nor asserted to show the same
physical vials. Browser playback time is distinct from the source timestamps.

## Full pipeline and current boundaries

**Offline training:** supplied CAD / Isaac rendering → synthetic images and
geometric targets → whole-family train/development/test split → frozen weights.
The vial detector has generic COCO initialization; SAM2.1 has generic pretraining.
The new temporal and cap-state networks use random initialization. New CSI-domain
fitting is synthetic-only; real imagery supplies diagnostic/review evidence.

**Filling camera:** retain native source clock and qualify empty reference →
YOLO11n vial proposal → temporal U-Net visible-pool proposal → optional SAM2.1
Hiera Tiny refinement → unchanged original-image measurement QA → RGB/Lab sample
→ intended-recipe comparison only with qualified reference, identity and actual
completion evidence. Otherwise report unavailable, never a fabricated acceptance.
The previous Faster R-CNN→SAM path remains a development comparator, not a required
stage in the newer temporal-guided SAM candidate.

**Final inspection camera:** qualify fixture alignment/visibility and golden →
global reference-gain correction → fixed target crops → dual-branch cap-state CNN
→ cap/occupancy evidence plus separately attributed expected placement progress.
Body damage is unassessed. One-pixel reference misalignment remains a demonstrated
failure; an automatic alignment-validity detector is not established. Authored
lying S-vial poses remain outside the canonical simulated coverage.

**Shared application:** source-bound observation / qualification → captured review
→ human acknowledgement and response → retained journal/export. A cap observation
does not clear a filling warning; observations from two cameras are not fused to
one physical vial without an attributed identity link. A closed follow-up does
not authorize release. Organizer scope is operator-mediated; there is no direct
line-control output in this demo.

The demo displays frozen results at the model boundary. Recipe commissioning,
qualified real references, live model adapters and approved camera-feed integration
remain work for the actual physical deployment. Srijan owns live model adapters;
Josiah/Hayden own the dashboard/backend. This opt-in implementation is a concrete
reference for their integration, not an overwrite of their work.

## Run locally

Prepared evidence root:
`C:/Users/Austin/ai-club-tools/results/full-app-demo-20261023/bundle-v3`.
Source/frame/model hashes are checked at startup; served image hashes are checked
again when requested. No runtime package installation or GPU is needed to serve
the prepared bundle. Run from the repository root with Python3.11+:

```powershell
python -m replay.operator.full_demo --bundle C:/Users/Austin/ai-club-tools/results/full-app-demo-20261023/bundle-v3 --out C:/Users/Austin/ai-club-tools/results/full-app-demo-20261023/operator-workspace-v2 --port 8870
```

Open `http://127.0.0.1:8870`. It binds only to loopback. Reuse the same workspace
to resume its demonstration history; use a fresh directory for a separate session.
The operator name is attributed input, not authenticated proof of physical action.

Preparation is CPU-only and refuses existing directories:

```text
python setup/vision/prepare_full_app_demo.py --results RESULTS_ROOT --out FRESH_BUNDLE
```

It reuses verified frozen filling/SAM output and cap confirmation artifacts; do
not rerun training for this demo. The earlier partial `bundle-v1` is preserved:
preparation completed its images but failed before its manifest on a path-type
error. `bundle-v2` is the complete verified input. No source was replaced.

## Verification and recording

`replay/tests/test_full_app_demo.py` covers required response order, no product
release, source/slot rejection, restart history and source-tamper rejection.
The headless browser helper `replay/tests/record_full_app_demo.py` records the
actual page/API, both evidence domains, branch switching, pinned review, response,
export, reload persistence and mobile layout. Its automated entries are explicitly
demonstrations, never approved team labels or physical actions.

Local capture and video artifacts are under
`C:/Users/Austin/ai-club-tools/results/full-app-demo-20261023`.
