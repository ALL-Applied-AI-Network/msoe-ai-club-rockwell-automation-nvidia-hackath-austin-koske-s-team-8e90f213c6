# Dataset 1 synthetic fixture pilot — October 22, 2026

This starts the previously proposed second-station work. It is separate from
Dataset 2 liquid segmentation and does not change the operator application.
New GPU work runs on GB10. No real data is used for fitting, selection or this
pilot's test, and Run 5 is excluded.

**Completed follow-up:** [frozen RGB inference on real cap_seal](DATASET1-REAL-TRANSFER.md)
now covers16 stills/480 G targets, with geometric crop proposals and an offline
human review packet. This does not change the synthetic training/test results
below; real accuracy still awaits reviewed labels.

## Completed first results

Both randomly initialized models completed 30 epochs on GB10 and classified
**360/360 synthetic test targets correctly**, with **12/12 complete scenes**
having all 30 predictions correct. Each class has 120 test targets. These are
held-out scene families, but all use the same supplied CAD geometry and positions.
They are not real transfer or physical defect-detection results.

| Frozen input arm | Capped /120 | Missing cap /120 | Empty /120 | 30-crop forward median / p95 |
| --- | ---: | ---: | ---: | ---: |
| RGB | 120 | 120 | 120 | 0.302 / 0.339 ms |
| Current + golden RGB | 120 | 120 | 120 | 0.305 / 0.345 ms |

The frozen occupied-target diagnostic thresholds produce zero missing-cap misses
and zero acceptable-target false alarms on 120 test examples of each. Empty
slots are excluded from that diagnostic. Thresholds were selected using synthetic
development only. The forward timings use 200 measured batches after 30 warm-up
batches on GB10, exclude preprocessing/softmax/serialization/network/dashboard,
and are not end-to-end station latency.

There is **no demonstrated advantage for the paired model** on this easy synthetic
fixture. A simpler RGB model is a valid baseline; neither is chosen as a real
deployment winner. Clear separation of three rendered states does not establish
robustness to real glass, a different camera, damaged bodies or tilted caps.

Local `summary-v1/` contains the source-exact single-panel fixture demo, a test
confusion matrix, exact metrics and 30 attributed experimental observations.
The video shows 12 independent synthetic test scenes at one scene per second;
it is not a live placement sequence. `gb10-results-v1/` retains the source RGB,
executed helpers, frozen models, logs and full evidence.

## Bounded task

The [organizer task](https://msoe-maic.com/library?article=Rockwell-NVIDIA-Fall-2026-Vial-Inspection)
defines predictions for an explicit fixture position. This first pilot covers
three geometry states in the supplied G fixture:

- `pass`: the original capped-vial geometry is present; this is a synthetic
  class name, not certification of hidden properties or a real quality approval.
- `missing_cap`: original tube present, cap hidden in the session layer.
- `no_vial`: both original tube and cap hidden.

Crooked caps and damaged bodies are **not implemented** by this pilot. The S
fixture remains context, not a trained target. Authored G01–G30 names identify
synthetic positions; correspondence with an independently supplied real fixture
coordinate file is not yet established. Existing G/S identifiers do not establish
the current physical good/scrap routing semantics.

The whole original station layer and camera JSON are hash-pinned. The renderer
uses session-layer edits only. It keeps the original cap and tube geometry,
changes visibility, cap diffuse color, light intensity and camera image-plane
translation, then applies the supplied 768×576 overscan-to-640×480 lens mapping.
The twin is static geometry, not a physics model.

## Frozen data and fitting design

| Item | Design |
| --- | --- |
| Scene families | 60, each with a paired all-capped golden and mixed inspection image |
| Training | 36 complete scene families, 1,080 target crops |
| Development | 12 complete scene families, 360 target crops |
| Test | 12 complete scene families, 360 target crops |
| States per scene | Ten targets per class, randomly assigned among 30 G slots |
| Nuisance | Light scale 0.7–1.35; four cap-color families with color variation; camera translation ±4 pixels |
| Crop | 48×48 native pixels around the projected authored target cap center; resized to 64×64 |
| Classifier | Custom Conv32/64/128, stride 2, GroupNorm8, adaptive 4×4 features, Linear128/3 |
| Input arms | Current RGB (3 channels); current + matched golden RGB (6 channels) |
| Initialization | Random, no generic pretraining |
| Fit | 30 epochs, batch 64, AdamW learning rate 0.001 / weight decay 0.0001 |
| Selection | Highest development macro recall each epoch, earliest tie |
| Test gate | Decode test RGB only after both models and their development thresholds freeze |

The paired arm is a custom small CNN, **not TAO Visual ChangeNet**. This is a
baseline comparison to establish a reproducible fixture task before adding a
larger NVIDIA model or more defect classes. There are no API/model downloads.

All crops and golden references from one scene family stay in one split. Image
hashes and scene IDs are checked for cross-split reuse. Every split uses the same
CAD geometry and the same 30 positions, so held-out synthetic scenes do not prove
new-geometry, unseen-position or real-domain generalization. Golden images share
the scene's camera/lighting and are generated as known-capped simulation states;
no real golden-acquisition procedure is implemented.

The diagnostic occupied-target score is `1-P(pass)`. Its threshold is the minimum
development score across synthetic missing-cap cases, so every such development
case is caught. It is frozen before test. Report its test false alarms and
misses alongside full three-class confusion, per-class recall and all-30-correct
scene counts. This score is not a calibrated probability of quality. Empty slots
are excluded from the occupied-target defect diagnostic; raw occupancy is still
scored as its own class.

## Integration boundary

The standalone experiment can emit attributed observations with fixture, target
slot, crop-to-full-image box, source RGB hash, model identity and class scores.
Its output must retain `data_domain=synthetic`, unavailable real metrics and an
unlinked physical-vial association. No filling warning is cleared by a cap
prediction. A `no_vial` prediction becomes a missed-placement concern only with
an externally confirmed expected occupancy or order-complete context.

Srijan owns the shared model pipeline; Josiah/Hayden own the operator/backend.
This pilot supplies an experimental model/evidence contract rather than changing
their modules. All observed states remain proposals for operator review; no
industrial device is connected or commanded.

## Execution and artifacts

Remote root:
`~/ai-club-tools/experiments/dataset1-pilot-20261022-v1`

Local root:
`C:/Users/Austin/ai-club-tools/results/dataset1-pilot-20261022`

- `inventory.json`: CPU USD mesh/camera inventory, original stage identity.
- `smoke/` and `smoke-targets.png`: first **training** family's current/golden
  images and all 30 labeled target crops, visually checked before the full batch.
- `experiment-plan.json`: frozen families, selection policy and helper identity.
- `rendered/`: native distorted RGB pairs and source/geometry/label manifest.
- `fitted/`: two model checkpoints, curves, freeze records, predictions and
  synthetic-only test confusion/metrics after completion.

Inspect process state and completion markers before continuing; do not duplicate
jobs or replace frozen helpers. The smoke capture succeeded; the first full
launch stalled during Kit initialization before any frame was generated. Its
log and empty output directory were retained, and recovery uses a fresh output
with the same pinned renderer/plan. This startup failure is separate from model
performance. No training was repeated to recover it.
The first smoke render also logged a relative-path asset resolution error despite
valid capture; recovery retained that log. The repository helper now resolves
its experiment root and removes experiment-specific arguments before Kit parses
argv. The experiment's executed renderer snapshot and plan hash remain unchanged.
No single cause of the initial startup stall was established by the recovery.

CPU integrity checks:

```text
python -m unittest discover -s setup/vision -p test_dataset1_pilot.py -v
```

Four checks cover scene/golden split leakage, missing target attribution and the
distinction between occupancy and the occupied-target defect diagnostic. Test
fixtures are synthetic examples, not team-approved annotations.

## What is still needed

**Real station imagery is already available:** Dataset 2's
`datasets/dataset2/csi-vial-fill-dataset/cap_seal/` contains the overhead robot and
two final inspection fixtures. Four indexed Run 2/4 stills were visually checked
against the tray twin. Its README also states that CapSealCam was matched to the
real cap_seal clip. The generic dataset README's “cap/seal station” name should
not obscure that this is the final-fixture view needed for a transfer probe.

The folder contains 1,181 stills and five videos. There are **335 indexed stills
in Runs 1–4** (5/78/95/157), plus 845 in Run 5 and one unassigned image. Run 5
pixels were not inspected for this audit and remain excluded from the new work.
The Run 4 video is 640×480, 30 fps, 316.066667 seconds. This folder contains only
images, videos and an index, with no reviewed per-slot defect labels or real
fixture-coordinate export. Raw footage exists; reviewed accuracy truth and
qualified real reference handling are the missing pieces.

The local source-hash-pinned comparison is
`C:/Users/Austin/ai-club-tools/results/dataset1-cap-camera-audit-20261022/real-vs-tray-twin.png`;
the four-source contact sheet and inventory are alongside it. The comparison is
illustrative, not a physically matched empty/full state or an approved annotation.
No model inference or new fitting was performed in the camera audit.

Use this footage for frozen-model transfer inspection after validating the real
target-slot crop mapping. A small human-reviewed evaluation subset can establish
labels without fitting on real imagery. October23 organizer clarification:
the previously advertised separate labeled package is deprecated. The supplied
participant tool provides native-image fixture calibration and manual review,
not approved labels or a trained autolabeler; see the
[five-class follow-up](DATASET1-FIVECLASS.md#organizer-participant-tool-and-native-imagery).
Obtain qualified real golden references
before interpreting the paired arm. Then assess misses and false alarms per class,
full-scene behavior and lighting/pose sensitivity. Extending the pilot to crooked
caps/damaged bodies requires visibly meaningful defect fixtures and its own
frozen evaluation; hiding a cap does not cover those classes.

There is no confirmed physical vial-to-slot link, supervised physical test result
or completed Dataset 1 dashboard integration from this pilot.
