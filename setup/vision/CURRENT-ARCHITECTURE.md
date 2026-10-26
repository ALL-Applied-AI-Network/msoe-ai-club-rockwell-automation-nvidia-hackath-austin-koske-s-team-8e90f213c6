# Current architecture decision — October 24, 2026

**October26 publication:** [team download, current diagram and runnable model
example](../../team_handoff/temporal_cnn/README.md) are published as a scoped
`main` handoff. Operator-app implementation remains separate development work;
this publication does not establish a connected live deployment.

Austin selected the **direct paired temporal U-Net as the leading liquid-model
candidate for team integration** after inspecting the U-Net, DINOv3 and SAM
comparisons. Its liquid branch is SAM-free and Faster R-CNN-free. This records
the integration direction; existing application defaults have not been changed.

SAM2.1 remains an optional experimental refinement branch. DINOv3 mask heads
remain research comparators. Neither is a required stage in the selected path.
Earlier team PDFs describe the original architecture and remain dated history;
use this document for the current model choice and integration boundaries.

## Selected processing path

```text
Current original RGB + native frame identity
  ├─ synthetic-trained YOLO11n → same-frame vial geometry
  └─ fixed station liquid ROI + earlier aligned reference
       → background-anchor gain correction
       → signed RGB difference + absolute RGB difference (6 channels)
       → small U-Net → liquid-body mask + presence probability
       → frozen presence gate + largest mask component
             ↓
Vial association + unchanged measurement QA on ORIGINAL RGB
  → eligible RGB/Lab sample, or measurement unavailable with reasons
  → recipe/process comparison only when its independent inputs are qualified
  → source-pinned operator review, acknowledgement and response
```

YOLO supplies geometry for association/QA; the U-Net currently consumes a fixed
112 × 192 station ROI, not a YOLO-generated moving crop. Preprocessing uses the
current and earlier reference images, not future frames. This is a paired-image
model, not a recurrent video model or a physical fluid simulator. Reference
capture, alignment, qualification and reset are separate integration responsibilities.
The six development clip references remain unreviewed, not certified empty.

Retain the original RGB for color measurement. Gain correction and differences
are liquid-model inputs only. A visible raw mask can coexist with failed sample
QA; keep those states separate. A missing mask never proves an empty vial.

## Frozen model identities and training

| Component | Identity / provenance |
| --- | --- |
| Liquid mask and presence | Custom GroupNorm U-Net, 16/32/64/128 channels; `paired_diff_gain`; random initialization |
| Liquid checkpoint | Selected epoch 40 of 60 using synthetic development criteria |
| Liquid CSI training | 144 synthetic images / 36 scene families; 48 development images / 12 disjoint families |
| Vial detector | Synthetic-trained YOLO11n, generic COCO initialization disclosed |
| Liquid mask / presence thresholds | Both 0.5, unchanged |
| Runtime split | GPU inference; CPU source handling, geometry, sampling QA and operator workflow |

Liquid SHA256:
`6714fe32191d8eba230aa87246f38a25c2906c619d60d7ba30c9e60205352eef`.

Vial SHA256:
`0bfac07a256eaf190416b632ad1f094a6b534309df1bb11a8c175e9d811b56bc`.

New CSI fitting and checkpoint/threshold selection use synthetic data. Real
imagery supports evaluation and independent review. Run 5 stays outside fitting,
selection, tuning and sampling decisions. Selected synthetic development scores
are not independent test performance. The scene masks are geometric supervision,
not human-reviewed optical pixel truth.

## Optional branches and measured evidence

| Branch | Status | Existing evidence |
| --- | --- | --- |
| Direct temporal U-Net | Leading integration candidate; no SAM | Latest matched GB10 CV median / p95: 16.37 / 24.86 ms; 189 proposals and 105 QA-eligible samples / 264 frames |
| Temporal → SAM2.1 Hiera Tiny | Optional experimental refinement | Earlier same-frame study: 185 QA samples / 264; 72.62 ms incremental refinement/QA with upstream outputs cached, not full pipeline latency |
| DINOv3 + mask/presence heads | Research comparator | Paired arm: 29.43 / 31.63 ms full CV and 180 QA samples / 264; overlays still extend into glass/stream |
| Faster R-CNN → SAM2.1 | Earlier architecture / comparator | Not required by the direct temporal branch; real-trained versions are explicitly legacy development comparators |

The latest direct/DINO trial used the same 264 inspected Run 4 frames, fresh vial
inference in each arm, a full warm-up and two measured passes. CV timing includes
vial inference, liquid processing and QA, excluding decode, transport, rendering
and browser latency. Optional SAM figures are from the earlier execution.

QA counts measure sampling availability, not mask, onset or recipe accuracy.
All 77 paired-DINO QA gains coincide with U-Net containment rejection; 76 already
had U-Net proposals. Taller masks can pass the box-containment gate while covering
glass. Do not rank masks or claim newly detected liquid from that count alone.
U-Net's leading status is an engineering decision informed by visual inspection
and compute; independent real segmentation/onset accuracy remains unavailable.

## Application and operator boundary

The opt-in full-app demo can replay cached direct temporal and optional SAM
outputs. Its branch switch does not establish live inference or alter the older
station defaults. The actual live adapters remain Srijan's integration work;
Josiah/Hayden own the operator frontend/backend. Cameron's SAM2.1 work remains
available as the optional comparison/refinement capability.

Intended recipe, independently accepted recipe reference, confirmed completion
and physical vial identity are separate inputs to a qualified recipe comparison.
An operator acknowledgement or closed follow-up does not approve product release.
The current output is evidence and operator advice; there are no direct camera,
industrial network, PLC, sorter or actuator connections introduced here.

Dataset 1 final inspection remains a separate cap/occupancy model and camera
adapter. Its cap observations do not clear a filling warning or establish body
condition. Cross-camera physical vial identity needs explicit evidence.

## Generalization and next validation

The architecture and commissioning workflow can be reused. This checkpoint's
cross-dataset generalization is unproved. Current dependencies include fixed ROI,
vial geometry, reference alignment and synthetic appearance coverage. A late
reference changed 126/264 proposal decisions in the existing perturbation test;
this is prediction sensitivity, not a reviewed error rate.

Next: freeze this baseline; obtain independent pool-boundary, empty/stream-only
and first-visible timing review; evaluate fresh supervised real recordings; test
reference capture/reset/alignment; and measure complete camera-to-display latency.
Any follow-up training uses expanded synthetic coverage and synthetic selection
before real scoring. Another station needs its own qualified camera profile and
transfer evaluation, potentially synthetic retraining. Do not promise arbitrary
camera/container transfer or infer physical accuracy from digital perturbations.

## Evidence and implementation references

- [Exact model input/output contract and artifact paths](TEMPORAL-POOL-CANDIDATE.md).
- [Synthetic training and stress experiments](SYNTHETIC-ITERATIONS.md).
- [Matched SAM comparison and optional refinement](SAM-TEMPORAL-COMPARISON.md).
- [DINOv3 trial and containment/visual failure audit](DINOV3-DIRECT-MASK.md).
- [Operator demo behavior and cached/live boundary](../../replay/operator/FULL-APP-DEMO.md).
- [Candidate evidence and reference qualification contract](../../replay/operator/COMMISSIONING.md).
- [Independent mask/onset review workflow](../../annotation_review/VISIBLE-POOL-REVIEW.md).

Local U-Net assets/media:
`C:/Users/Austin/ai-club-tools/results/synthetic-iterations-20261022/gb10-results/`;
start with `temporal-preview-v1/temporal-pool-candidate.mp4` and
`masks-v1/paired_diff_gain/`. Preserve these and frozen executed helpers during
workspace cleanup. This documentation decision starts no new GPU jobs.
