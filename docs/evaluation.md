# Evaluation

The liquid-model results support a development candidate. Reviewed real mask
and first-visible onset accuracy are not yet available.

## Matched recorded-frame comparison

The latest GB10 comparison uses 264 inspected Run4 frames from six correlated
clips, identical source/reference pairs, fresh vial inference, a full warm-up
and two measured passes per arm. Timings include vial detection, liquid
processing and sampling QA. They exclude decoding, transport and display.

| Liquid branch | Median CV | p95 CV | QA-eligible samples |
| --- | ---: | ---: | ---: |
| Direct temporal U-Net | 16.37 ms | 24.86 ms | 105 / 264 |
| Paired DINOv3 with mask/presence heads | 29.43 ms | 31.63 ms | 180 / 264 |

[Recorded comparison and verification](../setup/verification/dinov3-direct-mask-20261024.json).

QA eligibility means that sampling checks passed. It is not segmentation or
recipe accuracy. All 77 paired-DINO gains coincide with U-Net containment
rejection, and 76 already had U-Net proposals. Taller masks can satisfy a box
gate while covering glass. Visual inspection favored the tighter U-Net pools;
this is an engineering selection, not proof of superior real accuracy.

An earlier temporal-prompted SAM2.1 study produced 185 / 264 eligible samples.
Its 72.62 ms figure measures incremental refinement/QA with cached upstream
outputs, not full pipeline latency. Do not compare it directly with the table's
complete CV timing. [Study record](../setup/verification/sam-temporal-comparison-20261022.json).

## Synthetic development

The selected U-Net's synthetic development result is 48 / 48 presence decisions
and 0.8933 mean positive geometric IoU. The 48 images belong to 12 families and
were used for checkpoint selection. They are neither an independent synthetic
test nor real optical ground truth.

Stress tests show sensitivity to reference selection and alignment. A later
reference changed 126 / 264 real proposal decisions. This measures prediction
sensitivity, not an error rate against human labels.
[Training and stress record](../setup/verification/temporal-candidate-checks-20261022.json).

## Reproducibility

The publication check replays two frozen machine examples on CPU. Raw and
component masks match exactly; presence agrees within 1e-4. Sixteen focused
tests cover geometry/sampling, review integrity and onboarding file checks.
[Publication record](../setup/verification/temporal-cnn-release-20261026.json).

These checks validate bytes, implementation behavior and contract handling.
They do not establish independent model accuracy, reference suitability or a
working camera-to-dashboard deployment. Experiment JSON files retain their
original dates, paths and hashes; old local paths are provenance, not run commands.

## Remaining acceptance measurements

| Measurement | Evidence needed |
| --- | --- |
| Pool presence precision/recall and mask IoU/Dice | Independent source-matched human review, including empty, stream-only, faint and uncertain cases |
| First-visible delay | Reviewed absent/present intervals on native video clocks |
| Recipe warning false alarms and misses | Independently accepted recipe references and attributed observed outcomes |
| Reference robustness | Capture/reset/alignment tests on fresh recordings and supervised camera conditions |
| End-to-end latency | Source capture/PTS through inference, QA, transport and visible display |
| Transfer to another camera/container | Separate camera profile and evaluation before claiming reuse of this checkpoint |

Use a fresh evaluation set without tuning the frozen candidate on its results.
Report uncertain/excluded coverage and missed proposals alongside successful
cases. See [the test procedure](operations.md#supervised-test).
