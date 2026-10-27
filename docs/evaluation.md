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

## Operator application rehearsal

The CPU Docker path on `austin-work` processes a re-encoded eight-second Run4 visual
excerpt through the released YOLO/U-Net, unchanged QA, FastAPI and React/Nginx.
The browser samples 41 frames: 29 have a model proposal and 14 have an available
sample. These counts describe this integration clip, not segmentation accuracy,
independent test examples, manufacturing defect accuracy or production throughput.
Original Run4 source-frame hashes/PTS are retained separately from the browser
seek clock and re-encoded demo clip. No Run5 data or model fitting is involved.

The browser rehearsal covers original source display, mask artifacts, an attributed
operator response, stale-run response rejection, capture orientation, stopped and
offline states, and absence of invented progress. SQLite events, the response and
an exact rendered artifact survive an API restart. See the compact
[verification record](../setup/verification/demo-integration-20261027.json) and
[laptop instructions](demo.md). Physical camera/reference/recipe qualification and
performance on the target laptop/CSI host remain separate tests.

## GB10 native run

On spark8 (NVIDIA GB10, aarch64, Ubuntu 24.04), the [native kit](../setup/gb10/README.md)
installed torch 2.14.1+cu130 and served the app on `cuda:0` without Docker. The same 40
extracted Run4 excerpt frames were posted sequentially to the API:

| Path | Time per frame | Proposals | QA available |
| --- | ---: | ---: | ---: |
| x86 laptop, CPU Docker | ~300 ms | 28 / 40 | 14 / 40 |
| GB10 native, CUDA, first pass after start | 138–145 ms | 28 / 40 | 14 / 40 |
| GB10 native, CUDA, warm pass | 86 ms | 28 / 40 | 14 / 40 |

Times are client-measured HTTP round trips, including decoding, inference, QA, rendering
and SQLite/artifact persistence. They are not camera-to-display latency. Against the x86 CPU
path, proposal and QA decisions agreed on 40/40 frames; mask areas differed by at most one
pixel and Lab by at most 0.59. Two GB10 passes were identical. This shows device consistency,
not accuracy. The browser rehearsal on the GB10 (headless Chromium on the device) reproduced
41/29/14 and passed the response, artifact, reset, offline and Investigation checks.

The reproducible kit adds a bundled CPython 3.12.15 and hash-locked packages. From a fresh
extract on spark8, `START.sh --selftest` matched the frozen x86 reference outputs on 40/40 frames
with CUDA (max ΔLab 0.59, 90 ms median) and on CPU (ΔLab below 1e-13, 171 ms median). The GB10
CPU path also matched an offline wheelhouse install. A different GB10 should rerun the self-test
before the demo.

The GitHub path was tested from a fresh clone of `austin-work` on the x86 laptop with an NVIDIA
GPU: `setup/gb10/onboard.sh` downloaded and checksum-verified the model release, built the
`compose.gpu.yaml` image, confirmed CUDA inside the container and passed the self-test (40/40,
max ΔLab 0.59, 115 ms median). The aarch64 Docker GPU image has not been built; on the GB10,
onboarding falls back to Docker CPU and then to the validated native launcher.
[Record](../setup/verification/gb10-native-20261027.json).

## Sequence-diagnosis pilot

The investigation module was exercised on 1,620 analytic four-vial sequences across
ten scenario families (162 each), frozen before scoring. No fitting, rendered
Isaac images, real-data threshold selection or Run5 data was involved.

| Scenario | Observed software behavior |
| --- | --- |
| Nominal | 162/162 inside the experimental color band |
| Persistent wrong recipe | The legacy first-vial rule suggests carryover in 162/162; sequence logic reports persistent deviation with cause undetermined |
| Unhealthy camera / unqualified recipe reference | 162/162 of each withhold diagnosis |
| Attributed delivery discrepancy | All 162 remain visible, including 54 inside the color band |
| Carryover and constructed dosing/optical alternatives | 162 identical-input triplets receive identical diagnoses;76/162 in each family receive a carryover hypothesis, with alternatives retained |

These are feature-model scenario counts. Analytic counterexamples cover 54 process
settings and 216 endpoints where alternative mechanisms share the same ideal color.
They do not establish physical feasibility of the optical construction or real
causal accuracy. The audited Run4 probe has 264 frames/six correlated clips and 105
QA-eligible samples, but supplies zero qualified completed-vial RCA inputs.

The [Investigation panel](investigation.md) exposes this uncertainty, source
bindings and missing evidence. Its examples stay synthetic and separate from run
records. Cause confirmation, product quality and hardware commands remain unset.
