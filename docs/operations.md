# Operations

## Compute and input sources

Use CPU for local packaging, example replay and review. The existing GPU
experiments run on GB10; Rosie GPU jobs use Slurm and Singularity 3.10.
[Compute setup](compute.md) documents the launchers and environment profiles.
Inspect existing jobs and completion markers before resubmitting work.

The selected model currently accepts supplied recordings/images. No approved
industrial camera endpoint is configured in this release. A live feed requires
CSI to provide or approve the device, feed and recording arrangement. Do not
scan the plant network or connect to its cameras, sensors or control systems.

## Reference setup

1. Confirm camera ID, dimensions/orientation and the station ROI/anchor profile.
2. Record an earlier empty-vial reference with an attributed reviewer and
   supporting evidence. The supplied example reference is unreviewed.
3. Confirm alignment and native-clock comparability. Supply those facts to
   `ReferenceGate`; the gate cannot derive them from model output.
4. Reset on vial context, camera/profile or stream changes, rewind or declared
   discontinuity. Test each transition before making recipe comparisons.

Recipe endpoint references are separate from empty temporal references. They
must be independently accepted for the intended recipe. The operator also
needs completion and physical vial identity before an eligible comparison.

## Before the CSI visit

Booking time, assigned workstation/GPU and approved feed arrangement are still
unconfirmed. Verify OS, GPU/driver, software access, supervision and recording
permission with CSI. Do not assume that a desktop TensorRT engine or installed
GB10 environment transfers to the assigned workstation.

Bring the exact model/data manifests, a short source recording, a qualified
reference if available, runnable environment instructions, and the released
single-panel MP4 as a fallback. The original visit PDF and blank forms remain
in the [model release](https://github.com/ALL-Applied-AI-Network/msoe-ai-club-rockwell-automation-nvidia-hackath-austin-koske-s-team-8e90f213c6/releases/tag/temporal-cnn-v1.0.0).

First rehearse one recording through the actual worker, QA and operator UI.
Neither the model-only example nor the widgets page completes that rehearsal.
The operator display must show aligned masks, source age and QA reasons, and
must keep unavailable evidence distinct from a normal recipe result.

## Supervised test

Start with an approved saved recording on the assigned workstation. Establish
the following cases before switching to an approved live source:

| Case | Expected observation |
| --- | --- |
| Empty glass / incoming stream | No automatic empty or acceptable-product conclusion |
| First visible pool | A source-aligned proposal; color may still be unavailable |
| Established fill | Sampling evidence and actual QA reasons |
| Neighboring vial | No substitution for the vial under the nozzle |
| Missing/late/misaligned reference | Unqualified comparison and an explicit reason |
| Rewind, dropped frames, context change | Correct reset; no stale mask/result reuse |
| Matching RGB with failed QA | Measurement stays unavailable |
| Operator acknowledgement | Response recorded separately from recipe/product approval |

Retain source/model/config identities, frame timing, proposals, final sampling
masks, QA, reference decisions and operator actions. Measure source-to-display
latency and queue age, not just GPU inference. Human reviewers supply pool/onset
and recipe outcomes; model agreement is not ground truth.

The output for this test is operator advice. Any physical action follows CSI's
approved operating procedure, outside a direct software control path.

## Station onboarding

The project skill at
[commission-vision-station](../.agents/skills/commission-vision-station/SKILL.md)
coordinates source inventory, camera/reference profiling, synthetic preparation,
model freezing, transfer evaluation and deployment preparation.

`setup/vision/onboarding_readiness.py` is a standard-library file/hash checker.
Its `station-onboarding-request/v1` input names a station, task, goal and asset
records with `role`, `path`, `sha256` and `data_domain`. Required roles are
camera profile, synthetic manifest, checkpoint, training evidence, temporal
reference, real evaluation report and live rehearsal report. Paths resolve
relative to the request file.

```bash
python setup/vision/onboarding_readiness.py --request request.json --out readiness-01
```

Exit 2 means missing/invalid assets. Exit 0 means the declared required files
and hashes passed; it does not qualify their contents or authorize deployment.
The checker starts no models, jobs, servers or hardware connections.
