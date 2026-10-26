---
name: commission-vision-station
description: Onboard a manufacturing vision station in this repository from supplied recordings or digital-twin assets through synthetic fitting, source-matched evaluation and a reviewable deployment handoff. Use for station setup or transfer to new data; ordinary frame inspection does not require this workflow.
---

# Commission a vision station

Produce a resumable station package: source inventory, camera/reference profile,
frozen model provenance, evaluation status, deployment configuration and operator
runbook. Start at the earliest incomplete stage; reuse verified existing work.
Read `AGENTS.md`, `HANDOFF.md` and `setup/vision/CURRENT-ARCHITECTURE.md` from the
repository root. For stage artifacts and supported helpers, read
`team_handoff/integration/AI-NATIVE-ONBOARDING.md`.

## Establish the task and evidence

Inspect the supplied folder and existing reports before asking for paths already
available locally. Establish the inspection target, station/camera, input type,
operator action and intended deployment environment. Discover missing requirements
without inferring acceptable product quality from model predictions. Distinguish
supplied facts, AI suggestions, human attestations and executed measurements.

Create outputs outside source directories at fresh paths. Keep a progress report
with stage status, exact input/output hashes and the next concrete blocker.
Asset presence is not evidence that a reference, evaluation or deployment is valid.

## Follow the unfinished stages

1. **Inventory:** inspect files/hashes, dimensions, orientation and native clocks.
   Use `setup/vision/inspect_csi_capture.py` for recordings in an existing AV-capable
   environment. Do not invent frame timestamps from an assumed frame rate or join
   cameras without explicit mapping evidence. Retain real/synthetic roles.
2. **Profile and reference:** use the native fixed-ROI contract for the current
   U-Net. Another resolution/view needs a new profile and transfer test, not a silent
   mirror/resize. Capture independently qualified earlier empty/aligned reference
   evidence; reuse `replay/operator/commissioning.py` for reference checks. A reference
   is not empty merely because the model predicts no pool.
3. **Synthetic data and model:** inspect existing jobs/completion markers before
   any fitting. Preserve supplied scene assets and coordinate authored changes.
   New CSI fitting/selection uses synthetic data with disjoint scene families;
   real imagery is evaluation-only. Keep Run5 outside selection/tuning/sampling.
   Reuse the frozen direct U-Net baseline before proposing new training. Disclose
   generic pretraining and freeze settings before real evaluation. GPU work stays
   on GB10 or approved Rosie Slurm/Singularity; use the existing job/lock workflow.
4. **Transfer evaluation:** use independent source-matched human pool/onset review
   and existing annotation scorers. Preserve uncertain/excluded cases and missed
   proposals. Report prediction, sampling availability and actual accuracy separately.
   Missing human evidence yields unavailable accuracy, not an estimated score.
5. **Integration rehearsal:** use one source-bound observation contract for the
   model worker and operator app. Preserve raw mask, presence, measurement QA,
   reference status and source age separately. Failed/unqualified sampling cannot
   produce a normal recipe verdict. Exercise source/reset/reference changes, late
   frames and operator acknowledgement/response. Measure end-to-end latency.
6. **Deployment handoff:** pin assets/configuration, environment, measured limits,
   startup/reset/rollback procedure and unresolved gates. A file-replay rehearsal
   or candidate manifest does not establish physical deployment. Continue an
   already authorized local deployment when supported by evidence; industrial
   camera/network/PLC/actuator access remains outside this repository's current
   authorization. Produce the concrete package before requesting any genuinely
   missing deployment approval.

For a quick file/hash check, create a `station-onboarding-request/v1` JSON as
documented in the onboarding guide, then run from the repository root:

```text
python setup/vision/onboarding_readiness.py --request REQUEST.json --out FRESH_OUTPUT
```

This CPU-only helper does not inspect qualification semantics, run models or
authorize deployment. Its exit2 means missing/invalid input assets, not a crashed
training job. Follow the reported inputs with the stage-specific tools above.

## Return a reviewable result

Report what actually executed, source/model/config identities, applicable measured
metrics and unavailable fields. Give local PNG/MP4 paths for experiment review;
launch a browser/preview only when requested. Keep direct U-Net as the selected
candidate; SAM refinement is optional. Preserve mentee ownership and existing
app defaults unless the user authorizes the concrete integration change.
Do not claim the AI approved labels, calibrated a physical camera, authenticated
human input or proved cross-dataset transfer by completing a checklist.
