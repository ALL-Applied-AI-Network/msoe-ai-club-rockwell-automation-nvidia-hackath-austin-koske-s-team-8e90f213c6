# AI-guided station onboarding — design and initial implementation

## Purpose and current scope

Help an integrator take supplied data and a manufacturing inspection objective
through a reproducible, evidence-producing commissioning workflow. Austin's
discussion with Dr Riley/Brett motivates making the procedures reusable skills.
The initial implementation is one project skill plus a deterministic, read-only
asset checker. Existing rendering, training, evaluation and operator tools remain
the stage implementations. They are not yet a fully automated onboarding service.

The task-specific current pilot is Dataset2 liquid inspection with direct
temporal U-Net. Other stations reuse the workflow pattern with their own targets,
profiles and validation. A universal drop-in model is not claimed.

## One workflow with explicit stage outputs

```text
Inspection goal + recordings / twin assets + deployment constraints
  → AI inventories inputs and identifies missing requirements
  → source inventory + station/camera/reference profile
  → synthetic scene families and labeled training/development manifest
  → existing baseline or synthetic-trained model, frozen with provenance
  → independent transfer evaluation + reference/QA failure checks
  → actual model-worker → backend → operator rehearsal
  → pinned candidate deployment package + known limits + operator runbook
  → authorized deployment and supervised acceptance test
```

The AI coordinates tools, explains failures and prepares artifacts. Deterministic
helpers check hashes/clocks/contracts and compute metrics. Human/domain review
establishes empty references, visible liquid targets, acceptable recipe examples
and operating context. The operator responds to evidence. There is no LLM call
required for each camera frame and no new PLC/actuator control path.

| Stage | Reuse today | Output / completion evidence |
| --- | --- | --- |
| Understand inputs | Read repository status, dataset metadata and prior reports | Goal, source roles, camera, deployment target and unanswered questions |
| Inventory | `setup/vision/inspect_csi_capture.py`, existing dataset checks | Actual source hashes, decoded dimensions, native frame/PTS traces and anomalies |
| Profile/reference | Current fixed-ROI contract; `replay/operator/commissioning.py` | Versioned geometry and independently attributed reference/qualification state; resets tested |
| Synthetic preparation | Existing renderer and paired-family preparation helpers | Pinned scene/image/target identities, scene-family splits, label definitions and optics limits |
| Fit/freeze | Existing U-Net trainer and completed model metadata | Synthetic-selected checkpoint and helper/config hashes, initialization and data provenance |
| Transfer evaluation | `annotation_review.evaluation_bundle` and existing pool/slot scorers | Independent reviewed metrics, uncertain/excluded coverage and unavailable fields |
| Integration rehearsal | Candidate evidence contract, actual inference worker and operator app | Same-source images/masks/QA, reference/reset tests, source age, measured end-to-end latency and response record |
| Deployment handoff | Existing launch/config paths plus completed evidence | Exact artifacts/environment, startup/reset/rollback steps, operator procedure, approval and remaining limits |

Inspect completion markers/jobs before any GPU work. Preserve original data and
scene assets. New CSI fitting uses synthetic data; real review is evaluation-only.
Run5 remains outside fitting/selection/tuning/sampling decisions. Reuse the
existing GB10/Singularity/Slurm execution arrangements rather than inventing a
new infrastructure stack. GPU work stays off Austin's local machine.

## Project skill

Entrypoint:
[`.agents/skills/commission-vision-station/SKILL.md`](../../.agents/skills/commission-vision-station/SKILL.md).
It references the maintained architecture and existing stage tools. Normal
automatic skill selection is retained. In a compatible Codex session opened
with this checkout, request the skill explicitly, for example:

```text
Use $commission-vision-station to inspect this recording folder and prepare a
station onboarding package. Reuse the frozen synthetic U-Net baseline. Show
which inputs are missing and the next executable stage; preserve source files.
```

The `.agents/skills` project location follows the
[official customization documentation](https://learn.chatgpt.com/docs/customization/overview).
The skill and read-only asset checker are included in the October26 scoped
architecture handoff. They have not been verified across other agents. The skill's format validation is not
proof of autonomous behavior or complete deployment capability.

## Read-only asset checker

`setup/vision/onboarding_readiness.py` uses only the Python standard library.
It hashes declared local files, flags missing inputs/hash mismatches and produces
next actions. It never imports a model, installs software, starts a job/server,
qualifies a reference or approves deployment. Referenced reports require separate
semantic validation with the existing stage-specific tools.

Request JSON:

```json
{
  "schema": "station-onboarding-request/v1",
  "station_id": "filling-camera-A",
  "task": "vial_fill",
  "goal": "Detect the first visibly identifiable bottom pool and expose qualified color evidence",
  "assets": []
}
```

For each available asset, add an object with `role`, `path`, expected `sha256`
and `data_domain` (`real`, `synthetic`, `mixed` or `not_applicable`). Paths resolve
relative to the request JSON. Roles are `camera_profile`, `synthetic_manifest`,
`checkpoint`, `training_evidence`, `temporal_reference`, `real_evaluation_report`
and `live_rehearsal_report`; `accepted_recipe_reference` and
`deployment_manifest` are optional asset roles. Recipe decisions still require
the independent recipe qualification gates when applicable. A path or role name
does not certify the file's contents. A reported synthetic domain is a declaration
requiring provenance review, not a visual-origin classification.

```text
python setup/vision/onboarding_readiness.py --request PATH/request.json --out PATH/fresh-report
```

Use a fresh output outside the request and source directories. Outputs are
`request.snapshot.json`, `readiness.json`, `NEXT-STEPS.md` and a hashed completion
marker. Exit2 means asset checks incomplete; exit0 means only that the declared
required files/hashes passed. **Both retain `deployment_authorized=false`,
`qualification_checks_executed=false` and `real_accuracy=null`.** Human/physical
evidence cannot be manufactured by satisfying a file checklist.

## Tuesday-sized demonstration

Keep the first iteration small: give the agent a supplied folder/objective,
produce an actual inventory and missing-input report, reuse a frozen model,
then drive the same operator pipeline the mentees are integrating. Include a
bad-reference or failed-QA example and a source-pinned human response. Show which
stages actually ran and which remain pending. A saved-recording fallback is fine
when labeled accurately; it is not an approved industrial camera connection.

This workflow should wrap the common inference worker and dashboard interfaces.
Srijan owns the worker; Josiah/Hayden own the backend/operator flow; human reviewers
own labels and reference judgments. New onboarding UI, broad framework rewrites
and additional model families are not required for Tuesday's core demo.

## Executed starter verification

The skill passes the skill-creator format validator. Four CPU fixture tests check
hash/domain failures, source/output preservation, duplicate/unsupported requests,
and the invariant that even all files present cannot authorize deployment.
No independent agent end-to-end skill run has been performed.

A local invocation used the existing U-Net checkpoint, frozen training metadata,
actual192-image /36-train-family /12-development-family synthetic split spec and
an explicitly unqualified camera-profile draft derived from model metadata.
All four supplied files matched their hashes; the source files were unchanged.
The helper exited2 and listed three assets not supplied to that request: a
qualified temporal reference, independent real evaluation and a live integration
rehearsal report. This is request completeness, not a claim that other reports
cannot exist elsewhere or that file hashes qualify these evidence types.

Local evidence:
`C:/Users/Austin/ai-club-tools/results/ai-onboarding-20261024-v1/readiness-v2/`.
`executed-check-v2.json` records the invocation and boundaries. The earlier
two-asset request/report remain preserved. No new fitting, live deployment,
physical qualification or main push occurred.

## Measuring the proposed value

Run the same onboarding task manually and with assistance. Record active engineer
time, elapsed time, required decisions, configuration edits, failed/repeated runs,
artifact reuse and downstream quality/latency. First test frozen-checkpoint
transfer on new conditions; report subsequent synthetic adaptation separately.
Retain failures and abstentions. Claim setup savings or cross-station scalability
only after those measurements exist.

Presentation wording/evidence:
[lessons and claims](../presentation/LESSONS-AND-CLAIMS-20261024.md).
