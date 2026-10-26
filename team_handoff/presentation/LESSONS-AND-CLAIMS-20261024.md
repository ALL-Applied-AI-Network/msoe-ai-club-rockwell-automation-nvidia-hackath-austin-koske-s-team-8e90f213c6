# Presentation lessons and claims — October 24

Working presentation notes from the executed experiments, the mentee branch
review and Austin's discussion with Dr Riley/Brett. The AI-guided onboarding
direction is a proposed extension with an initial skill/checker, not a completed
autonomous data-to-physical-deployment system.

## Main message

> We have promising synthetic-trained perception. Making it useful on a new
> manufacturing line requires a repeatable way to qualify the camera, references,
> measurements and operator response. We are packaging that commissioning
> knowledge into reusable AI-guided skills.

Use “promising model behavior” rather than “the models are good everywhere.”
Current U-Net boundaries look better than the tested DINO branches in several
inspected clips; independent real mask/onset accuracy is still unavailable.
The product opportunity is reducing the engineering work between a model and a
trustworthy station installation. Do not frame this as a failure by the mentees:
their integration exposes assumptions that any new installer would face.

## Lessons to show, with concrete evidence

| Lesson | What we observed | Presentation implication |
| --- | --- | --- |
| Models require an input contract | U-Net uses native640×480, fixed112×192 ROI and earlier aligned reference; dashboard capture mirrors/resizes pixels | Deployment needs an explicit camera profile and source identity, not simply a checkpoint file |
| References are part of the system | A later reference changed126/264 proposal decisions in the existing stress test | Demonstrate reference capture/reset and invalid-reference handling; changes are sensitivity, not accuracy scores |
| QA availability is not segmentation accuracy | Paired DINO supplies180/264 QA samples versus U-Net105/264, while some DINO masks include glass/stream; all77 gains coincide with U-Net containment rejection | Explain why more accepted samples alone cannot choose a better liquid model |
| Integration can erase model safeguards | CPU branch-review fixture with `sample_usable=false` still returned color `within_threshold` | Carry QA and qualification through the API/UI; unavailable evidence must stay unavailable |
| Useful output goes beyond classification | The app can retain source-pinned review/acknowledgement/response in the research demo; mentee integration is still pending | Show an operator receiving evidence, responding and retaining a traceable record |
| Sim-to-real is a measured transfer problem | U-Net trained from random initialization on144 synthetic images; real video shows promising transfer, reviewed real accuracy pending | Show source renders, frozen model provenance and real overlays; clearly label what has been measured |
| Latency is a complete-system property | Latest U-Net CV median16.37ms excludes decode, network and browser | Report end-to-end source age separately before claiming realtime deployment |
| Overall accuracy can conceal the defect that matters | Separate cap confirmation has98.75% exact accuracy but31/38 crooked-cap classifications | Present defect recall, false alarms, exclusions and class scope; synthetic results remain synthetic |
| Generalization includes commissioning effort | Fixed ROI/reference dependencies remain; no second-station liquid-checkpoint transfer proof | Measure setup time, human decisions and modifications required on new data |

Evidence: [current architecture](../../setup/vision/CURRENT-ARCHITECTURE.md),
[U-Net/reference checks](../../setup/vision/TEMPORAL-POOL-CANDIDATE.md),
[DINO audit](../../setup/vision/DINOV3-DIRECT-MASK.md),
[branch review](../integration/BRANCH-REVIEW-20261024.md),
[cap limits](../../replay/operator/FULL-APP-DEMO.md#cap-display--metric-correction).

## The product story

An integrator brings recordings, digital-twin assets, an inspection objective
and deployment constraints. The AI-guided workflow helps inventory the data,
prepare a camera/reference profile, reuse or train a synthetic model, run the
appropriate checks, and produce a deployment handoff with specific open questions.
Operators or domain reviewers supply the physical/quality judgments the software
cannot establish. The same inference worker and dashboard remain the runtime
system; the agent helps configure and validate that system offline.

Suggested speaking line:

> A checkpoint does not tell an installer whether its camera is aligned, its
> empty reference is valid, or its color sample supports a recipe decision.
> Our workflow makes those requirements explicit and helps the installer work
> through them with traceable evidence.

This supports adoption alongside existing equipment with operator-mediated
actions. Do not claim automatic PLC integration, recipe acceptance, physical
vial tracking across cameras or autonomous process correction.

## Five-minute presentation outline

| Approximate time | What to show |
| --- | --- |
| 0:00–0:40 | Real problem: intended recipe versus observed vial appearance; why an operator needs timely evidence |
| 0:40–1:25 | Twin → synthetic images/targets → frozen compact U-Net; show actual renders and real transfer |
| 1:25–2:05 | Architecture: vial geometry + direct liquid mask/presence → sampling QA → qualified comparison or unavailable |
| 2:05–3:20 | Integrated operator demonstration: early/established pool, unusable reference/sample, retained response; distinguish live inference from replay |
| 3:20–4:20 | AI-guided onboarding prototype: point at inputs, inventory/pin assets, identify missing reference/profile evidence, produce next steps |
| 4:20–5:00 | Actual metrics, known failures, adoption effort and the next physical/transfer validation result |

Until the live loop is tested, label its recording as file replay/cached evidence
as applicable. Until downstream onboarding stages execute, show them as planned
stages rather than claim the readiness helper trained or deployed a model.

## Measurements to collect before the final submission

- Independent reviewed first-visible delay, pool mask IoU/Dice, presence recall,
  empty/stream-only false alarms and uncertain/excluded coverage.
- Sampling availability and failure reasons, separately from segmentation scores.
- Camera-to-display latency p50/p95, processed rate, source age and dropped frames.
- Reference invalidation/reset behavior and operator acknowledgement/response.
- Onboarding elapsed and active engineer time, number of required human inputs,
  manual configuration edits, repeated runs/failures and reused artifacts.
- For a second condition/station: frozen-checkpoint results first; then separately
  report any synthetic adaptation, its cost and independent transfer evaluation.

No onboarding time saving or physical defect-rate reduction has been measured
yet. Record the manual baseline and assisted workflow on the same task before
claiming a reduction. A reused recording or alternate clip is not evidence of
generalization to another factory.

## Current versus proposed capability

| Capability | Current status |
| --- | --- |
| Synthetic U-Net fit, frozen selection and real overlays | Executed; independent real accuracy pending |
| Model comparisons, stress checks, CPU reference guards | Executed, with documented evidence limits |
| Mentee frame/WebSocket transport | CPU fixture verified; U-Net/QA/operator integration pending |
| Project onboarding skill and asset/hash checker | Initial implementation; checker verified separately |
| Agent automatically performing every stage on a new station | Proposed; not demonstrated end to end |
| Physical station qualification and operator deployment | Requires the actual supervised trial and qualified inputs |

Detailed workflow and stage outputs:
[AI-native onboarding design](../integration/AI-NATIVE-ONBOARDING.md).
