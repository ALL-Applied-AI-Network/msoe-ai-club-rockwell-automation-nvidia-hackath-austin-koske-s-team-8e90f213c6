# Consolidated handoff — October 20, 2026

Continue from **this checkout on `austin-work`**. Replay/vision (`487fa6f`) and annotation
review (`17490c5`) are integrated, with history preserved. The former annotation
worktree is retired. This page supersedes the separate branch/worktree directions.
Austin requested no push to `main`; local `main` remains at `5658d58`, matching
`origin/main`. The two original remote branches remain as recovery copies.

## Where things live

| Work | Source and instructions | Current state |
| --- | --- | --- |
| Replay, tracking and measurement QA | `replay/`; [current vision guidance](replay/VISION.md) | Existing detector/defaults preserved |
| Annotation tooling | `annotation_review/`; [reviewer handoff](annotation_review/HANDOFF.md) and [contracts](annotation_review/CONTRACTS.md) | Published 179-case package; human reviews outstanding |
| Completed comparison | `setup/vision/`; [results and protocol](setup/vision/LIQUID-SPECIALISTS.md) | Four fits/evaluations complete; 12 MP4s verified |
| Latest detector iteration | [First-visible-liquid pilots](setup/vision/EARLY-LIQUID.md) | All three fits and Run 4 evaluations complete; eight-detector scorecard assembled |
| Mask pipeline | [SAM liquid probe](setup/vision/SAM-LIQUID.md) | SAM2.1 tested on 30 Run 4 images; raw masks only |

Local assets are under `C:/Users/Austin/ai-club-tools/` (`/mnt/c/Users/Austin/ai-club-tools/`
in WSL). Results, environments and source labels stay outside Git. The ChatGPT
project mirror is reference data, not a checkout.

Existing untracked `analysis/`, `datasets/`, `docs/`, `experiments/`, `handoffs/`
and `team_plan.md` are preserved in place and excluded **locally** using
`.git/info/exclude`. They are not part of this integration; use `git status
--short --ignored` to see them. Recovery snapshots, including the retired
worktree's ignored build files, are in
`C:/Users/Austin/ai-club-tools/recovery/repo-consolidation-20261020T203829Z/`.

## Completed detector comparison

Root: `C:/Users/Austin/ai-club-tools/results/liquid-specialists-20261020`.
Live scheduler verification during consolidation confirmed array **286264**, all
four jobs **COMPLETED, exit 0:0**. Collector log says the bundle is ready.

`completed-artifacts.tar.gz` and its checksum are now local and verified:
`521608a3d63b6de0458914bcc91c421899acc9addfbacb3312f892d22271cc2f`.
The bundle's existing model files matched local bytes; missing DINO model files
were restored. Local smoke artifacts were preserved separately from remote smoke
artifacts in the bundle. No replacement fits were launched for that experiment.

| Arm | Completed epochs | Selected Run 4 epoch | Local evaluation |
| --- | ---: | ---: | --- |
| `yolo-all` | 29 | 19 | `evaluation-yolo/yolo-all` complete |
| `yolo-focused` | 28 | 18 | `evaluation-yolo/yolo-focused` complete |
| `rtdetr-all` | 28 | 18 | `evaluation-rtdetr/rtdetr-all` complete |
| `dinov3-all` | 25 | 15 | `evaluation-dino/dinov3-all` complete |

The final assembly is `evaluation/`: `comparison.json`, PR/problem-frame PNGs,
12 MP4s and `verification.json`. All videos passed full decode/hash verification
(39,292 output frames). DINO achieves 51.34% liquid AP50–95 / 93.53% recall /
80.19% precision at .35 on provisional Run 5 targets, versus baseline 49.18% /
72.64% / 96.80%. More unmatched boxes accompany the recall gain. See the protocol
for all arms and limits. No detector is promoted; measurement QA is unchanged.

## Current model work: first-visible liquid

Austin wants detection as soon as a visible pool appears at the vial bottom and
continued architecture iteration. New root:
`C:/Users/Austin/ai-club-tools/results/early-liquid-20261020`.
Rosie array **286291**: `yolo11n-960`, `yolo11s-640`, `fasterrcnn-resnet50`.
Local ResNet GPU smoke passed empty targets, two steps and exact checkpoint reload.
All three fits completed, exit 0:0. Selected epochs are 29, 27 and 7 respectively.
VPN access was restored. A stale/NUL-filled management-node read of a training
CSV blocked packaging. CPU-only Slurm job **286305** read the originals on the
training node, verified complete curves/checkpoints and produced
`completed-artifacts-node.tar.gz`, SHA256
`31b86fe156e70d839daa90389030c0f836046682a2ae8d56dfed5dd7ca489a94`.
The incomplete local retrieval is preserved under `retrieval-first-attempt/`;
no remote source file was corrected or sanitized. The polling collector was stopped.

Local `development-run4/` contains the combined eight-detector comparison,
`model-scorecard.png` and short-target PNGs. Common-adapter Run 4 liquid AP50–95:
53.81% (nano 960), 53.66% (small 640), 53.50% (ResNet). The 18 short supplied
targets matched at .35 are 15, 15 and 17 respectively, versus baseline 15/18.
Run 4 has **no <=32px target**; these results cannot measure first-visible pools.
See the [completed verification](setup/verification/early-liquid-results-20261020.json).

SAM2.1 Tiny was tested with liquid-box, vial-box and lower-vial prompts on 30
Run 4 stills. Liquid prompts follow lower colored regions in inspected examples;
vial-only prompts outline glass even on label-absence cases. All 75 raw masks
and 30 overlays are under `prompted-liquid-20261020/run4-probe`. The
[SAM pipeline plan](setup/vision/SAM-LIQUID.md) separates presence/localization,
mask refinement and measurement QA. SAM3 remains an untested concept-prompt arm
requiring authorized checkpoint access. No human masks or onset labels were made.

Next: compare current-frame liquid prompts from the stronger detectors, then
forward-only mask propagation with resets per vial. Human body/stream review and
reviewed first-visible-liquid timing remain separate required evaluation inputs.
Use `envs/vision-alternatives-windows/Scripts/python.exe` locally. Do not rerun
completed fits/evaluations or use Run 5 to select settings.

## Annotation review: next work

Portable package:
`results/annotation-review-20261019/review-handoff-v1.zip`, also published in the
[private v0.2.0 release](https://github.com/ALL-Applied-AI-Network/msoe-ai-club-rockwell-automation-nvidia-hackath-austin-koske-s-team-8e90f213c6/releases/tag/annotation-review-v0.2.0).
From an extracted package, reviewers run `python launch_review.py --reviewer NAME`.
Each gets a separate workspace. Review Aleyna/Zoe/Reagan's 32/32/31 Run 3 stills
first; 40 recipe-navigation and 44 QA frames are follow-up cases.

Collect each correction JSON and uncertain cases without merging divergent
histories. Approved boxes and timing remain the data team's decisions. Keep timing
CSV/mappings separate from spatial corrections; never infer still/video alignment
from assumed 30 fps. QA rejection reasons are diagnostic context.
The original labels at `results/typesafe-box-all-20261018` remain read-only.

## Integration verification

Passed: 12 annotation-review tests; 56 replay tests using `data-windows`; two
specialist-data tests using `vision-alternatives-windows`; annotation JavaScript
syntax check. Follow-up: seven liquid-focused tests, compilation/shell syntax,
full four-arm media verification and ResNet GPU smoke passed. WSL's bare Python
lacks video dependencies; use the existing Windows environments for replay/vision.
No approved labels, physical-onset accuracy or detector promotion is claimed.
