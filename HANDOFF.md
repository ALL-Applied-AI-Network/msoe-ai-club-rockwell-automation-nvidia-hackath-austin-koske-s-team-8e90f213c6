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
| Pipeline video | [Offline application](setup/vision/SAM-LIQUID.md#offline-pipeline-video--october-20) | 60 s Run 4 video; detector → SAM → existing per-frame sampling gates; 300 frames verified |
| Early pool prompts | [Paired trial](setup/vision/SAM-LIQUID.md#pool-prompt-comparison-and-causal-bridge--october-20) | Localized box/points plus causal bridge; before/after videos complete; QA unchanged |
| Innovation lab handoff | `results/innovation-lab-brief-20261020/` under the local tools root | Six-page architecture PDF, editable Word brief, single-panel final MP4s and code-free sharing ZIP |
| Team data + student brief | `results/team-data-20261020-v1/` under the local tools root | Verified data archives, source manifests, YOLO/COCO starters and three-page architecture assignment PDF |

Local assets are under `C:/Users/Austin/ai-club-tools/` (`/mnt/c/Users/Austin/ai-club-tools/`
in WSL). Results, environments and source labels stay outside Git. The ChatGPT
project mirror is reference data, not a checkout.

The innovation lab package is
`C:/Users/Austin/ai-club-tools/results/innovation-lab-brief-20261020/Innovation-Lab-Liquid-Pipeline-Handoff.zip`.
Its `share/` source directory contains only PDF/DOCX, two final-solution H.264
videos (60 s and a 12 s early-fill highlight), two PNGs and a readme. No source
code, weights or labels are included. The videos reuse the verified causal-box
experiment, with source-pixel/PTS checks and full decoding of all 360 output
frames. Builder/provenance files stay separately in `work/`, outside the ZIP.

For the **student-facing team release**, use the newer
`results/team-data-20261020-v1/Team-Architecture-Implementation-Brief.pdf`.
It outlines roles, interfaces and evaluation questions without the final prompt,
threshold or temporal-handling recipe. The earlier six-page innovation brief is
the more detailed reference and is not included in this student data release.
`Team-Data-Guide.pdf` separately explains inventory, formats, splits and gaps.

`archives/Team-Core-Data.zip` (281,264,726 bytes) combines components 00–03:
metadata/handouts, detector starter, raw vial Runs 1–4 and 300 source-aligned
Run 4 PNGs with unreviewed detector inputs. The 721-original/157-development
profile and optional 1,442-sample translated training profile both loaded through
Ultralytics; nine COCO detection exports also passed format/identity checks.
Other-camera context (04) and Run 5 diagnostics (05) remain separate archives.
Full catalog: 4,647 indexed JPEGs, 15 videos, 16,814 exported provisional boxes;
21 vial stills require manual review. No approved masks/onsets, final SAM masks,
prompt/state recipe, executable code, weights or reviewer workspaces are included.
Six focused packaging tests passed; archive CRC/SHA read-back checks and all 300
SAM PNG pixel/PTS checks passed. See `setup/verification/team-data-20261020.json`.
The repository entry point is now [team_handoff/README.md](team_handoff/README.md),
with PDFs, data and checksum links in the versioned
[team-data-v1.0.0 release](https://github.com/ALL-Applied-AI-Network/msoe-ai-club-rockwell-automation-nvidia-hackath-austin-koske-s-team-8e90f213c6/releases/tag/team-data-v1.0.0).
Release attachments keep the large data out of Git. No source labels/data were
modified; `main` remains untouched.

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

An offline application now uses ResNet liquid prompts with SAM2.1 and unchanged
per-frame geometry/measurement gates on the first 60 seconds of Run 4. Artifacts:
`results/liquid-pipeline-demo-20261020/render-v1/` (local MP4, posters, decoded
storyboard, source-aligned raw masks and provenance). All 300 video frames and
300 source RGB hashes/native PTS were checked. There are 49 per-frame sampling
passes; this is not full-window QA or an accuracy score. Oversized prompts can
include glass/upper structures. See the SAM protocol for exact counts and limits.

The paired prompt experiment now fixes the inspected early full-vial masks using
localized pool seeds/points plus a causal prompt-box bridge. Final local media:
`results/liquid-pipeline-demo-20261020/causal-box-comparison-v1/`.
`early-fill-before-after.mp4` covers source PTS 10–22 s; `before-after.mp4` covers
60 s. All three new experiment directories and their six MP4s are preserved.
Association-only did not fix oversized prompts; point-only still included lower
glass despite better coarse geometry counts. The final bridge used boxes only
0.2 s old at 13.4/13.8 s, with current seeds and no SAM mask propagation. This is
an inspected Run 4 pilot with a colored-pool prior, not onset/mask validation.

Next: additional development fills and pale/clear cases, other detector prompts,
then forward-only SAM mask propagation with resets per vial. Human body/stream review and
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
