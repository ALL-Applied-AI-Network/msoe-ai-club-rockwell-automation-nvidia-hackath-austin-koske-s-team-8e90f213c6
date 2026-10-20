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
| Detector experiments | `setup/vision/`; [frozen protocol](setup/vision/LIQUID-SPECIALISTS.md) and [detailed commands](setup/vision/CLI-HANDOFF-20261020.md) | Four fits complete; final DINO evaluation and assembly outstanding |

Local assets are under `C:/Users/Austin/ai-club-tools/` (`/mnt/c/Users/Austin/ai-club-tools/`
in WSL). Results, environments and source labels stay outside Git. The ChatGPT
project mirror is reference data, not a checkout.

Existing untracked `analysis/`, `datasets/`, `docs/`, `experiments/`, `handoffs/`
and `team_plan.md` are preserved in place and excluded **locally** using
`.git/info/exclude`. They are not part of this integration; use `git status
--short --ignored` to see them. Recovery snapshots, including the retired
worktree's ignored build files, are in
`C:/Users/Austin/ai-club-tools/recovery/repo-consolidation-20261020T203829Z/`.

## Detector experiment: next work

Root: `C:/Users/Austin/ai-club-tools/results/liquid-specialists-20261020`.
Live scheduler verification during consolidation confirmed array **286264**, all
four jobs **COMPLETED, exit 0:0**. Collector log says the bundle is ready.

`completed-artifacts.tar.gz` and its checksum are now local and verified:
`521608a3d63b6de0458914bcc91c421899acc9addfbacb3312f892d22271cc2f`.
The bundle's existing model files matched local bytes; missing DINO model files
were restored. Local smoke artifacts were preserved separately from remote smoke
artifacts in the bundle. No replacement jobs or evaluations were launched.

| Arm | Completed epochs | Selected Run 4 epoch | Local evaluation |
| --- | ---: | ---: | --- |
| `yolo-all` | 29 | 19 | `evaluation-yolo/yolo-all` complete |
| `yolo-focused` | 28 | 18 | `evaluation-yolo/yolo-focused` complete |
| `rtdetr-all` | 28 | 18 | `evaluation-rtdetr/rtdetr-all` complete |
| `dinov3-all` | 25 | 15 | Pending; final model now in `models/dinov3-all` |

Use `envs/vision-alternatives-windows/Scripts/python.exe` for final evaluation.
Follow the **DINO evaluation and collect commands** in the detailed handoff;
bundle retrieval/extraction and training are already done. Use a fresh output
directory if a partial evaluation exists. Then inspect local MP4/PNG outputs,
assemble all four arms, and record hashes, decoded frame counts and limitations.
Do not promote a model automatically. Teacher agreement is not reviewed accuracy
or measurement QA; Run 5 has already been inspected and is not a blind test.

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
syntax check. WSL's bare Python lacks video dependencies, so use the existing
Windows environments for replay/vision. No new scientific validation, approved
labels, detector promotion or four-arm evaluation is claimed by this cleanup.
