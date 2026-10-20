# CLI / Cursor handoff — October 20, 2026

Continue the liquid-specialist and non-YOLO experiments already in progress.
Do not start replacement training jobs before inspecting the existing jobs and
artifacts. This is an in-progress handoff, not a completed four-model report.

## Workspace and branch

Repository / Cursor folder / CLI working directory:

```text
C:/git/FDS-Dark/ALL-Applied-AI-Network/msoe-ai-club-rockwell-automation-nvidia-hackath-austin-koske-s-team-8e90f213c6
```

Branch: `austin/recipe-aware-replay`.
Last completed experiment commit before this handoff: `f822c90`, already pushed.
The handoff commit adds the new experiment scripts and these notes; inspect
`git log -1` for its hash. Commit/push scoped follow-up work on this same branch.

Do not sweep unrelated untracked directories into commits: `analysis/`,
`datasets/`, `docs/`, `experiments/`, `handoffs/`, or `team_plan.md`.
The ChatGPT project mirror under `C:/Users/Austin/.codex/.chatgpt-projects/`
is not the repository. Its synced `sources/` files are read-only reference data.

User preferences / boundaries:

- Local MP4/PNG files and direct paths. Do not launch web previews or browser servers.
- Rosie uses **Singularity 3.10, never Docker**. GPU work only through Slurm.
- Keep Run 5 out of training, checkpoint selection, threshold tuning and sampling decisions.
- Preserve the deployed detector, measurement QA, app defaults and source labels.
- No industrial equipment/network/camera connections.
- Reviewed boxes, timing/windows, synthetic rendering, workflow and CSI coordination
  belong to the mentees. Do not independently adjudicate or replace their work.
- Austin authorized using existing vial-only labels as **provisional negatives**.
  They are not reviewed/confirmed-empty examples.
- No subagents/delegation unless the user explicitly authorizes them.

## Active processes: inspect first

Rosie SSH alias: `msoe-bastion` (`dh-mgmt1.hpc.msoe.edu`). Campus VPN required.
Use SSH config keys, not credentials copied into source files.

```powershell
ssh -o BatchMode=yes -o GSSAPIAuthentication=no -o ConnectTimeout=12 msoe-bastion 'sacct -X -j 286264 -n -P -o JobID,State,ExitCode,Elapsed,NodeList'
```

Array **286264**, four independent H100 fits. Last checked:

| Job | Arm | State | Completed / selected epoch |
| --- | --- | --- | --- |
| `286264_0` | `yolo-all` | COMPLETED, exit 0 | 29 / 19 |
| `286264_1` | `yolo-focused` | COMPLETED, exit 0 | 28 / 18 |
| `286264_2` | `rtdetr-all` | COMPLETED, exit 0 | 28 / 18 |
| `286264_3` | `dinov3-all` | RUNNING at handoff | Latest clean CSV read: epoch 23; recheck |

Remote experiment:
`/home/ad.msoe.edu/koskea/ai-club-tools/results/liquid-specialists-20261020`.
Use `$HOME/ai-club-tools/results/liquid-specialists-20261020` in remote commands.

Detached collector PID **2861044** waits for four successful job exits, then
creates `completed-artifacts.tar.gz` and its `.sha256` file in that directory.
Its log is `collection.log`; scheduler snapshots are `scheduler-status.txt`.
Training/collection survive VPN disconnection. The collector has a 135-minute
limit and does not resubmit jobs. A failed job produces no successful bundle.

An earlier array **286263 was canceled while pending**: unpacking began before
its 640 MB input transfer completed. It never supplied a fitted model. The full
transfer was then gzip-verified, unpacked again, and every frozen helper/plan hash
verified before submitting 286264. **Wait for transfers to finish before extracting.**

Some reads of actively written Rosie log/CSV files returned NUL bytes. Later
reads returned valid data. Do not change original logs or silently sanitize
final evidence; re-read completed files and validate CSV/model metadata.

Local RT-DETR evaluation was started immediately before the conversation was
interrupted, survived that interruption, and **completed while this handoff was
being written**. Both completion markers below now exist. Its former processes:

- PID **25616**: venv Python launcher.
- PID **27620**, parent 25616: actual Python process.

These were a launcher/child pair, not two separate evaluations; both have exited. Command:
`setup/vision/evaluate_liquid_specialists.py ... --out .../evaluation-rtdetr --arms rtdetr-all`.
Do not launch a duplicate. The old desktop tool-session IDs are not needed.

Completion requires both:

```text
<local root>/evaluation-rtdetr/rtdetr-all/completed-arm.json
<local root>/evaluation-rtdetr/comparison.json
```

If a future export is interrupted, preserve its partial directory and rerun to
a **fresh** path. Scripts refuse existing output directories. Remote Slurm jobs
are independent of the desktop app and VPN connection.

## Local assets and environments

Experiment root:
`C:/Users/Austin/ai-club-tools/results/liquid-specialists-20261020`.

Already present locally:

- Frozen `experiment-plan.json`, `preparation.json`, training helper snapshots.
- Derived `data/all/` and `data/focused/`, plus pinned `pretrained/` assets.
- Completed `models/yolo-all`, `models/yolo-focused`, `models/rtdetr-all` weights,
  metadata and training CSV/args.
- Completed `evaluation-yolo/` for both YOLO arms: six verified local MP4 files.
- `evaluation-rtdetr/`: scoring and three fully decoded video exports complete.
- `dino-smoke/`: initial local GPU smoke artifacts, not the final trained model.

Python executables:

```text
C:/Users/Austin/ai-club-tools/envs/data-windows/Scripts/python.exe
C:/Users/Austin/ai-club-tools/envs/vision-windows/Scripts/python.exe
C:/Users/Austin/ai-club-tools/envs/vision-alternatives-windows/Scripts/python.exe
```

Use **vision-alternatives-windows** for final evaluation/collection. It installs
Transformers 4.57.1 and reuses existing vision packages via `team_vision.pth`.
Local Torch is 2.11.0+cu128; Ultralytics 8.4.182. RTX 4080 Super is available.
Rosie uses Torch 2.8.0a0+5228986c39.nv25.05, Ultralytics 8.4.182, H100 80 GB;
DINOv3 uses the isolated `vision-alternatives-rosie` environment and Transformers
4.57.1. The existing shared vision environment was not replaced.

Model assets were downloaded or copied from already cached authorized weights.
No need to request DINOv3 access or download again: the exact backbone is local
and remote in `pretrained/dinov3-convnext-small/`.

Other paths used by evaluation:

```text
Baseline PT:
C:/Users/Austin/ai-club-tools/results/crop-clahe-detector-20261019/baseline-best.pt
Teacher images/boxes:
C:/Users/Austin/ai-club-tools/results/typesafe-box-all-20261018
Run 5 image-only event/source cache:
C:/Users/Austin/ai-club-tools/results/tracked-liquid-20261019/cache/run5-8bdd8cad2977ee74e6a3.json
Recording directory, relative to repository:
datasets/dataset2/csi-vial-fill-dataset
Previous longer two-class reference root:
C:/Users/Austin/ai-club-tools/results/epoch-augmentation-20261020
```

## Experiment design and interpretation

Read `LIQUID-SPECIALISTS.md` and the frozen plan for full details.

- One liquid class derived by removing vial targets and remapping old liquid
  class 1 to class 0. Coordinates unchanged. Original labels untouched.
- `yolo-all`: 708 positive / 734 provisional-negative training images.
- `yolo-focused`: 1,152 positive / 290 provisional-negative samples. Positive
  repeats and paired existing translations keep the same 1,442-sample epoch.
  It has 499 distinct source originals versus 721 for the same-images arm.
- Runs 1–3 train; identical Run 4 development set: 89 positive / 68 provisional-negative.
- Maximum 50 epochs, patience 10, seed 19. Select using **Run 4 liquid AP50–95**.
- `rtdetr-all`: actual non-YOLO RT-DETR-L with a COCO initialization.
- `dinov3-all`: actual DINOv3 ConvNeXt Small frozen backbone, trainable FPN256 +
  standard TorchVision Faster R-CNN. COCO RPN/ROI box head initialization, new
  background/liquid predictor. **Custom detector**, not Meta's published 7B detector.
- DINOv3 has its own native normalization, batch 8, AdamW/cosine schedule and
  no online augmentation beyond existing translated images. This is a family
  pilot, not a pure architecture-only ablation or an optimized benchmark.
- DINOv3's local smoke test passed empty targets, two training steps and exact
  checkpoint reload checks. Its Rosie GPU smoke runs before the real fit.
- Every comparison uses the **same baseline vial proposals on both sides**.
  Blue boxes do not score improved tracking. Only the liquid branch changes.
- Display confidence stays 0.35; shared final liquid NMS IoU 0.45. RT-DETR is
  normally NMS-free, but this diagnostic applies shared final NMS explicitly.
- Scores are not calibrated across families. Interpret AP and PR curves,
  not just recall at 0.35 or how often a green box appears.

Run 5 scoring so far (unreviewed teacher agreement; **not measurement QA**):

| Model | Liquid AP50–95 | Precision at .35 | Recall at .35 | Unmatched boxes at IoU .5 |
| --- | ---: | ---: | ---: | ---: |
| Deployed baseline | 49.18% | 96.80% | 72.64% | 13 |
| Previous longer two-class anchor | 48.85% | 96.40% | 79.11% | 16 |
| `yolo-all` | 49.28% | 86.09% | 90.39% | 79 |
| `yolo-focused` | 47.46% | 96.93% | 75.79% | 13 |
| `rtdetr-all` | 50.21% | 90.45% | 89.28% | 51 |
| `dinov3-all` | Pending | Pending | Pending | Pending |

Each score uses the same 816 frames / 541 liquid targets; 29 unknown records
excluded. Neither YOLO specialist makes a liquid prediction on the 275 scored
frames without a supplied liquid target. Thus the extra `yolo-all` unmatched
boxes occur in positive frames: 54 frames have multiple liquid boxes, and 78
frames contain 79 unmatched boxes. `yolo-focused` has three multiple-box frames
and 13 unmatched boxes in 13 frames. Diagnostics are in each arm's
`unmatched-box-review.json`. This describes disagreements, not human adjudication.

The previous longer two-class anchor selected combined Run 4 AP, while new
specialists select liquid-only AP. It is a useful reference, not a perfectly
isolated causal control. Run 5 has already been inspected repeatedly and is
not a blind final test. Do not choose new hyperparameters using these scores.

## Next steps, in order

1. Inspect the remaining remote DINOv3 job. Let it finish; do not duplicate it.
2. Retrieve and hash-verify the completed Rosie bundle once all four fits succeed.
3. Evaluate the final DINOv3 checkpoint locally. RT-DETR exports are already complete.
4. Assemble all four completed arm directories with `collect_liquid_specialists.py`.
5. Inspect local videos, problem-frame PNG and precision–recall PNG. Check actual
   localization, oversized/duplicate boxes, pale liquid and early-pour examples.
6. Write final verification JSON and update protocol/README with actual results,
   model hashes, selected epochs, source identities, decoded video counts and limits.
7. Run focused tests/compilation and commit/push scoped follow-up changes. Return
   local media links and the conclusion; do not promote a detector automatically.

PowerShell variables from repository root:

```powershell
$vision = 'C:/Users/Austin/ai-club-tools/envs/vision-alternatives-windows/Scripts/python.exe'
$root = 'C:/Users/Austin/ai-club-tools/results/liquid-specialists-20261020'
$baseline = 'C:/Users/Austin/ai-club-tools/results/crop-clahe-detector-20261019/baseline-best.pt'
$teacher = 'C:/Users/Austin/ai-club-tools/results/typesafe-box-all-20261018'
$data = 'datasets/dataset2/csi-vial-fill-dataset'
$cache = 'C:/Users/Austin/ai-club-tools/results/tracked-liquid-20261019/cache/run5-8bdd8cad2977ee74e6a3.json'
$cacheHash = 'aff504002184d856ac06451f1730677eace2e840b89110c2cb6c0af39a8850d8'
```

Retrieve after remote completion (wait for each command; check its exit):

```powershell
scp -q msoe-bastion:ai-club-tools/results/liquid-specialists-20261020/completed-artifacts.tar.gz "$root/completed-artifacts.tar.gz"
if ($LASTEXITCODE -ne 0) { throw 'Bundle transfer failed' }
scp -q msoe-bastion:ai-club-tools/results/liquid-specialists-20261020/completed-artifacts.tar.gz.sha256 "$root/completed-artifacts.tar.gz.sha256"
if ($LASTEXITCODE -ne 0) { throw 'Hash transfer failed' }
$expectedHash = ((Get-Content "$root/completed-artifacts.tar.gz.sha256") -split '\s+')[0]
if ((Get-FileHash "$root/completed-artifacts.tar.gz" -Algorithm SHA256).Hash.ToLower() -ne $expectedHash) { throw 'Bundle hash mismatch' }
tar -xzf "$root/completed-artifacts.tar.gz" -C "$root"
if ($LASTEXITCODE -ne 0) { throw 'Extraction failed' }
```

Evaluate DINOv3 to a fresh directory:

```powershell
& $vision setup/vision/evaluate_liquid_specialists.py --experiment $root --baseline $baseline --teacher $teacher --data $data --cache $cache --cache-sha256 $cacheHash --out "$root/evaluation-dino" --device cuda:0 --arms dinov3-all
```

Use the completed RT-DETR directory below. If its artifacts later fail integrity
checks, rerun with `--arms rtdetr-all` to a fresh output and adjust that input.

Collect (this helper is written/compiles but has not yet run with all four inputs):

```powershell
& $vision setup/vision/collect_liquid_specialists.py --experiment $root --data $data --out "$root/evaluation" --two-class-reference C:/Users/Austin/ai-club-tools/results/epoch-augmentation-20261020 --inputs "$root/evaluation-yolo/yolo-all" "$root/evaluation-yolo/yolo-focused" "$root/evaluation-rtdetr/rtdetr-all" "$root/evaluation-dino/dinov3-all"
```

Outputs: `evaluation/comparison.json`, `liquid-precision-recall.png`,
`problem-frame-comparison.png`, and four arm directories containing 12 MP4s.
Each arm has a full recording at 5 fps (8,491 frames), early neighborhoods at
30 fps (968 frames), and two complete fills (364 frames at 5 fps). The evaluator
fully decodes each video before writing `completed-arm.json`; the collector
checks those exact video hashes before copying. Fix any actual collector issues
without changing frozen training scripts, source labels or QA.

Checks already passed:

```powershell
& $vision -m unittest discover -s setup/vision -p test_liquid_specialists.py -v
& $vision -m py_compile setup/vision/liquid_specialists_data.py setup/vision/train_liquid_specialists.py setup/vision/dino_liquid.py setup/vision/evaluate_liquid_specialists.py setup/vision/collect_liquid_specialists.py
```

Two focused data tests passed; local DINOv3 GPU smoke passed; both YOLO and
RT-DETR full evaluations completed with nine fully decoded videos and identical
fixed-vial metrics. No final DINOv3 evaluation or four-arm assembly has been claimed yet.

## Files added for this work

All code lives under `setup/vision/`:

- `liquid_specialists_data.py`: immutable derivation, sampling and input/weight pins.
- `train_liquid_specialists.py`: Ultralytics arms, frozen-backbone detector fit and smoke.
- `dino_liquid.py`: custom detector architecture and trainable-head checkpoint loading.
- `train-liquid-specialists-rosie.sbatch`: four-arm Slurm job, DINO GPU smoke first.
- `bootstrap-alternatives-rosie.sh`: isolated Transformers environment.
- `collect-liquid-specialists-rosie.sh`: detached successful-fit artifact packaging.
- `evaluate_liquid_specialists.py`: fixed-vial adapter, scoring, local MP4 exports.
- `collect_liquid_specialists.py`: final assembly, PR plot, problem-frame comparison.
- `test_liquid_specialists.py`, `LIQUID-SPECIALISTS.md`, and this handoff.

Actual executed helper bytes are retained in the experiment roots and pinned by
`preparation.json`; source snapshots normalized to LF before fitting. Do not edit
them in place while jobs run. Changes to evaluation/collection documentation are
separate from frozen fitting. No new dataset, weights or large video is committed.
