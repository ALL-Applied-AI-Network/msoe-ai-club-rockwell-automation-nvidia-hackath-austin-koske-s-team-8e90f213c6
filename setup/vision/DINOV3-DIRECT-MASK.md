# SAM-free DINOv3 fluid segmentation pilot — October 24

This experiment replaces the liquid detector and SAM refinement with a direct
liquid-body mask and presence model. It does not replace vial detection,
measurement QA, operator review or the application defaults. GPU fitting and
inference run on the GB10; local work uses CPU only.

## Architecture

```text
Source RGB, native640x480
  ├─ frozen synthetic-trained YOLO11n → vial boxes for unchanged measurement QA
  └─ fixed station ROI112x192
       → generic DINOv3 ConvNeXt Small, frozen
       → four feature scales, projected to32 channels and fused
       → native RGB detail stem16 channels
       ├─ full ROI liquid-body mask
       └─ liquid-presence probability
             → threshold0.5 + largest component
             → unchanged sampling QA on ORIGINAL RGB
             → qualified color sample or measurement unavailable
```

The pilot uses a fixed camera ROI, not a learned vial crop. A production adapter
still needs validated camera geometry, reference/reset handling and source frame
identity. The model's gain-corrected input is never substituted for original RGB
when measuring color.

Two inputs are tested. `rgb` uses only the current frame; `paired` uses current
and reference frames, background-anchor gain correction, and signed/absolute
feature differences. The paired branch still depends on an appropriate aligned
reference. The first frame of each predefined real clip is an unreviewed
reference, not a certified empty vial.

The encoder has49,454,688 parameters; new trainable heads have70,018 (`rgb`) and
163,042 (`paired`). This is a custom fluid decoder, not Meta's published
segmentation checkpoint. The previous DINOv3/Faster R-CNN CSI-trained detection
head is never loaded. The generic pretrained backbone is disclosed and pinned.

The [official DINOv3 examples](https://github.com/facebookresearch/dinov3#getting-started)
include training a foreground segmentation head on frozen features. The cached
ConvNeXt Small was used to avoid a new download and to obtain multiscale features;
it is not the smallest DINOv3 encoder. The official model table lists ViT-S/16 at
21M and ConvNeXt Tiny at29M. They remain untested here. Our existing SAM model is
already [SAM2.1 Hiera Tiny,38.9M](https://github.com/facebookresearch/sam2#model-description).
Model size alone does not establish pipeline latency or fluid-mask quality.
[SAM3](https://github.com/facebookresearch/sam3) supplies broader concept-prompt
segmentation/tracking; it is not benchmarked by this pilot.

## Frozen protocol

- Existing native synthetic images/body geometry masks, source hashes unchanged:
  144 training images in36 families and48 development images in12 disjoint
  families. Empty, stream-only, early pool and established pool are represented.
- No real CSI imagery enters training, checkpoint selection or threshold tuning.
  No Run5 use. No new scene/CAD edits or human-approved labels.
- Frozen generic encoder; AdamW heads only, lr0.001, weight decay0.0001, batch8,
  60 epochs, seed19. Same exposure/noise/reference-jitter augmentation as U-Net.
- Mask BCE plus positive Dice and presence BCE. Select every5 epochs using
  synthetic development presence accuracy plus positive geometric IoU, earliest
  tie. Selected development scores are not independent test performance.
- Both fits finish and model hashes are frozen before accessing real clips.
  The primary DINO arm is also chosen by synthetic development score only.
- Matched real transfer: same264 exact RGB/native PTS frames from six previously
  inspected Run4 clips, frozen U-Net comparator, same synthetic vial detector,
  unchanged QA code/configuration. Every arm performs fresh vial inference.
- Full264-frame warm-up; two measured passes, reversing arm order. GPU-synchronized
  timing includes vial inference, mask/presence and QA. It excludes video decode,
  transport, rendering, serialization and browser latency.

Real mask/presence/onset accuracy remains unavailable until independent human
review. Synthetic body geometry is supervision, not reviewed optical truth.
More QA-eligible frames cannot establish correct liquid masks or recipe quality.

## Completed results

Both60-epoch fits completed on GB10. Both select epoch50 using synthetic
development only; paired is the primary DINO arm by the same development rule.

| Model | Synthetic development presence | Positive geometric mask IoU | Real proposals /264 | Real QA samples /264 | CV median /p95 ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| Frozen temporal U-Net | 48/48 | 89.33% | 189 | 105 | 16.37 /24.86 |
| DINOv3 RGB | 47/48 | 76.21% | 264 | 172 | 29.73 /31.65 |
| DINOv3 paired | 47/48 | 82.69% | 206 | 180 | 29.43 /31.63 |

Development IoU is a mean over24 synthetic positive images, before the presence
gate; it is not real segmentation accuracy. The encoder/head architecture does
not beat the U-Net on this selected synthetic metric. Paired DINO nevertheless
supplies77 new real QA samples while losing2 compared with U-Net, across the
same264 development frames. RGB gains84 but loses17 and proposes liquid on every
frame: presence transfer is a concern, not an established early-detection win.

All three arms repeat masks, presence probabilities and QA decisions exactly;
all792 across-arm vial comparisons match. The U-Net reproduces105/264 from the
prior study. Each arm has528 measured CV samples after264 warm-up frames.

For context, the earlier temporal→SAM refinement supplied185/264 QA samples.
Its72.62ms was incremental with upstream outputs cached, not a full pipeline
benchmark; it must not be compared as if it measured this trial's full CV path.
The earlier full synthetic R-CNN→SAM system supplied210/264 at99.84ms median.
These are prior executions, not a simultaneous new SAM run. None establishes
correct optical masks; the paired DINO result supports a SAM-free candidate for
human review and integration testing, not automatic deployment.

### Visual check and why the QA count rose

The aligned overlays expose substantial remaining errors. RGB-only supplies
high-confidence masks in upper glass in several inspected early frames. Paired
DINO localizes the bottom region more often but can extend far above the visible
colored pool/into the stream, especially the shown fill1 and fill5 examples.
U-Net looks tighter around several pools. These are visual observations, not
human-approved target masks or accuracy scores. Do not present180/264 as a proven
segmentation improvement or a solved early-onset detector.

An independent output audit finds that **all77 paired QA gains coincide with
U-Net's `liquid_missing_or_ambiguous_inside_vial` rejection**. In76 of those frames,
U-Net already supplied a liquid proposal. The gains therefore do not mean77 newly
detected liquid frames. The unchanged QA requires at least90% of the liquid
bounding box inside the vial box. A larger/taller prediction can satisfy this
test while including glass; this gate is not an optical segmentation validator.

Concrete same-frame example, native frame414: vial box is`[270,64,343,420]`.
U-Net's liquid box`[283,363,344,426]` has88.99% containment and is rejected.
Paired DINO's`[287,306,340,422]` has98.28% containment and passes, although its
overlay extends visibly upward beyond the bottom pool. This motivates separate
review of the vial-base geometry and liquid-body target, not weakening QA or
choosing a model from availability alone. Audit:`visual-audit-v1/audit.json`.

Next: shared independent visible-pool/onset review, more synthetic boundary,
glass and stream-only coverage, then synthetic-selected follow-up models and
source-matched evaluation. Keep U-Net as the fast baseline and paired DINO as
the new SAM-free comparison branch. A reference-independent winner is not yet
established by the RGB-only pilot.

## Artifact locations

Local root: `C:/Users/Austin/ai-club-tools/results/dinov3-direct-mask-20261024`.
Remote root: `~/ai-club-tools/experiments/dinov3-direct-mask-20261024-v1`.

- `summary-v1/sam-free-comparison.mp4`: all three branches, same six clips.
- `summary-v1/dino-synthetic-selected.mp4`: paired DINO alone, selected using
  synthetic development only, including its real-transfer errors.
- `summary-v1/early-later-comparison.png`: raw source plus all three proposals,
  two predefined source frames per clip.
- `summary-v1/verification.json`: recomputed counts/timing, model hashes and media.
- `gb10-results/comparison-v1`: all264 exact source PNGs, raw/full component and
  gated masks, QA evidence, two measured passes, immutable helper snapshots.

Both MP4s fully decode240 frames at5fps (48seconds). The local summary checks
280 remote artifact hashes,264 source RGB identities and792 reference joins;
it recomputes development and real metrics. Both decoder checkpoints reload
exactly in the CPU smoke, and all eight new experiment helpers compile.

Input archive hash/size and frozen plan are checked before extraction. The
launcher refuses an existing run and retains a PID; the trainer takes the shared
GB10 GPU lock. Preserve failed/partial outputs and use fresh evaluation paths.

Scripts are `prepare_dinov3_direct_mask.py`, `bootstrap_dinov3_direct_mask.py`,
`dinov3_liquid_mask.py`, `train_dinov3_liquid_mask.py`,
`smoke_dinov3_liquid_mask.py`, `compare_dinov3_direct_mask.py` and
`summarize_dinov3_direct_mask.py`, with `collect_dinov3_direct_mask.py` for remote
verification/packaging. The CPU smoke checks native
mask/presence shapes, frozen encoder behavior and exact head checkpoint reload.

Preserve Srijan's integration ownership and the operator team's API/client work.
No detector or segmentation experiment is automatically promoted by this report.
