# Visible liquid, masks and first appearance

The October 22 packet contains **72 exact Run 4 frames across six fill clips**.
They are real evaluation images. They must not enter synthetic-only training.
The temporal model supplies editable drafts, not approved labels or accuracy.

Local packet:
`C:/Users/Austin/ai-club-tools/results/synthetic-iterations-20261022/visible-pool-review-v2/`.
Open `review.html` in Chrome or Edge. The sibling ZIP is portable; extract it
before opening. No server, package installation, API or GPU is required.

## What a person supplies

| Label | Meaning | Required review |
| --- | --- | --- |
| Empty / no visible pool | No accumulated liquid is visibly present in the filling vial. An incoming stream alone does not count. | Explicit decision; saves a zero mask. Do not infer absence from a model's output. |
| Visible pool | Accumulated liquid is visible at the bottom or higher in the vial. | Outline visible liquid body/surface. Exclude stream, glass, holder, label and exterior reflections. |
| Uncertain | Presence or boundary cannot be determined reliably. | Explain the ambiguity; excluded from accuracy scoring. |
| First-visible interval | Last definitely absent frame, followed by first definitely present frame. | Review both endpoint stills, then save the interval. Sparse endpoints do not give exact onset. |

Agree on the first five images together. Inspect raw images before revealing a
draft; independently outline some cases without drafts to check anchoring bias.
Do not invent a mask through an occluder. If the visible-pool convention cannot
be applied consistently, record uncertain and resolve the convention with the
data team rather than silently changing target scope.

Use **Bottom zoom**, polygon drawing or paint/erase. **Load machine draft** does
not approve anything. Select a status and explicitly **Save human decision**.
Download JSON regularly and keep each reviewer's original export intact.
Browser storage is a convenience, not the handoff artifact. Matching imports
restore history; divergent histories and source mismatches are rejected.
Reviewer identity is self-reported. Team approval remains a separate human step.

The six clips are navigation context. Exact stills preserve native frame index,
PTS and time base, source video SHA256 and RGB/image hashes. Never convert
wall timestamps or browser playback time to native frame indices using assumed
30 fps. If an onset interval is too broad, request additional native frames
inside it rather than declaring its first positive endpoint exact onset.

## Output and evaluation

`visible-pool-human-review/v1` JSON retains append-only frame/onset events,
reviewer, notes, immutable source identities and draft-model provenance.
Masks use full-source **640 × 480**, row-major alternating RLE starting with a
zero run. A run of length zero at the beginning is valid. This differs from
replay's cropped foreground-run representation; do not interchange them.
Unreviewed suggestions remain in the original hash-pinned packet.

Normal human exports are eligible for team review, not automatically team
approved. `?demonstration=1` uses isolated browser storage and produces exports
ineligible for team labels; demonstration imports cannot enter normal review.
The browser verification artifacts deliberately contain demonstration geometry.

After real human review, score with the CPU data environment:

```text
python -m annotation_review.score_mask_review --probe PATH/gb10-results/mask-probe-v1 --packet PATH/visible-pool-review-v2 --review PATH/human-export.json --out PATH/fresh-human-evaluation
```

The scorer replays source-checked history, rejects demonstrations, excludes
uncertain/unreviewed frames and reports review coverage. Frozen thresholds are
0.5. It reports presence precision/recall, mask IoU/Dice on reviewed positives,
and interval-valued onset delay using native PTS. No checkpoint selection,
threshold tuning, training or app promotion occurs. The six correlated,
repeatedly inspected development clips cannot establish production accuracy.
These labels do not certify metrology or product quality.

The [matched SAM/temporal experiment](../setup/vision/SAM-TEMPORAL-COMPARISON.md)
uses the same72 exact source identities. One completed review can therefore
score its three arms and the optional temporal-prompted SAM refinement:

```text
python -m annotation_review.score_sam_temporal_review --comparison PATH/gb10-results/sam-comparison-verified-v2 --source-probe PATH/gb10-results/mask-probe-v1 --refinement PATH/gb10-results/temporal-sam-refinement-v1 --packet PATH/visible-pool-review-v2 --review PATH/human-export.json --out PATH/fresh-reviewed-comparison
```

This reports proposal precision/recall for every arm: SAM's lack of a proposal
is not a calibrated empty-vial decision. Missing masks on reviewed positives
count as misses and contribute zero positive-mask IoU. The comparison uses raw
visible-pool proposals, not inset sampling masks. No new label copy is required.

## Rebuilding and verification

```text
python -m annotation_review.mask_review --probe PATH/gb10-results/mask-probe-v1 --context PATH/early-fill-review-v1 --out PATH/fresh-packet
```

The standard-library builder checks the completed probe, every embedded RGB/mask
and all copied context clips/mappings against saved hashes. The standalone HTML
embeds images and drafts and reads only its neighboring context videos.
`verify_mask_review.py` exercises the offline page in headless Chromium with
GPU disabled: mask editing, explicit save, source/history checks, import/export,
onset intervals, mobile layout and demonstration separation. It never approves
human labels. The existing box editor is unchanged.
