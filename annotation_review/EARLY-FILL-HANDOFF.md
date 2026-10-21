# Early liquid review: Run 4 development batch

This is a new, unreviewed batch for Aleyna, Zoe and Reagan. It does not replace
the existing Run 3 assignments or modify any prior review workspace. Cases remain
unassigned so the team can schedule them alongside its current milestone.

The batch has six eight-second source clips and 72 exact, lossless source PNGs.
They surround six image-derived stream-start candidates from Run 4. Those
candidates select neighborhoods to inspect; they do not establish when liquid
first becomes visible. No approved onset, presence, empty-vial or mask labels
have been produced.

## Review in this order

1. Agree on a written visibility/mask convention using a few shared examples.
   Decide how to handle a thin basal pool, reflections, the incoming stream,
   the glass base, bubbles and uncertain liquid boundaries. Record the convention
   version with each returned review. The data team owns these decisions.
2. Watch the raw clips before inspecting model outputs. The clips have no model
   overlays. Identify the first definitely visible pooled liquid and the last
   definitely absent frame, if either can be established.
3. Use zero-based **source decoded frame indices**, rather than clip-local frame
   numbers. Each `clips/*-source-frames.json` maps clip frames to the source index,
   native PTS and time base. Its array position is the clip-local frame index.
4. If liquid is already visible at the first clip frame, mark the onset
   `left_censored`. If it never becomes definitely visible, use `right_censored`
   or `uncertain` with notes. Request a longer source neighborhood when needed;
   do not use the proposed anchor or first model detection as the answer.
5. Review source PNGs for visible pooled-liquid geometry. Return full-source
   640×480 binary PNG masks (0 background, 255 liquid) using the agreed convention.
   The incoming stream and vial glass should be addressed explicitly in that
   convention. Ambiguous masks remain uncertain.

The 72 images are a sparse geometry sample. They cannot by themselves establish
an exact onset between images; use the frame-indexed source clip/full video.
This is inspected development footage, not a blind final test. Pale/clear-liquid
generalization cannot be established merely by reviewing these six examples.

## Return formats

- Fill `onset-review.csv` or `onset-review.json`: source-video hash, case ID,
  reviewed first-visible frame, last definitely absent frame, reviewer, status,
  convention version and notes. Leave unresolved frame values empty/null.
- Fill `mask-review.json`: image/source hashes, presence status, reviewer,
  convention version, notes and relative path/hash of each reviewed binary PNG.
  Keep masks and the completed manifest together in a new export directory.
- Onset `review_status` values are `reviewed`, `uncertain`, `left_censored`,
  `right_censored` or `unreviewed`. Mask `presence_status` values are `present`,
  `confirmed_absent`, `uncertain` or `unreviewed`. A null mask means unreviewed
  or unresolved; it never means confirmed empty. A confirmed-empty spatial example
  requires `confirmed_absent`, an identified reviewer and an all-zero mask.

Keep every reviewer's original return intact. Agreement/adjudication is a separate
data-team step. Do not silently combine conflicting onset or mask decisions.
Spatial masks do not approve recipes, timing synchronization, product acceptance
or physical vial identities.

## Optional existing box editor

Python 3.11+ is sufficient to launch the included box editor. From this extracted
package, use a fresh workspace dedicated to this batch, for example:

```text
python launch_review.py --reviewer Aleyna --workspace ../early-fill-Aleyna-v1
```

Then open `http://127.0.0.1:8775`, choose your reviewer name and select
**Initial vial-fill batch**, with **Assigned: Everyone**. This package's initial
batch contains the 72 `run4_early_liquid` cases. Replace the reviewer/workspace
for Zoe or Reagan.
The preparer does not launch a server automatically.

The existing editor supports box corrections, not pixel-mask drawing or onset
adjudication. Use it for any helpful spatial box review; return onset/mask
decisions in the separate formats above. New video-frame cases begin without
machine candidates. The saved prelabel metadata is included for compatibility;
viewing unrelated stills requires the full dataset via `--dataset FULL_DATASET`.

The package includes no trained weights, corrected team labels, model masks,
synthetic images or Run 5 source files. `package-hashes.json` identifies the exact
distributed files. Keep the extracted source package immutable and export reviews
to a separate folder.
