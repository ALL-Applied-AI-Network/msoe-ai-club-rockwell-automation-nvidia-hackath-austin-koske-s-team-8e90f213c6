# Local human box review: Aleyna's group

The published label contract is [LABEL-CONTRACT.md](LABEL-CONTRACT.md). It defines
`vial`, `liquid_body`, and `stream`, with explicit present/absent/uncertain states,
visible-only boundaries, one active station vial/body, and separate falling
segments. Legacy `colored_liquid` is **not** automatically renamed or corrected.

## Ready-to-share pack, October 19

Final local folder:
`C:/Users/Austin/ai-club-tools/results/aleyna-label-review-20261019-final`

Share the whole `aleyna-label-review-20261019-final.zip` beside that folder
(22,661,060 bytes; SHA-256
`c129bc7ca5920b5a7266422ee181eb91199a89b87881d9d739bd2e3cf28e84c5`).
Extract the ZIP first and double-click `review.html`. This is an offline local
file with relative RGB images, not a web service. No install, account or network
is needed. The editor works with hand-drawn boxes and separate JSON downloads;
it has no accept/copy proposal or training-export function.

- 210 unique original 640 x 480 JPEGs, copied byte-for-byte from the frozen real
  inventory. No augmentation, synthesis, Run 5, new predictions or label edits.
- Runs 1, 2 and 3: 60 frames each, training-candidate role. Run 4: 30 development
  frames. All already have unreviewed teacher proposals; they are not truth.
- Aleyna, Zoe and Reagan: 70 assigned frames each. The 12 shared calibration
  frames are a subset of the same 210, reviewed independently before discussing
  disagreements. No extra unique images are introduced by shared review.
- 105 frames in 21 five-still chronological neighborhoods, selected using known
  Runs 1-3 RGB references and provisional presence/height transitions. These are
  **not adjudicated early pours or physical onset labels**.
- 38 additional bright/low-chroma candidates selected among legacy liquid-positive
  proposals, 12 additional short-pool candidates, 15 no-liquid-proposal cases,
  18 additional scope-review cases and 22 broad recording-coverage frames.
  Selection categories describe their primary sampling reason, not human labels.
  All 34 known top-above-y200 scope flags from the old Runs 1-4 inventory are
  included: 16 are already in the chronological neighborhoods and 18 added.
- Filename wall timestamps are retained with explicitly unverified clock status.
  MP4 frame/time fields remain null. Existing stills are roughly two seconds apart;
  this pack cannot establish sub-second first-droplet timing.

Review raw RGB first. Old proposal boxes are hidden initially and on navigation;
their original `.txt` files remain separate, byte-identical under `proposals/`.
The blank review has **zero boxes and 630 unreviewed class decisions**. Draw
visible objects manually, mark absence only when supported, explain uncertainty,
then submit each frame for Aleyna's adjudication. Save named JSONs frequently.
The tool validates completion but does not approve training or resolve semantics.

## Files and annotation-agent boundary

| Path within pack | Purpose |
| --- | --- |
| `START-HERE.txt`, `LABEL-CONTRACT.md` | Reviewer instructions and definitions |
| `review.html` | Offline draw/save/load editor; best-effort local recovery |
| `images/` | Unmarked original RGB |
| `proposals/` | Unmodified legacy two-class labels; not reviewed body/stream truth |
| `manifest.json` | Source hashes, run/split, selection cues, assignments, temporal neighbors |
| `review-schema.json` | JSON Schema for separate human review documents |
| `reviews/blank-review.json` | All 210 records; no accepted labels |
| `reviews/{aleyna,zoe,reagan}-queue.json` | Main assignments and shared calibration sample IDs |
| `contact-sheets/`, `calibration.jpg` | Raw RGB overview and 12 shared examples |
| `sequences/` | Five-frame source-timestamp context sheets |
| `verification.json` | Counts, split exclusion and source-preservation checks |

The manifest has top-level `schema_version`, `annotation_contract`, `pack_id`,
`source_inventory_sha256`, `training_eligible:false`, `automatic_label_changes:false`,
`run5_included:false`, `classes`, `samples`, and `sequence_groups`. Each sample
binds `sample_id`, `image`/`image_sha256`, dimensions, `source_run`, `split_role`,
source/legacy-label paths and SHA, unreviewed `proposals`, `selection_cues` and
numeric selection-only metrics, `sequence_groups`, `assigned_reviewer`,
`calibration`, filename timestamp, clock status, and null MP4 fields. Group IDs
are neighborhood identities, **not physical vial or event IDs**.

Downloaded review JSONs bind `pack_id` and the exact manifest SHA. Each record
preserves sample/image/run/split identities; `review_status` is unreviewed, draft,
or submitted_for_adjudication. Submitted records include reviewer and time.
`decisions` contains vial/body/stream status, boxes (`xyxy` plus occluded,
truncated and low_contrast booleans), notes, and frame notes. Unknown/uncertain
is never an implicit negative. All documents retain `training_eligible:false`.

Imports replace the editor session after validation; they **do not merge** people’s
reviews. Save independent calibration files and each main queue. Aleyna owns
reconciliation, and a later approved version must be integrated separately.

## Reproduce and validate

From the repository root, using a fresh output folder:

```powershell
$python = "$HOME/ai-club-tools/envs/data-windows/Scripts/python.exe"
& $python setup/vision/review_pack.py build --source "$HOME/ai-club-tools/results/real-synthetic-detector-20261019/data/real" --out "$HOME/ai-club-tools/results/aleyna-label-review-reproduced"
& $python setup/vision/review_pack.py validate --pack "$HOME/ai-club-tools/results/aleyna-label-review-reproduced" --review "$HOME/Downloads/REVIEWER-SAVED-FILE.json"
& $python -m unittest discover -s setup/vision -p test_review_pack.py -v
```

Paths/provenance and exact manifest identity are local to a build: validate a
download against the pack it actually came from. For the delivered ZIP, use the
`.../aleyna-label-review-20261019-final` pack directory in the validate command.
The builder verifies the original inventory, images and labels, then rechecks
selected originals and copied bytes after packaging preparation. Run 5 never
enters selection. Python needs Pillow and NumPy, already in the data environment;
reviewers only need a browser opening the local file.

Four unit tests cover untouched blanks, source/split and training-approval guards,
incomplete/uncertain submissions, box geometry and deterministic sampling.
An actual headless Edge `file://` test exercised drawing, all queue counts,
uncertainty rejection, submission, JSON download/reload, local recovery and
cross-pack import rejection with no browser errors. Test drawings are explicitly
marked and kept **outside** the review pack, under
`C:/Users/Austin/ai-club-tools/results/aleyna-review-editor-qa`.
All 210 copied JPEGs/legacy labels are hash-bound; the raw calibration/contact
sheets and the blank editor layout were visually inspected. No reviewed labels,
training dataset, detector change or measurement-QA change was produced.
