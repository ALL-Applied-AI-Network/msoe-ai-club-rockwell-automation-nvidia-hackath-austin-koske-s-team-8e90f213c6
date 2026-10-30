# Dataset 1-style real-camera transfer — October 22, 2026

The frozen synthetic RGB classifier now runs on real Dataset 2 `cap_seal`
images from the same final-fixture station. This is completed development
inference, not measured real defect accuracy. No weights, operating thresholds
or checkpoint selections changed. The operator application remains unchanged.

## Executed scope

| Item | Actual evidence |
| --- | --- |
| Input | 16 real indexed stills: four deterministic samples per Run 1–4 |
| Target | 30 proposed G positions per still; 480 slot-frame predictions |
| Model | Synthetic-only random-initialized RGB CNN, frozen epoch 5; SHA256 `6a84398a3e53d09750173e3a50a75a7af8bf70f37199e2cfca733ebf3ef552f7` |
| Inference | GB10 GPU, shared lock; all source/model/input hashes verified |
| Reference arm | Unavailable: no human-qualified capped real golden; no empty-tray or invented substitute |
| Proposed classes | 41 `pass` (capped-looking), 439 `no_vial`, zero `missing_cap` |
| Human truth | No actual labels supplied; accuracy, recall and false-alarm rate unavailable |

Four uniformly spaced image-index rows per run are selected before predictions.
Run 1 has only five stills, so its selected row indices are 0/1/3/4. These sources
include repeated observations of the same fixture contents; 41 predictions are
not 41 distinct physical vials. Some endpoint examples were already inspected
during the camera audit. This is not a blind final test. Run 5 and unassigned
images are excluded, including from sampling and geometry setup.

No confidence claim follows from softmax scores. Zero uncapped proposals does
not demonstrate that the data contains no uncapped vial or that missing-cap
recall is good. Unsupported crooked caps and damaged bodies cannot be cleared
by a `pass` proposal. Robot/hand occlusion is not qualified by the classifier.

## Geometry proposal and checks

The first indexed Run 4 still is the geometry anchor. A fixed search window
finds 30 aperture circle candidates with fixed Hough parameters. Their 5×6 grid
corresponds to the authored G01–G30 arrangement. With supplied lens intrinsics
and radial distortion held fixed, solvePnP fits a pose using the CAD tray plane.
The source cap-center geometry then projects at its own height, retaining the
same 48×48 crop convention used during synthetic training, resized to64×64.

The anchor has **0.659 px RMS / 1.414 px maximum in-sample aperture reprojection
residual**. This is a fit residual, not independent calibration accuracy. The
estimated camera position is constrained near the supplied pose. Source overlays
from all four runs and all30 crops in the last Run 4 sample were visually checked.
Other runs are not independently calibrated. Physical slot numbering and each
target's mapping/visibility remain human-review questions; no mapping is approved
by the numerical fit. A wrong or ambiguous crop is excluded from reviewed scoring.

Indexed still timestamps remain their original capture timestamps. No decoded
video-frame index or native PTS has been inferred from wall time or assumed
30 fps. Still-to-video mapping remains unmatched. No filling-vial-to-fixture-slot
physical identity is inferred from timing, color or sequential identifiers.

## Actual local artifacts

Root: `C:/Users/Austin/ai-club-tools/results/dataset1-real-transfer-20261022`.

- `batch-v1/`: original JPEG bytes, sample plan, proposed registration, pinned
  checkpoint, exact fitted architecture helpers and completed GB10 predictions.
- `batch-v1/registration-review.png`: proposed slot centers on all four runs.
- `crop-geometry-review.png`: all30 target crops from the last Run 4 sample.
- `summary-v1/real-fixture-predictions.png`: source-exact last-sample overlay.
- `summary-v1/real-fixture-predictions.mp4`: actual model proposals across16
  sampled stills, one still per second. This is a visualization, not live playback
  or a source-timed physical sequence. All16 output frames fully decode.
- `summary-v1/report.json`: provisional counts and explicit unavailable accuracy.
- `summary-v1/sample-observations.json`: 30 attributed real observations with
  proposed mapping, unqualified visibility, null quality/placement decisions and
  unlinked physical identity. No equipment command is emitted.
- `review-v1/review.html` and `review-v1.zip`: offline source-matched review.
- `browser-check-v1/`: demonstration-only test/export/import evidence; ineligible
  for real scoring and not included as team labels in the review packet.

The real input archive and returned predictions were SHA256-verified before
extraction into fresh paths. Every source/slot/box/model join is checked in the
summary builder. Executed remote root:
`~/ai-club-tools/experiments/dataset1-real-transfer-20261022-v1`.

## Human evaluation gate

See [the review guide](../../annotation_review/SLOT-REVIEW.md). The packet has
480 unreviewed tasks and96 positional priority tasks: G01/G03/G05/G16/G26/G30 in
each selected still, independent of model outputs. Reviewers inspect full-scene
context and the enlarged source crop, confirm target mapping and visibility,
then save a class/uncertain decision under their own names. All five organizer
classes are allowed in human review, even though the classifier supports only
three. Drafts are hidden until an initial human save; this reduces anchoring
but is not permanent blinding or a guarantee of independent review.

The scorer refuses demonstration exports and packet/source/crop mismatches.
It retains unsupported-class misses and reports coverage, per-class precision/
recall, wrong-empty proposals on reviewed occupied targets and fully scored
30-slot scenes. Unreviewed, occluded, uncertain and mismapped cases are excluded
with separate counts, never counted as correct. No-defect examples alone cannot
establish sensitivity to rare cap/body defects. Reviews are evaluation-only;
there is no real-data fitting or threshold tuning permission.

Srijan remains integration owner. Josiah/Hayden own operator/backend work.
This experiment supplies standalone observations and review evidence; it does
not silently replace their modules or join camera identities.

Verification: [dataset1-real-transfer-20261022.json](../verification/dataset1-real-transfer-20261022.json).
