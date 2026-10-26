# Two-station integration rehearsal — October 23

This document describes the development app on `austin-work`. The scoped
October26 `main` handoff publishes the liquid model/example and QA contracts;
it does not include this operator server or establish live deployment.

Completed filling and fixture model outputs now pass through one candidate
`station-evidence/v1` contract. This is an opt-in integration path for Srijan
and Josiah/Hayden. The running operator application's model defaults are unchanged.

## Executed evidence

The local root is
`C:/Users/Austin/ai-club-tools/results/two-station-rehearsal-20261023`.

| Check | Actual result | Interpretation |
| --- | --- | --- |
| Cached stream ingestion | 264 filling frames + 480 fixture target observations | Two independent source streams; no new inference |
| Filling evidence | 189 mask proposals; 105 unchanged-QA color samples | Availability, not reviewed mask or recipe accuracy |
| Fixture evidence | 41 capped-looking + 439 empty proposals | Unreviewed slot-frame proposals, not distinct vials |
| Reference/identity guard cases | 19/19 passed | Synthetic software fixtures, not physical calibration |
| Fixture grid diagnostic | 0/16 sampled stills flagged | Does not approve slot identity, crops or occupancy |
| Artificial grid controls | Both 8-pixel shifts flagged | Digital diagnostic checks, not measured camera robustness |
| Source-pinned review journal | 757 events; 2 demonstration responses | No actual operator response or product release |
| Real accuracy | Unavailable | 0 human reviews supplied for either packet |

`rehearsal-v2/report.json` is current. Earlier `rehearsal-v1` has the same cached
input counts and 18 guard cases; v2 adds explicit demonstration-reference refusal.
`geometry-v1/geometry-checks.json` contains per-image correspondences/residuals.
`evaluation-v1/evaluation-status.json` retains unavailable metrics as null.

## Contract and reference handling

Each observation contains source camera/asset/stream, explicit hash kind, native
clock or capture-only timestamp, target crop/context, frozen model/profile IDs,
raw proposal, unchanged measurement QA and qualification state. Evidence ingestion
does not carry product decisions or equipment commands. Coordinates are source
pixels at 640×480 in this candidate contract; other resolutions require a new
explicit adapter rather than implicit resizing.

`commissioning.ReferenceGate` stores reference snapshots separately from model
predictions. It refuses unreviewed, demonstration or model-qualified references;
wrong conditions; same/future frames; incompatible clocks; changed camera/profile;
lost alignment; declared discontinuity; temporal stream/context changes; wrong
golden slot; and source clock rewind. An optional maximum age is an explicit
integration policy, not a tuned model threshold. No lifetime is invented by default.

Reference purposes are temporal background (confirmed empty), capped golden
(confirmed capped), and recipe endpoint (independently accepted intended recipe).
V1 requires comparable native clocks and confirmed alignment; capture-only stills
cannot become causal references. A future reviewed still-reference/mapping adapter
needs its own explicit contract. The six current clip-first references remain
unreviewed. The gate therefore blocks qualified use without deleting raw masks
or the 105 valid diagnostic color samples. It does not recompute inference.

Recipe comparison also needs intended recipe, confirmed completion and physical
vial identity. Fixture placement assessment needs confirmed mapping, visibility
and attributed expected occupancy. The classifier does not support crooked caps
or damaged bodies. Neither path automatically authorizes a product-quality release.

`associate_physical_vial` never joins observations by color, order or apparent
time alone. Complete, non-demonstration, attributed association evidence must name
both exact observation IDs and the physical vial. Such a statement is attributed
operator input, not independent identity authentication. Filling video clocks and
fixture still timestamps stay separate throughout this rehearsal.

## Exact pixels and portable evidence

`sources-v2/source-assets.json` provides 72 original lossless filling PNGs and
16 original fixture JPEGs: pixels for 552/744 observations, since each fixture
image supplies 30 target observations. The other 192 filling observations retain
their frozen source identity and model outputs but have no exact local RGB asset
in this portable subset. They are explicitly listed as missing.

The original video SHA and decoded native frame/PTS match, but local PyAV 19.0.1
and FFmpeg 6.1.1 RGB conversion did not reproduce the frozen GB10 RGB bytes at
frame264 (mean absolute channel difference 0.8865, maximum2). The cause has not
been established. Strict extraction refused the mismatch before writing images;
the failed `sources-v1` directory is retained. No evidence was silently sanitized
or replaced. Recover the remaining frozen frames using the original environment
and verify every recorded RGB hash before claiming full pixel coverage.

The bundle's filling MP4 and 16-still fixture slideshow are presentation media,
not source-exact pixels or synchronized views of the same physical vials.

## Reproduce on CPU

Run from the repository with a fresh output path for each command. `ROOT` below
is the existing results directory, not a training input folder.

```text
python setup/vision/prepare_two_station_rehearsal.py --results ROOT --out ROOT/two-station-rehearsal-20261023/bundle-new
python -m replay.operator.rehearsal --bundle ROOT/two-station-rehearsal-20261023/bundle-new --out ROOT/two-station-rehearsal-20261023/rehearsal-new
python setup/vision/check_fixture_geometry.py --batch ROOT/dataset1-real-transfer-20261022/batch-v1 --out ROOT/two-station-rehearsal-20261023/geometry-new
python -m annotation_review.evaluation_bundle --results ROOT --out ROOT/two-station-rehearsal-20261023/evaluation-new
```

The geometry command requires OpenCV/NumPy; exact evidence materialization needs
Pillow and optionally PyAV for strict decoding. Use the existing vision environment.
`materialize_station_evidence.py --frozen-probe` copies only identity-verified
lossless probe images and lists missing frames rather than substituting pixels.

Once actual human exports exist, add `--slot-review PATH` and/or `--pool-review
PATH` to the evaluation command. It uses the existing packet/source-matched
scorers, refuses demonstration reviews, and preserves exclusions/uncertainty.
Real reviews remain evaluation-only and do not select models or thresholds.

## Team integration and physical test

- Srijan: adapt candidate inference outputs to the evidence contract; implement
  actual capture/reset/alignment and stream discontinuity handling; measure the
  complete live pipeline after wiring it in. Cached timings are not that measurement.
- Josiah/Hayden: display raw evidence and qualification reasons separately;
  pin the exact source on review requests; keep acknowledgement, response and
  follow-up distinct from recipe/product approval. The journal here exercises
  those event semantics without installing a dashboard or authenticated service.
- Data reviewers: complete the existing 72-frame liquid and 96-priority/480-total
  fixture packets. Include definite positives, empty/occluded/uncertain cases and
  missing-cap examples before interpreting per-class accuracy.
- Austin/CSI: establish independent acceptable recipe endpoints, confirmed empty
  and capped references, reviewed slot orientation/visibility, intended occupancy,
  physical vial-to-slot evidence and approved supervised feed/test arrangements.

The generated `handoff-v1` includes capture worksheets with blank human-decision
fields. Those worksheets are planning records, not automatically importable
approvals. No synthetic-only fitting, source labels, QA thresholds, application
defaults or hardware behavior changed in this work.
