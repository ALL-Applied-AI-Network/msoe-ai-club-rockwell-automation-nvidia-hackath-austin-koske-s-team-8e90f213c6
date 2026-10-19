# Current tracked-liquid correction â€” October 19

Start with [VISION.md](../VISION.md),
[Run 5 evidence](tracked-liquid-run5.json),
[training provenance](bootstrap-training.json), and
[teacher-box agreement](run5-teacher-box-agreement.json).
The fixed-patch measurements below are historical and do not validate liquid color.

The new complete replay has 30 detected fills, 13/30 heuristic measurement
windows passing, and 30/30 trusted assessments unavailable pending geometry
review. The explicit earlier-vial reference produces one diagnostic deviation
of 7.42 Î”E2000. Ninety measurement samples and 360 PNGs were audited for exact
source pixels, masks, colors and aggregates; the v2 review schema passes.
38 replay and one bootstrap split/augmentation test pass from the source archive.

![Actual measurement frame, proposed geometry and exact sampling mask](tracked-liquid-evidence.png)

# Replay verification

Current image-based detection: [October 18 results](fill-detection.md), including
Run 5 one-to-one matching and boundary timing. The notes below preserve the
October 16 annotation-guided baseline; it is now an explicit optional mode.

October 16, 2026, local Windows data environment from `setup/`.

All five vial-fill videos decoded completely. All 15 camera files were probed and
hashed. The batch pass produced 17,450 sampled feature records and 30 annotated
completed-vial observations. Run 5 had 18 observations pass the fixed pilot QA
policy and 12 unavailable observations. Runs 1â€“4 remain unlabeled, so their image
telemetry does not create guessed fill/recipe or defect records.

| Run | Frames decoded | Samples | Annotated observations | QA-valid observations |
| --- | --- | --- | --- | --- |
| 1 | 33,662 | 5,611 | 0 | 0 |
| 2 | 4,802 | 801 | 0 | 0 |
| 3 | 5,792 | 966 | 0 | 0 |
| 4 | 9,482 | 1,581 | 0 | 0 |
| 5 | 50,942 | 8,491 | 30 | 18 |

The first batch took approximately 37 seconds including decoding and extraction
on this machine. This is a local throughput measurement, not camera-to-alert
latency. Default no-reference decisions remain unavailable; this is intentional.

Thirteen regression tests passed, including synthetic future-frame changes that
leave earlier features unchanged, backward seeking, preventing current/future
references, source/profile mismatch, invalid-image rejection, annotation validation,
video HTTP ranges, publication timing, export cutoffs and configuration/override
validation. JavaScript syntax was
checked with Node; Node is not required to run the app.

The initial cache roundtrip check caught tuples becoming lists after JSON reload.
Configuration now exports JSON-compatible values so cached and fresh results agree.

Browser checks exercised three-camera playback and synchronized seeking, exact
vial navigation, completed-check jumps, reference publication, backward seeking,
unlabeled run switching and CSV download. At the checked paused point all three
videos reported the same 935.562216-second MP4 time with no media error. A CSV
download through frame 27,444 contained exactly 17 observations and none available
after that frame.

A manually selected development reference from vial 16 gave a 5.96 Î”E2000 review
for vial 17; returning to vial 16 remained unavailable, so its own reference was
not reused. This tests plumbing, not the correctness of either vial. That test
reference was moved into a separate verification output folder; the normal app
starts without a preselected development reference.

Browser testing caught stale recipe cards during run preparation, a seek before
metadata was ready and fractional-time rounding at annotation boundaries. Run
switches now clear old state, pending seeks wait for metadata, and programmatic
seeks land inside the requested frame. The rapid-switch walkthrough passed after
those fixes.

These are software and measurement-coverage checks. There are no independently
reviewed defect labels here, so no false-alert rate, recall or production-reliability
claim follows. The older analysis's measurement policy and scores are separate.
