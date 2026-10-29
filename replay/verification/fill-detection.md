# Independent fill detection verification

October 18, 2026, local Windows data environment. Full per-fill evaluation is in
[run5-fill-detection.json](run5-fill-detection.json). The earlier
[annotation baseline](README.md) remains separate.

The normal app and analysis default to `--fill-source image` and skip annotation
files entirely. `FillDetector.update(feature)` consumes sampled image features,
frame indices and configuration only. Offline `validation.evaluate` consumes the
finished detections and supplied annotations; it does not change detections,
measurement windows, recipe IDs or replay decisions.

The full Run 5 pass was also repeated with all 30 annotations disabled. Detected
cycles, emitted detector events and completed measurements were exactly identical
to the labeled evaluation pass. Processing summaries and implementation/config
IDs are in [image-replay-processing.json](image-replay-processing.json).

## Run 5 development evaluation

| Check | Result |
| --- | --- |
| Decoded frames / sampled frames | 50,942 / 8,491 |
| Supplied fills / accepted detected fills | 30 / 30 |
| One-to-one matches, interval IoU ≥ 0.5 | 30 |
| Missed fills / extra detections | 0 / 0 |
| Start error, absolute median / p95 / max | 18 / 35.4 / 47 frames |
| End error, absolute median / p95 / max | 20 / 66.1 / 72 frames |
| Completion confirmation lag, median / p95 / max | 71.5 / 136.15 / 163 frames |
| Color measurements passing pilot QA | 19 / 30 |
| Open fill at EOF | None |

Start error excludes fill 1, which begins mid-stream at frame 0. Signed medians
are +18 frames at the start and −20 at the end: visible-stream estimates tend to
start later and end earlier than the supplied boundaries. Confirmation lag is
relative to annotated end, not to estimated flow stop. A fill is completed only
after carrier departure is observed twice, so this detector makes no early-end
claim during an ingredient pause.

At MP4 30 fps, the absolute median errors correspond to 0.60 and 0.67 encoded
seconds; median confirmation lag is 2.38 encoded seconds. These are **not physical
seconds**. Camera capture timing and cross-camera offsets remain unverified.

This is development evaluation: Run 5 imagery and boundaries were inspected
while choosing pilot ROIs and thresholds. It establishes event correspondence on
this recording, not held-out performance, defect detection, stage classification,
exact pump timing or warning lead time. The supplied boundaries have not been
independently adjudicated.

## Causality and integration checks

Twenty-two regression tests pass, including:

- Image-filled temporary video produces the same nonempty detections and color
  observations with labels present, removed or deliberately wrong. Image-mode
  catalog loading also succeeds with a malformed annotation file.
- Multiple ingredient episodes with a long pause produce one fill; a pause or
  EOF cannot publish a completed fill. Empty carrier cycles and brief spikes do
  not create accepted fills.
- Running a prefix gives exactly the same states and published events as the
  same prefix of a longer recording. Changing future images leaves earlier
  image features and detector state unchanged.
- Frame gaps reset confirmation and make affected completed measurement QA
  unavailable. Measurements use only a stationary tail before departure;
  samples cannot come from the next vial.
- Backward seeking hides later observations. Events/exports stop at the playhead;
  full-recording fill navigation is explicitly labeled navigation-only. A
  duplicate detection cannot match the same annotation twice.
- The HTTP image-mode flow works without labels and blocks recipe-reference
  creation when recipe identity is unknown. The existing annotation-baseline,
  reference provenance, byte-range and export checks continue to pass.

JavaScript syntax validation passes. Browser checks exercised image-mode fill
navigation, a rejected and an accepted color measurement, completion jumps and
backward seeking. No recipe identity is guessed from color or copied from the
offline evaluation join. Recipe/stage alerts need a separate controller/operator
recipe context input.

At the final checked pause, all three browser videos were at MP4 728.408333
seconds with ready state 4 and no media error. The completed fill 14 measurement
had three accepted samples; its recipe assessment correctly remained unavailable.

## Other recordings: processing checks only

All five recordings decode and process with annotations disabled. Pilot accepted
fill counts are 19, 3, 2, 6 and 30 for runs 1–5. These counts are **not validation**
on runs 1–4. In Run 3, two additional stream candidates fail the minimum peak
contrast gate, and weak evidence elsewhere requires review. The older offline
color-change analysis also uses a different event definition and cannot establish
ground truth. The data group's reviewed boundaries are needed before measuring
cross-run recall or changing the weak-stream policy.

The 11 unavailable Run 5 color windows remain in the ledger; they are not missed
fill detections. In particular, the pre-departure tail can contain initial carrier
motion or too few QA-valid samples. This work does not loosen color QA to improve
the coverage figure.

## Reproduce

```powershell
& "$HOME/ai-club-tools/envs/data-windows/Scripts/python.exe" -m unittest discover -s replay/tests -v
node --check replay/static/app.js
& "$HOME/ai-club-tools/envs/data-windows/Scripts/python.exe" -m replay validate-detection --runs 5 --out "$HOME/ai-club-tools/results/fill-validation"
& "$HOME/ai-club-tools/envs/data-windows/Scripts/python.exe" -m replay analyze --runs all --out "$HOME/ai-club-tools/results/image-replay"
```

`run5/detection-validation.json` includes recording/annotation hashes, configuration
and implementation IDs, the matching rule and every matched boundary. Analysis
outputs also include source provenance, all detected stream episodes, availability
frames, QA flags and per-sample diagnostics. Raw recordings and labels are not
modified.
