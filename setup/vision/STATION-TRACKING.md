# Causal vial tracking and liquid association: local Run 5 review

The replay app now offers **Show station tracking** as an optional, default-off
display. Its independent source-frame view and seek-safe live overlay consume
these same tracking rules without changing measurement QA. See the
[app instructions](../../replay/README.md#optional-station-tracking-overlay).

This is an **opt-in box-only diagnostic and display**, implemented in
`replay/box_tracking.py`. It operates on the current detector's unchanged raw
proposals and original RGB. It does not replace production `VialGeometry`,
change masks/measurement QA, assign supplied vial IDs, consume timing/recipe
annotations, or promote a model. The current detector stays pinned to SHA-256
`89ed11f7046171e5340dc45a9d528010de1cefbba21e163b23b7098151e0880e`.

## What changed

The earlier ByteTrack diagnostic produced four unique IDs across Run 5. A
location-based tracker can reuse an identity when another physical vial arrives
at essentially the same station coordinates. This implementation tracks
**provisional active-station segments**, with explicit uncertainty boundaries:

- Use the existing camera profile: a fresh vial box must span nozzle center
  x=306.5 and have height >180 pixels. Neighbor vials cannot supply station liquid.
- Recompute the existing grayscale thumbnail motion feature from current and
  previous RGB. Reuse image-detector departure threshold 8, confirmation .2 s,
  and settling .6 s. No cached fill events, geometry tracks or QA outputs enter.
- Withhold IDs/links during movement and settling. Confirmed movement retires
  the segment, even when incoming and outgoing boxes occupy the same location.
  A single unconfirmed spike blocks links without falsely treating fresh boxes
  as missing detections. Motion changes can also be caused by lighting.
- Confirm a track with three consecutive current observations. Match stable
  boxes at IoU >=.65. A detection-only gap up to .6 s can retain an internal
  candidate identity, but renders no predicted box and associates no missing
  observation. Reappearance must be reconfirmed. Longer gaps retire the track.
- Input cadence gaps immediately reset identity/motion confirmation. Decreasing
  or duplicate frame indices are rejected. New instances get fresh UUID session
  namespaces by default; an explicit name supports reproducible replay. A future
  source must start a new instance/namespace on restart and after interruption.
- Liquid links require one fresh, confirmed active vial, original **90%** liquid
  containment, and exactly one eligible liquid proposal. No clipping, invented
  liquid, confidence-based choice among competing proposals, temporal liquid
  masks or stale observations. Missing/ambiguous evidence has an explicit reason.

These are association rules, separate from unchanged color-measurement QA. A
`candidate_link` does not verify body-versus-stream semantics or liquid pixels.
Every liquid output says `measurement_authorized:false` and
`liquid_scope:unreviewed_legacy_class`. The detector cannot acquire a third
stream class or recover missed pale liquid through tracking alone.

## Local deliverables

Directory: `C:/Users/Austin/ai-club-tools/results/station-tracking-review-20261020`

| File | Review purpose |
| --- | --- |
| `run5-carrier-transitions.mp4` | Eight image-motion neighborhoods, 54.6 seconds |
| `run5-association-review.mp4` | Every changed-link neighborhood, 48.2 seconds; unreviewed reasons in `association-review.json` |
| `run5-early-pours.mp4` | Eight prior image-estimated early neighborhoods, 49.6 seconds |
| `run5-tracking-full.mp4` | Entire Run 5 sampled at 5 fps, 28m18.2s |
| `tracking.jsonl` | Per-frame source identity, original proposals, states, candidate IDs/links and causal events |
| `comparison.json`, `plan.json` | Frozen settings/input/code hashes, preservation checks, counts, timing and video hashes |
| `stills/` | Side-by-side examples including carrier, pale/missed-liquid and changed-link cases |
| `calibration-intake.json` | Structural intake of Austin's separate draft; not tracker/training input |

Videos show the **old ByteTrack diagnostic on the left**, new station segments
on the right, with identical source pixels, box coordinates and confidences.
Cyan is vial, green is a candidate liquid link, amber is withheld association.
Headers show original frame index and MP4 time; no unverified wall-clock mapping.
The short clips concatenate neighborhoods, so source time can jump between clips.
Green is **not** a QA pass, verified mask, recipe match or release authorization.

## Executed diagnostic

All 50,942 source frames decoded; all 8,491 sampled RGB identities and raw proposals
matched the original pinned cache and previous diagnostic. New outputs have
30 candidate station IDs versus four old IDs, with 32 confirmed image-motion
events (including movement with no new station track). **These counts are not
verified physical vial identities, detection accuracy, IDF1, or measurement coverage.**

Of 4,213 raw liquid proposals over 4,207 sampled frames, 3,375 get new candidate
links versus 3,405 old links with an actual candidate ID. The old diagnostic had
3,408 `unique_vial` statuses, three without a track ID. The new tracker removes
30 ID links over 25 frames and adds none: 13 withheld for carrier transition,
seven no-fresh-station-vial, eight competing liquid proposals, two tentative track.
These are changed behaviors, not adjudicated old mistakes.
All remaining new reasons: 768 outside-vial, 42 no fresh
station vial, 18 during carrier transition/settling, eight competing liquid
proposals, and two track-not-confirmed. Detection thresholds and containment
were not weakened to increase coverage. Reviewed temporal identities are needed
before claiming an association or tracking accuracy improvement.

Initial-prefix replay independently reproduced 120 outputs; a 20-sample restart
probe used a fresh namespace and started without inherited associations. Every
delivered MP4 was fully decoded to verify its frame count. Eight new tracking
tests plus existing geometry/detection/pipeline checks passed (31 total), covering
same-position replacement, short/long detection loss, ambiguous liquid/vials,
off-station neighbors, motion spikes, source immutability, causality, restart,
and preservation of the existing 90% containment minimum.

The full-run comparison additionally hashes config, original geometry,
measurement, fill detection, pipeline and detector before/after processing. None
changed. No training, label edit, live equipment connection or web preview used.

## Reproduce from the repository root

Use a fresh output directory. The prior baseline JSONL is intentionally pinned
as the old diagnostic; installing or re-running ByteTrack is unnecessary.

```powershell
$python = "$HOME/ai-club-tools/envs/data-windows/Scripts/python.exe"
$out = "$HOME/ai-club-tools/results/station-tracking-reproduced"
$cache = "$HOME/ai-club-tools/results/tracked-liquid-20261019/cache/run5-8bdd8cad2977ee74e6a3.json"
$baseline = "$HOME/ai-club-tools/results/box-tracking-review-20261019/boxes.jsonl"
$video = "datasets/dataset2/csi-vial-fill-dataset/vial_fill/videos/vial_fill_video_5.mp4"
& $python setup/vision/compare_station_tracking.py --video $video --cache $cache --baseline $baseline --out $out
& $python setup/vision/export_association_cases.py --trial $out --video $video --cache $cache --baseline $baseline
& $python -m unittest replay.tests.test_box_tracking replay.tests.test_geometry replay.tests.test_detection replay.tests.test_pipeline -q
```

## Application / future input boundary

`StationTracker.update(rgb, proposals, frame_index)` returns fresh vial/liquid
evidence and events, without knowing frame source, labels, recipes, colors or
QA. A frame-source adapter can use this for replay or a future approved camera;
the existing measurement path is deliberately untouched. Input must currently
match this 640x480, 30 fps / stride-six profile. Do not silently reuse its timing
thresholds at another cadence. Unknown/missing source frames reset evidence;
there is no interpolation or connection to industrial hardware.

This is active-station continuity, not full conveyor multi-object tracking.
It cannot establish identity through an unseen same-position swap, fix the raw
detector's vial-top/stream confusion, or certify pale/clear liquid. Run 5 has
already guided inspection and is not a blind final benchmark. Aleyna's group
owns reviewed boxes and temporal identities. New CSI footage remains the final
generalization-test input.

## Austin's calibration intake

Source: `C:/Users/Austin/Downloads/aleyna-visible-boxes-v1-f29fdb0db46b-Austi-calibration-2026-10-20T00-35-34-617Z.json`.
It validates against the delivered pack: 12 edited calibration records, all
three class decisions populated (vial:12 present; body:5 present/7 absent;
stream:6 present/6 absent). They remain **drafts**, with zero submitted records.
No statuses or annotations were changed, and the file is not consumed by the
tracker or training. Aleyna still owns adjudication. Reload the saved JSON and
use Submit frame for adjudication if the reviewer wishes to finalize that pass.
