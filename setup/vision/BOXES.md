# Next milestone: vial and liquid bounding boxes over video

The immediate task is object localization and temporal continuity. Segmentation,
color QA coverage, recipe comparison and defect decisions are downstream tasks.
The [pale-liquid visibility pilot](VISIBILITY.md) compares local image cues
without changing that milestone or the replay's measurement pipeline.
The existing YOLO11n already predicts two box classes; use it as the first
detection baseline. Start with one detector, rather than two independently
trained networks. ByteTrack supplies candidate vial tracklets; associate each
current liquid box to its current vial. A tracker cannot correct a bad fluid
box, and an ID does not establish physical vial identity through occlusion or
carrier replacement.

## Target definition for the reviewed box dataset

- `vial`: the full vial body at the filling station, excluding the nozzle and
  falling stream. The current weak labels target the fill-position vial; they
  do not support a claim that every vial elsewhere in the scene is labeled.
- `liquid_body`: the visible filled liquid extent inside that vial, including
  the visible meniscus and lower extent. A small patch of color, base reflection
  or existing sampling mask is not its full bounding box. The existing model's
  `colored_liquid` class is a provisional baseline for this target, not proof
  that this definition was followed consistently.
- Record empty, unobservable, occluded and uncertain cases separately. An absent
  label is not an empty-vial negative. Clear liquid needs its own evidence;
  the existing colored-liquid labels cannot establish its absence or presence.
- A rectangular envelope can include an opaque holder, curved glass or glare.
  The box defines a crop, not a set of verified measurement pixels. Record
  visibility/occlusion instead of inventing hidden outlines.

The label team retains review and corrections. This work supplies a review
interface; exported predictions are not silently promoted to training truth.
No timing/event labels are changed or inferred from box reviews.

## Conventional experiment sequence

1. Review a small representative box set and verify the target definition.
   Cover empty/early/partial/full vials, all observed colors, motion, occlusion,
   reflections and departures. Keep an independently reviewed scoring subset.
2. Freeze a versioned dataset. Keep recordings and their augmented descendants
   in one split: Runs 1-3 fit, Run 4 development. Run 5 stays out of fitting and
   checkpoint selection; it has already been repeatedly inspected and is a
   diagnostic recording, not a new blind benchmark. A later CSI recording can
   provide a fresh final test.
3. Retrain/fine-tune the two-class detector on Rosie using the corrected box
   data. Compare it to the frozen existing detector. Change capacity or split
   the tasks into two models only if per-class errors justify that experiment.
   Use the existing Singularity/Slurm setup; no Docker. Do not train another
   pixel-mask model during this milestone.
4. Evaluate detections and tracking separately. Report per-class precision,
   recall, IoU and mAP50-95 on reviewed boxes, stratified by the difficult states.
   Inspect full clips for misses, drift and incorrect liquid/vial association.
   ID switches, fragmentation and IDF1 need reviewed temporal identities; until
   those exist, export candidate tracklets without claiming tracking accuracy.
5. Once boxes are solid, test a simple interior sampling region within the
   liquid crop, rejecting walls/glare/holder pixels. Compare it to reviewed
   measurement pixels before deciding whether learned segmentation is needed.
   This later experiment must not quietly weaken the current color QA rules.

The first milestone ends with a reproducible box-only video, per-class box
metrics against reviewed targets, explicit failure clips and a frozen model.
The 13/30 and 8/30 color-QA counts are not its success criterion.

## Current diagnostic, with no retraining

`review_box_tracking.py` exports a side-by-side source/box video, current raw
boxes and candidate ByteTrack IDs for the entire recording. It uses the pinned
image-only detector cache; every consumed source frame's RGB hash is checked.
No masks, evaluation annotations, recipes, cached tracking, fill IDs or color
decisions are consumed. The detection threshold remains the cache's existing
0.35; consequently this does not exercise ByteTrack's low-confidence detections.
The review is sampled at five frames/second, with original MP4 timing retained;
it is not a native-frame-rate tracking validation.

Use the isolated vision environment from earlier detector evaluation. Add the
tracker's assignment solver there, not to the ordinary replay app environment:

```powershell
$vision = "$HOME/ai-club-tools/envs/vision-windows/Scripts/python.exe"
& $vision -m pip install lap==0.5.13
& $vision setup/vision/review_box_tracking.py --cache "$HOME/ai-club-tools/results/tracked-liquid-20261019/cache/run5-8bdd8cad2977ee74e6a3.json" --cache-sha256 aff504002184d856ac06451f1730677eace2e840b89110c2cb6c0af39a8850d8 --out "$HOME/ai-club-tools/results/box-tracking-review-20261019"
```

Run from the repository; add `--data` for recordings elsewhere. Use a fresh
output directory. The script uses the pinned installed Ultralytics 8.4.182
ByteTrack implementation. All displayed rectangles are fresh raw detector
boxes; Kalman predictions do not paint a missing object or expand the fluid.
Missing detections remain missing. The lost-track buffer is scaled from 30
native frames to five sampled updates. These tracking parameters are separate
from the unchanged replay's measurement QA.

- `boxes.mp4`: source on the left, detected boxes/tracklets on the right. Blue
  is vial; green liquid has one containing current vial; red is unassociated
  or ambiguous. This is a visual review artifact, not an accuracy claim.
- `boxes.jsonl`: one row per sampled frame with original frame index, MP4
  seconds, source RGB SHA256, raw boxes/confidences, vial candidate track ID,
  and each liquid's current-vial association. Source coordinates use exclusive
  upper bounds `[x0,y0,x1,y1]`.
- `manifest.json`: recording/model/cache identity, tracker version/settings,
  complete decode counts, and **presence counts**, which are not precision or
  recall. Reviewed detection/tracking metrics stay null.
- `review.json` and `stills/`: 60 uniformly sampled exact source/overlay pairs
  for the label team's correction interface. Fields `review_status`,
  `corrected_boxes` and `physical_vial_identity` start unreviewed/null. A future
  corrected box should specify class, source-coordinate box, visibility and
  reviewer provenance. This format is not automatically imported into training.
- `index.html`: local video and review gallery. To view in a browser, serve
  this output directory on loopback; no camera or industrial network is used.

The existing teacher-agreement evaluation reports vial AP50-95 about 0.814 and
liquid about 0.499. These describe agreement with unreviewed teacher boxes,
not independent human accuracy. They motivate investigating liquid localization
before adding more downstream color logic.

Official background: [object detection](https://docs.ultralytics.com/tasks/detect/)
and [video tracking](https://docs.ultralytics.com/modes/track/).
