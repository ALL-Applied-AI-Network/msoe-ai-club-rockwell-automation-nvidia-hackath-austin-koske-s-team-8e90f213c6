# Active-station vial, liquid body and stream: review contract v1

This contract defines **new human review**, not the meaning of existing teacher
labels. Existing `colored_liquid` proposals can combine body and stream. Do not
rename them into reviewed body labels or accept a proposal without inspecting RGB.
No source image, source label, timing annotation or training dataset is changed by
this review pack. The current detector and measurement QA remain unchanged.

## Classes and box boundaries

Use original 640 x 480 RGB. Boxes are continuous, full-image pixel coordinates
`[x_min, y_min, x_max, y_max]`, with exclusive right/bottom bounds, zero origin,
and `0 <= x_min < x_max <= width` (likewise height). Draw tight axis-aligned boxes
around **visible image evidence**, not an inferred complete physical object.
An encompassing box can contain background or an intervening occluder; the box
is not a liquid-pixel mask. Never extend its edges to imagined hidden geometry.

| New class ID / name | Include | Exclude |
| --- | --- | --- |
| 0 / `vial` | The one vial occupying the active filling station under the nozzle; its visible glass exterior, rim and base, including printed label within its exterior. One box maximum. | Neighbor/queued vials, nozzle, black holder, guard, wall and imagined hidden glass. If the station vial cannot be uniquely identified, use uncertain. |
| 1 / `liquid_body` | The accumulated pool **inside that active vial**: visible bottom/side extent and visible meniscus/surface. Interior bubbles and a visibly connected small impact disturbance belong to its extent. One box maximum. | Falling stream, isolated wall film/wetting, droplets, glare alone, paper label, holder and foam/splash extending beyond the identifiable pool. Do not use the vial box as a substitute. |
| 2 / `stream` | Visible falling liquid from below the nozzle to the visible body surface/impact interface. Separate disconnected droplets/segments get separate boxes. A contiguous jet gets one tight box. It can pass through the vial mouth into the unfilled space. | Nozzle, reflections on glass, accumulated body, wet walls and unidentifiable splash. Do not extend a faint jet through an invisible gap to join segments. |

Where body and stream meet, end the stream at the visible impact/surface interface
and start the body's upper extent at that surface. Axis-aligned boxes may overlap
in projection. Do not invent a boundary if it cannot be located: use uncertain
for the affected class and explain why. A visible jet does not prove a visible
pool, and a visible pool does not prove current pouring. Label what this frame
shows; neighboring frames help interpretation, not hallucinating missing pixels.

## Every class needs an explicit decision

- **unreviewed**: untouched; never a negative training example.
- **present**: identifiable in RGB, with the required tight box(es).
- **absent**: reviewer can establish absence from the visible scene. No boxes.
  An empty, inspectable vial can have vial present/body absent/stream absent.
- **uncertain**: cannot distinguish presence or locate its bounds reliably because
  of pale/clear liquid, occlusion, motion, ambiguous station identity, glare, etc.
  No training box; explain the uncertainty. A hidden pool is not automatically
  absent. When no active vial is visibly present, a pool in a neighbor is not a
  positive: use absent if the active station is clearly empty, otherwise uncertain.

For each present box flag `occluded`, `truncated` (cut off by the image boundary)
and `low_contrast` when applicable. These flags describe visible evidence, not
certification. Use notes for wall film, foam or disputed boundaries. No automatic
size, position, color or detector-confidence rule overrides a human decision.

## Review workflow for Aleyna, Zoe and Reagan

1. Extract the **whole** pack ZIP into a normal folder. Open `review.html` locally.
   No server, network connection, Python installation or account is needed.
2. First review the 12 shared calibration frames **independently**, with proposals
   hidden. Save separate named JSONs. Compare disagreements together with Aleyna;
   record agreed interpretations in notes before proceeding. The 12 are already
   included in the 210 unique frames, not additional training examples.
3. Select your assigned queue (70 frames each). Enter your name. Read the raw RGB
   first, then draw each visible class and explicitly mark absent/uncertain cases.
   Proposals are optional context only; there is intentionally no accept/copy
   proposal button. At 150% display, coordinates still refer to original pixels.
4. Use adjacent-frame links / `sequences/` sheets to inspect context. Supplied
   image timestamps are **not verified MP4 positions or physical pour onsets**.
   Existing stills are roughly two seconds apart; no sub-second timing labels
   are established here. Early/pale selection tags are sampling hypotheses.
5. Mark each completed frame **Submit for adjudication**. This validates the
   three decisions and boxes. It does not approve training or color measurement.
   Save progress JSON frequently, especially before closing or switching a
   reviewer. Local recovery is best effort, not a substitute for a saved file.
6. Send your named JSON to Aleyna. Preserve calibration files separately from
   your main queue. Aleyna reconciles overlaps/uncertainties; Austin integrates a
   separately approved version later. Do not overwrite source `.txt` files.

The pack is deliberately enriched for difficult cases; its proportions are not
an estimate of production prevalence or overall detector accuracy. Runs 1-3 are
training candidates and Run 4 is development only. Run 5 is not in this pack.
Keep all reviewed samples and temporal neighbors from a run on that run's side
of the split. Run 4 has already been used in model selection, so it is not blind
test data. A new untouched CSI recording is needed for final generalization.

## Handoff format

`manifest.json` binds source identities, encoded image hashes, byte-identical
legacy-label copies, selection cues, sequence groups, reviewer assignments and
split roles. `reviews/blank-review.json` and editor downloads follow
`review-schema.json`. A download contains pack/manifest identity, reviewer,
save time, and one entry per pack frame. Each entry carries immutable sample ID,
image hash, run and split, review status, submitted reviewer/time, three class
decisions, per-box flags and notes. No real vial/recipe IDs are assigned.

`training_eligible` stays false. `submitted_for_adjudication` means a human has
completed this pass, not that Aleyna has approved it. Unreviewed/uncertain entries
must never be interpreted as empty YOLO targets. There is no training export or
automatic merging in this tool. `review_pack.py validate` checks identity,
decisions and geometry but cannot adjudicate semantic correctness.
