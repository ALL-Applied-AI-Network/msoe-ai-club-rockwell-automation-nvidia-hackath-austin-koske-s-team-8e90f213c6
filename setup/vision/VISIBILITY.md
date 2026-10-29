# Pale liquid and early pouring: image-processing pilot

Keep the current localization milestone. This is a local feature/visibility
comparison, not a segmentation model, production detector or early-alert rule.
The replay, color QA, trained weights and recipe references are unchanged.

## Techniques to try

1. **Vial crop at useful model resolution.** The vial is narrow in the full
   frame; a crop-focused detector can spend more input pixels on the fluid.
   Train and evaluate with the same crop convention. Upsampling alone cannot
   restore detail absent from the source. A crop model was not trained here.
2. **CLAHE on lightness.** Improve local contrast without independently
   equalizing the RGB channels. This pilot converts RGB to OpenCV Lab, applies
   CLAHE only to L (clip limit 2, 8x8 tiles), then returns to RGB. The frozen
   detector sees raw and enhanced full frames through identical inference.
   Later crop-only preprocessing is a separate ablation; its tile geometry
   differs from this full-frame pilot.
3. **Earlier-frame difference.** Subtract a previous stationary, aligned image
   of the same vial to expose the falling stream, changing refraction and
   accumulating liquid. This avoids requiring strong saturation. Production
   would need a reliable earlier image, alignment and reset on carrier motion;
   the diagnostic references here are not certified empty/same stationary vial.
   A difference image indicates change, not guaranteed liquid.
4. **Meniscus/edge evidence.** Horizontal gradients can expose a liquid surface
   with little color contrast. Labels, glass, the holder and glare also produce
   strong edges. Use this as an additional cue, not a standalone liquid mask.

The falling **stream** and the **liquid body accumulating inside the vial** are
different targets. The current colored-liquid box class excludes the falling
stream. At the earliest moment there may be a visible stream but too little
body to box reliably. Keep an image-derived stream/change cue separate from
the liquid-body detector rather than stretching the body box to cover both.

Enhancement is for localization only. Any later color measurement must still
read the original RGB pixels. Evaluate raw versus enhanced training/validation
consistently; a human-readable enhancement is not automatically better for a
network trained on raw images. Include pale/clear/early-fill examples and hard
negatives in the reviewed box data; filters do not repair missing labels.

## Local comparison

The script takes the first eight chronological non-partial image-estimated
starts from the existing Run 5 image-only result, excluding the partial fill
already underway at frame 0. These select clips, not reviewed true-onset
labels. Each clip runs from -0.5 to +3.5 seconds relative to that estimate at
the original 30 fps. A fixed analysis crop comes from a fresh vial detection
one second before the estimate, and stays fixed throughout the clip. There is
no crop warping, motion compensation or claim of an empty-vial reference.

Four panels show source RGB, full-frame L-CLAHE, absolute mean RGB difference
from the earlier image (fixed display gain 4), and Sobel dy on enhanced grayscale
(kernel 3, fixed gain 2). No thresholded liquid mask is generated. The raw and
CLAHE detector passes are both fresh inference with identical frozen weights
and thresholds. The original cache is only used for clip selection, crop
reference proposals and source-pixel verification; cached masks/colors/recipes
never supply a result.

```powershell
$python = "$HOME/ai-club-tools/envs/data-windows/Scripts/python.exe"
& $python setup/vision/compare_liquid_visibility.py --cache "$HOME/ai-club-tools/results/tracked-liquid-20261019/cache/run5-8bdd8cad2977ee74e6a3.json" --cache-sha256 aff504002184d856ac06451f1730677eace2e840b89110c2cb6c0af39a8850d8 --out "$HOME/ai-club-tools/results/liquid-visibility-clahe-20261019"
```

Run from the repository and use a new output root; add `--data` if needed. This
bounded pilot requires the existing 30 fps recording and stride-6 cache.

- `visibility-comparison.mp4`: eight short native-rate clips, 968 output frames
  in total. Clips are concatenated; the footer retains original source frame
  and relative MP4 time.
- `contact-sheet.png`: all eight cases at +0.6 seconds.
- `frames/*-comparison.png`: 48 still comparisons at -0.2, 0, +0.2, +0.6, +1.2
  and +2.4 seconds, plus exact full-size source PNGs and eight earlier images.
- `manifest.json`: exact source/model/cache/code identities, earlier-image
  provenance, crop bounds and all raw/enhanced detector proposals. These are
  predictions, not corrected boxes, reviewed onset labels or measured color.

The current diagnostic is post-hoc: Run 5 has already been inspected. It
cannot establish a new blind-test improvement. See the
[verification record](../../replay/verification/liquid-visibility-run5.json).

The frozen model proposes liquid on 9/48 raw frames versus 14/48 CLAHE frames,
with five added proposals and none lost. This is proposal presence, not recall
or accuracy: no independent box labels were used to verify those additions.
Fresh raw inference reproduces the original cached proposals exactly on all
48 frames. Neither variant proposes liquid at the selected 0 or +0.2 second
samples in any case. CLAHE therefore does not solve the earliest pour in this
pilot. The difference view makes stream/accumulation changes easier to inspect,
while horizontal edges are cluttered by labels and glass. Test the temporal cue
against reviewed onset/negative clips before using it as a detector.

Official references: OpenCV [CLAHE](https://docs.opencv.org/4.x/d5/daf/tutorial_py_histogram_equalization.html),
[background subtraction](https://docs.opencv.org/4.x/d1/dc5/tutorial_background_subtraction.html)
and [edge detection](https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html).
