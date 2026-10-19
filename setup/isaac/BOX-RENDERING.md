# Camera-matched box rendering: Cameron / Hayden handoff

This is a **12-image review batch**, not a released training set. It exercises
native GB10 Isaac Sim 6.0.0.1, the supplied fill-station camera, separate vial,
liquid and stream objects, geometric visibility masks, and YOLO box export.
All examples have `training_eligible: false` until appearance and label policy
are reviewed. It does not change the replay detector, measurement QA, labels,
recipe comparisons, or existing renderer. No industrial connection is involved.

## Run on the GB10

The shared onboarding copy is `~/ai-club-onboarding`. Use your own SSH key and
coordinate GPU use; the existing runner holds the team GPU lock. From a current
checkout on Windows, deploy updated setup and the supplied twin if necessary:

```powershell
pwsh -File setup/deploy.ps1 -Target Spark -IncludeTwin
ssh msoe-spark
```

The assets remain separately supplied data, outside Git. The renderer requires
the original folder structure under
`datasets/mixing_twin/assets/csi_digital_twin_v1`. It reads only that asset tree;
it does not depend on or edit the mentees' existing `mixing` renderer code.
Alternatively pass another supplied asset root with `--asset-root` directly.

On the GB10:

```bash
cd ~/ai-club-onboarding
export TEAM_MEMBER=cameron  # Hayden: use hayden
export OMNI_KIT_ACCEPT_EULA=YES
source setup/env.sh
bash setup/isaac/render_boxes.sh "$AI_CLUB_HOME/results/$TEAM_MEMBER/boxes-review-001" 12
```

Read the license guidance in the parent setup README before accepting. This
uses the already installed environment, not Docker or an environment update.
Always choose a new output directory. On failure, `failure.json` and the console
trace explain where it stopped; an absent `run.json` means capture is incomplete.
The launcher fails if either rendering or validation fails. Success also requires
`verification.json` with `passed: true`.

For debugging just three cases, use a different output path and `3` instead of
`12`. The fixture is `box_fixture.json`. Change a copy for future experiments;
preserve this fixture as a repeatable small test. Direct rendering options:

```bash
bash setup/spark/run.sh isaac setup/isaac/render_box_batch.py \
  --out "$AI_CLUB_HOME/results/$TEAM_MEMBER/boxes-review-002" \
  --fixture setup/isaac/box_fixture.json --limit 12 --subframes 16
bash setup/spark/run.sh data setup/isaac/review_box_batch.py \
  "$AI_CLUB_HOME/results/$TEAM_MEMBER/boxes-review-002"
```

From Windows, copy the folder back, then open the PNG files directly:

```powershell
scp -r msoe-spark:ai-club-tools/results/cameron/boxes-review-001 C:/Users/Austin/Downloads/
```

Substitute your actual local destination and member name. No web server is used.

## Review these files first

* `contact-rgb.png`: all normal rendered images, without boxes.
* `contact-boxes.png`: the same images with boxes.
* `contact-masks.png`: green vial / orange liquid / cyan stream geometric masks.
* `overlays/<case>-review.png`: full-size RGB, boxes, and masks side by side.
* `label-passes/<case>-vial-labels.png` and `...-content-labels.png`: annotation
  render passes, deliberately different from the normal RGB image.
* `camera-alignment.png`: optional real/render/50% overlay comparison when the
  local reviewer is given the Run 1 reference frame. This is visual review, not
  a camera optimization or a certified calibration.

To make the alignment sheet, extract zero-based frame 2100 (70.000 MP4 seconds)
from Run 1 without resizing. From the repository on Windows (adjust destinations):

```powershell
& "$HOME/ai-club-tools/envs/data-windows/Scripts/python.exe" setup/isaac/prepare_alignment.py --video datasets/dataset2/csi-vial-fill-dataset/vial_fill/videos/vial_fill_video_1.mp4 --out C:/path/to/run1-frame2100.png
& "$HOME/ai-club-tools/envs/data-windows/Scripts/python.exe" setup/isaac/review_box_batch.py C:/path/to/copied/batch --real-frame C:/path/to/run1-frame2100.png
```

## Camera and geometry contract

The renderer uses the supplied `cameras/FillCam.json` and matching USD camera
prim, native **640 x 480**, pinhole K with fx=fy=728.76768 px, principal point
(320,240). It checks the actual camera pose, focal length and aperture against
the JSON; it does not move, crop, or resize the camera. Station animation is
frozen at source time code 686 in a session layer. Source files are never saved;
all asset SHA256 values are compared before and after capture.

USD focal length/aperture values use tenths of a stage length unit, not always
millimeters. The check converts using `metersPerUnit` before comparing with the
camera JSON; focal length and aperture ratios are preserved.

The supplied twin was made for this view. Matching pose/intrinsics to its JSON
does **not** prove a fresh physical camera calibration. Run 1 frame 2100 is only
a visual alignment reference. Run 5, its boxes, timing labels and comparisons
are not inputs to this renderer or fixture.

Liquid is an upright cylinder at the frozen source liquid's world-space
envelope, with `level` specifying a fraction of that envelope's height. This is
not a calibrated mL measurement. Stream is a 1.5 mm radius cylinder, from the
source nozzle bottom down to the liquid surface. The first-stream case is only
12 mm long below the nozzle with no accumulated liquid. These are static
snapshots, not CFD, droplets, meniscus, bubbles, stratification or motion blur.
They do not establish temporal tracking performance.

Beauty materials use linear diffuse RGB and opacity proxies. Pale/clear cases
are deliberate stress examples; they are not validated optical models of clear
water or dye absorption. The background, glass and reflections are still the
supplied approximations.

## What the boxes mean, especially through transparent plastic

Replicator `instance_segmentation_fast` supplies the per-pixel instance planes.
Each target mesh has a class semantic label. Saved runtime instance IDs are
mapped back to exact USD prim paths; source object identities are separate and
stable in the manifest. Every visible supplied vial, including neighbors and
partial edge vials, is included rather than silently becoming a negative.

Two label passes are necessary because the vial and contents overlap:

1. **Vial labels:** vial shells become opaque; liquid and stream proxies are
   hidden. Opaque holder, nozzle and paper/barcode remain occluders.
2. **Content labels:** vial shells and the transparent splash guard are hidden;
   liquid/stream proxies become opaque. Opaque holder, paper/barcode and nozzle
   still occlude them.

The splash guard is hidden in both label passes. Normal RGB is captured with
original vial/guard appearance and configured liquid appearance before these
overrides; RGB bindings/visibility are restored between cases.

These masks describe **geometric visibility through transparent shells**, not
the exact refracted liquid pixels in RGB. Reflection, transparency, refraction,
anti-aliasing and subpixel visibility can differ. Do not present them as reviewed
real segmentation masks or use them as color measurement masks. NVIDIA documents
that transparency settings affect segmentation behavior:
https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator/annotations_with_transparency.html

`bbox_xyxy` is computed directly from nonzero pixels in that object's saved
binary mask. Coordinates are zero-based **half-open** `[x0,y0,x1,y1]`; x1/y1
are maximum pixel + 1. Vial and content masks can overlap across passes.
`projected_geometry_bbox_xyxy` is a separately calculated projection of the
full object geometry, before clipping/occlusion, using the supplied camera. It
is retained for audit and **is not exported as the YOLO target**. A missing
visible mask gets no box; absent contents never get a fabricated box.

## Machine-readable outputs

`manifest.jsonl` has one record per image:

| Field | Meaning |
| --- | --- |
| `schema_version`, `sample_id` | Contract version 1; deterministic fixture case ID |
| `parameters` | Fixture RGB/opacity, level fraction, stream mode |
| `image_file`, `image_sha256` | Relative normal RGB PNG path and integrity hash |
| `annotations` | One object record per vial/liquid/stream, including absent contents |
| `annotations[].class_name` | `vial`, `liquid`, or `stream` |
| `annotations[].instance_id`, `prim_path` | Stable scene identity and exact target mesh |
| `annotations[].present`, `status` | Present in scene; visible proxy / absent / occluded-or-outside-view |
| `annotations[].mask_file`, `mask_sha256`, `visible_pixels` | Native binary PNG and pixel count |
| `annotations[].bbox_xyxy` | Visible geometric half-open pixel box, or null |
| `annotations[].projected_geometry_bbox_xyxy` | Unclipped full-geometry projection, or null when absent |
| `annotations[].label_pass`, `bbox_convention` | Which pass made the mask; coordinate definition |
| `label_policy`, `geometry` | Transparency/occlusion policy and actual proxy dimensions in meters |
| `training_eligible`, `physical_optical_calibration` | Both false for this batch |

`labels/<case>.txt` contains normalized YOLO boxes with class IDs **0 vial,
1 liquid, 2 stream**. These files are a convenience export, not approval to train.
`label-passes/*-ids.npy` and `*-mapping.json` preserve raw renderer instance IDs.
`camera.json` stores actual and supplied camera parameters. `run.json` stores
runtime, host, fixture/script/assets hashes, source preservation, and completion.
`verification.json` records integrity, projection and negative-case checks.

## What Cameron and Hayden should do next

First review RGB and label passes together: do vial boundaries line up, do tiny
fills show up, does the stream stop at the surface, and are holder/label
occlusions reasonable? Record case IDs and corrections needed. Do not scale a
batch with systematically incorrect or imperceptible targets.

Then improve glass/liquid/background materials and real-camera appearance,
confirm the visible-versus-full-object box policy, and only then vary nuisance
conditions around the fixed camera. Keep scene parameters, seed, source hashes
and class definitions with every image. Retain empty/glare negatives and
deliberate stream-only examples. Add temporal sequences with stable scene
instance IDs separately if tracking evaluation is needed.

Austin owns detector ingestion and real-only versus mixed-data evaluation.
Keep Run 5 out of rendering optimization and fitting; the repeatedly inspected
Run 5 is diagnostic, not a fresh blind validation set. The labeling group still
owns reviewed real boxes/windows. No detection improvement or completed-vial
QA coverage improvement is claimed by rendering this batch.

Before mixed training, resolve one label-scope difference: the real bootstrap
teacher proposals targeted the **fill-position vial**, whereas this synthetic
export labels **all visible vials**. Agree on consistent real/synthetic labeling
or an explicit station/ignore-region policy; do not train with unlabelled
neighbor vials treated inconsistently as negatives.

## Executed evidence, October 19

The GB10 was reachable for source inspection at the start of this work. Campus
VPN/DNS connectivity then failed (`spark8.ai.msoe.edu` stopped resolving) before
the new scripts could be uploaded or run. **The GB10 box-rendering path is still
pending an actual execution.** The earlier 12-image GB10 onboarding smoke was
RGB only and is not evidence for this new annotation path.

As a separate debugging check, the existing local Windows Isaac 6.1 RC / RTX
4080 SUPER rendered the three-case smoke and then the complete 12-case fixture.
The latter took about 45 seconds including startup. Seven local tests passed;
the saved 12-case batch passed RGB/mask/ID/box/YOLO/projection and negative-case
checks. It contains 36 vial boxes, 10 liquid boxes and 8 stream boxes, with no
unexpected missing masks. A deliberate blue-to-pale material change produced
mean RGB difference 56.20 inside the common liquid region, guarding against a
stale appearance capture. This is not a photometric calibration metric.

Austin's local review folder is:
`C:/Users/Austin/ai-club-tools/results/isaac-camera-boxes-20261019/windows-review`.
Full RGBs, per-case comparisons, masks, native ID planes, camera metadata and
manifests are there. The adjacent `run1-frame2100-sequential.json` records the
reference video/image hashes and sequential frame extraction. A selected sheet
and executed report are also in `setup/verification/camera-boxes-windows.png`
and `setup/verification/camera-boxes-windows.json`.

Visual inspection of the real/render overlay shows approximate alignment, with
remaining neighbor-vial/holder/background differences. It does not establish an
exact physical camera match. The liquid/stream shapes and especially clear
appearance still require review. No scaling, training, or detector promotion
has occurred.

When the VPN is restored, deploy the scripts, run the exact GB10 launcher above
on the pinned 6.0.0.1 environment, copy back its outputs, and verify them before
claiming GB10 completion. Do not substitute these Windows results for that check.
