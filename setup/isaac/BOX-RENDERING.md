# Camera-matched box rendering: Cameron / Hayden handoff

This documents the initial **12-image review batch**, not a released training set. It exercises
native GB10 Isaac Sim 6.0.0.1, the supplied fill-station camera, separate vial,
liquid and stream objects, geometric visibility masks, and YOLO box export.
All examples have `training_eligible: false` until appearance and label policy
are reviewed. It does not change the replay detector, measurement QA, labels,
recipe comparisons, or existing renderer. No industrial connection is involved.

**Current status:** the complete 12-image batch rendered and passed export
verification on GB10 Isaac 6.0.0.1. Local review images are ready; review them
before scaling or training. Interrupted diagnostic batches remain ineligible.

## Run on the GB10

The shared onboarding copy is `~/ai-club-onboarding`. Use your own SSH key and
coordinate GPU use; the existing runner holds the team GPU lock. From a current
checkout on Windows, deploy the updated setup (the assets are already on Spark 8):

```powershell
pwsh -File setup/deploy.ps1 -Target Spark
ssh msoe-spark
```

This deployment leaves the mentees' existing twin code and assets in place.
On a new machine, obtain the supplied asset bundle separately. The assets remain
separately supplied data, outside Git. The renderer requires
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

Replicator `semantic_segmentation` supplies the default per-pixel label planes.
Each persistent object Xform has a unique scene-authored semantic tag, such as
`vial__101`, `liquid__201`, or `stream__301`. Its child geometry can change
visibility/material without deliberately removing the object's identity. The
exporter matches the renderer's explicit semantic label table to those tags;
it does not guess numeric IDs, infer object identity from color, or use a model.
YOLO class IDs remain **0 vial / 1 liquid / 2 stream**; the unique object tags
are an annotation implementation detail, not extra model classes.

`--label-annotator instance_segmentation` or `instance_segmentation_fast` remain
available for diagnostics. Both showed missing prim-path mappings on the
GB10, even though the liquid pixels existed in the renderer plane. Legacy
path-based outputs require valid absolute paths for all rendered object IDs;
the default semantic output requires explicit nonempty object tags. Ambiguous
metadata is rejected. The selected annotator and `annotation_encoding` are
recorded in `run.json`. The original successful Windows preview used the fast
path-based variant; the reader supports both encodings.

Source object identities and exact geometry paths are separate and stable in
the manifest. Every visible supplied vial, including neighbors and
partial edge vials, is included rather than silently becoming a negative.
The review fixture requires a visible mask for every present object. An
absent liquid or stream must have an empty mask. After a label-pass visibility
change, up to four captures of the **unchanged** scene can be attempted for
metadata synchronization. Missing labels are never accepted; unresolved gaps
abort the batch. `annotation_capture` records attempts and rejected raw ID/map
files. This does not change measurement QA or mask/box acceptance requirements.

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
| `annotations[].semantic_label`, `label_prim_path` | Unique renderer tag and persistent object Xform |
| `annotations[].present`, `status` | Present in scene; visible proxy / absent / occluded-or-outside-view |
| `annotations[].mask_file`, `mask_sha256`, `visible_pixels` | Native binary PNG and pixel count |
| `annotations[].bbox_xyxy` | Visible geometric half-open pixel box, or null |
| `annotations[].projected_geometry_bbox_xyxy` | Unclipped full-geometry projection, or null when absent |
| `annotations[].label_pass`, `bbox_convention` | Which pass made the mask; coordinate definition |
| `label_policy`, `geometry` | Transparency/occlusion policy and actual proxy dimensions in meters |
| `require_visible_present_objects`, `annotation_capture` | Fixture visibility requirement; capture attempts and rejected diagnostics |
| `training_eligible`, `physical_optical_calibration` | Both false for this batch |

`labels/<case>.txt` contains normalized YOLO boxes with class IDs **0 vial,
1 liquid, 2 stream**. These files are a convenience export, not approval to train.
`label-passes/*-ids.npy` and `*-mapping.json` preserve raw renderer ID planes and
their semantic label tables (legacy outputs use instance prim-path tables).
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

The subsequent October 19 experiment uses a consistent **fill-position vial**
target in both real and synthetic training copies: synthetic object 101 maps
to the existing vial class and object 201 to accumulated liquid. Neighbor
vials and streams remain in the raw three-class export. The station-specific
training scope and explicit admission of provisional proxies are documented
in [the mixed and sim-only experiment](../vision/REAL-SYNTHETIC.md).

## Executed evidence, October 19

The GB10 (`spark8`, NVIDIA GB10 driver 580.178.04) rendered the complete 12-case
fixture on pinned Isaac **6.0.0.1 / PathTracing / 640 x 480** in **99.17 seconds**
including startup. Export verification passed remotely and again after copying
back to Windows. There are **36 vial, 10 liquid and 8 stream boxes**, no missing
expected masks, and **zero synchronization retries** in this accepted batch.
Nine tests also passed on Windows and GB10.

The exported boxes equal the saved masks' bounds; masks equal the saved native
ID planes decoded through explicit semantic tags; boxes lie within independent
camera projections. Empty/stream-only cases and liquid-height progression pass.
The deliberate mid-fill blue/pale change produced mean RGB difference **56.35**
in the shared liquid region (a stale-capture check, not optical calibration).
The camera and all supplied asset hashes stayed unchanged. Executed renderer,
helper and fixture hashes match the current setup files. Run 5 was not used.

Austin's **GB10 review output** is:

`C:/Users/Austin/ai-club-tools/results/isaac-camera-boxes-20261019/gb10-review`

The remote source batch is:

`~/ai-club-tools/results/austin/camera-boxes-gb10-parent-tags-20261019`

Start with `contact-boxes.png`, `camera-alignment.png`, and
`overlays/03-tiny-pale-review.png`. The folder contains all RGBs, per-case
comparisons, masks, raw label planes/maps, camera metadata, manifest and YOLO
labels. `real-run1-frame2100.png` and its JSON record preserve the real alignment
reference and video/image hashes. The original downloaded archive remains next
to the review folder. Published evidence is in
`setup/verification/camera-boxes-gb10.json` and `camera-boxes-gb10.png`.

Visual inspection shows boxes following the rendered vials, liquid bodies and
streams, including the partial edge vial. Real/render alignment is approximate;
neighbor-vial/holder/background differences remain. Liquid/stream shapes and
especially clear appearance still require review. No scaling, training or
detector promotion has occurred. The previous Windows 6.1 RC preview remains
in the adjacent `windows-review` folder, with its separate evidence; it is not
being substituted for the GB10 execution.

### Runtime fixes to preserve

Keep render preflight standard-library-only: importing NumPy before
`SimulationApp` stalled native startup on this pinned ARM runtime. Deferring it
resolved startup; the test suite checks this boundary. Retain the runner's
existing `setarch -R` / libgomp workaround as well.

The first GB10 attempts exposed incomplete instance-to-path tables and lost
leaf-mesh labels after visibility changes. Liquid pixels existed even when
metadata could not identify them. Both path-based annotator variants were
rejected. The working path uses explicit unique semantic tags on **persistent
object Xforms**, with the renderable mesh/cylinder as a child. Do not move the
identity labels back onto transient leaf renderables. The accepted complete
run required no retries. `camera-boxes-gb10-diagnostic.json` preserves an earlier
failure's evidence and points to the resolved result; those diagnostic exports
must never be used as training examples.

The VPN disconnected repeatedly during debugging. The successful capture ran
as a detached GB10 job, which survives the SSH connection closing. For longer
runs, after exporting the environment variables in the launch section:

```bash
source setup/env.sh
out="$AI_CLUB_HOME/results/$TEAM_MEMBER/boxes-review-003"
log="$AI_CLUB_HOME/logs/$TEAM_MEMBER-boxes-review-003.log"
nohup bash setup/isaac/render_boxes.sh "$out" 12 > "$log" 2>&1 < /dev/null &
tail -f "$log"
```

A dropped VPN still prevents copying results back until it reconnects. Check
both `run.json` and `verification.json`; a process launch or captured PNG alone
is not completion. A second complete repeat was not executed after the final
copy because connectivity dropped again; the evidence above describes one
complete verified GB10 batch plus a preceding three-case GB10 smoke.

## Subsequent 768-image experiment

Austin explicitly requested scaling and provisional training after the initial
review. The seeded 768-image batch ran on the same pinned GB10 runtime in
2,689.24 seconds. Remote and local verification passed: 768 unique RGB frames,
2,304 vial boxes, 576 liquid-body boxes and 497 stream boxes, with no unavailable
required objects or synchronization retries. Source asset hashes match the
preceding GB10 pilot. Camera and source assets remain unchanged.

The scale profile uses effective 32 SPP, total SPP 32, one subframe, and compact
exports retaining native ID planes, mappings and masks. It preserves the same
mask/box checks. Raw `training_eligible` and optical-calibration flags remain
false. These are geometric proxies admitted for the requested experiment.
See [REAL-SYNTHETIC.md](../vision/REAL-SYNTHETIC.md) for both training arms,
the additional real-only control, checkpoint policies and local comparison outputs.
