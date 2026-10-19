# Pale liquid and early-pour review, October 19

This is a small revision of the supplied-camera renderer following the
[RGB/target audit](../vision/SYNTHETIC-TRANSFER-AUDIT.md). It changes liquid
appearance and procedural geometry, while preserving the deterministic native
ID-to-mask-to-box export. It performs no training or model/measurement-QA changes.

The profile is opt-in: `renderer_profile: meniscus-transmission-v2` in a separate
fixture. The original `box_fixture.json` and legacy cylinder/opacity path remain
available. Source USDs are edited only in a session layer, never saved; the
supplied FillCam pose/intrinsics and 640x480 output remain unchanged.

## What changed

- The body and stream use separate MDL OmniSurfaceBase materials with transmission
  weight 1, diffuse weight 0, IOR 1.333, cutout opacity disabled, and finite-volume
  absorption depth. The thin stream can be less colored than the thicker pool.
  These are provisional parameters, not measured fluid properties. See NVIDIA's
  [transmission documentation](https://docs.omniverse.nvidia.com/materials-and-rendering/latest/templates/parameters/OmniSurface_Transmission.html).
- Body geometry is a closed mesh with a curved contact line, shallow impact
  depression and small surface ripples. Early states use a partial-radius wet
  patch, rather than a full-width cylinder at every nonzero level. Native
  positive height fractions now start at 0.002 instead of 0.02. Height fractions
  still are not calibrated mL.
- A separate closed stream mesh represents pendant drop, partial falling jet
  and a narrow wavy jet reaching the pool/base. Stream object 301 and body 201
  retain separate persistent parent semantics and separate masks/boxes.
- Meshes have explicit closed topology, bounds and corner normals. The initial
  nine-case native pilot exposed a vertex/corner normal-buffer warning. The
  final profile uses explicit face-varying normals; keep that fix on Isaac 6.0.
  Top, side and bottom normals are smoothed separately to avoid a spurious lens
  at the contact line. Full pools use 0.997 of the source fluid radius, just
  inside the supplied vial's inner wall; the first patches retain small radii.
  A read-only CAD inspection found source fluid radius 13.2029 mm and an inner
  vial ring near 13.18 mm. This is CAD geometry, not a physical measurement.

`liquid_geometry.py` contains the mesh/material implementation. `render_box_batch.py`
keeps camera, asset-preservation, annotation mapping, projection, absent-object,
mask and export guards. Body/stream are never identified by observed color.
The `rgb` and `opacity` fixture keys remain only for legacy schema compatibility;
the revised material uses `transmission_color` and `absorption_depth_m` instead.

## Reference boundary and case inventory

`prepare_pale_review.py` copies nine **RGB-only** training references: three each
from Runs 1, 2 and 3. Run 1/2 show thin colored jets and early/partial blue pools;
Run 3 shows pale yellow-green pools and curved surfaces. Each copied image is
checked against the frozen real inventory and recorded by source ID/run/SHA256.
No teacher boxes, timing/event labels, recipes, Run 4 or Run 5 pixels are inputs.
The fixture validator rejects revised appearance references from other runs.
This is qualitative visual guidance, not a numerical optical/camera fit.

18 cases cover:

| IDs | Contents |
| --- | --- |
| `v2-00` | Empty vial, neither body nor stream target. |
| `v2-01`–`v2-03` | Pendant drop, partial jet, jet reaching the base. Stream target only; no accumulated body. |
| `v2-04`–`v2-07` | Localized first patch, small/growing pale pool, low pale fill. Separate body and stream. |
| `v2-08`–`v2-09` | Mid/high pale fill, with/stopped stream. |
| `v2-10`–`v2-11` | Low/mid clear transmission controls. |
| `v2-12`–`v2-13` | Low/mid blue controls. |
| `v2-14`–`v2-16` | Stronger impact/surface ripples and a stopped pale body. |
| `v2-17` | Weaker absorption at the small-pool geometry, to bracket pale observability. |

These are deliberately chosen static states, not a CFD trajectory, measured
flow timing or a tracking sequence. No bubble interiors, splashes, wall films,
real-camera noise/blur, glass/background calibration or randomized paper-label
orientation are added in this revision. Very small/clear pools can remain hard
to see; do not force a visible boundary by changing their geometric labels.

## Prepare and render

Use a fresh preparation directory on Windows:

```powershell
$python = "$HOME/ai-club-tools/envs/data-windows/Scripts/python.exe"
$prepared = "$HOME/ai-club-tools/results/pale-early-render-review-new"
& $python setup/isaac/prepare_pale_review.py --real "$HOME/ai-club-tools/results/real-synthetic-detector-20261019/data/real" --out $prepared
```

On the GB10, use a private experiment directory and copy these exact files into
it: `render_box_batch.py`, `liquid_geometry.py`, `box_annotations.py`,
`review_box_batch.py`, and the prepared `fixture.json`. Preserve the executed
source snapshot with the result. No environment/container update is required.
Use the existing setup runner and team GPU lock; coordinate GPU use normally.

```bash
export TEAM_MEMBER=cameron  # or hayden
export OMNI_KIT_ACCEPT_EULA=YES  # use the existing onboarding/license guidance
cd ~/ai-club-tools/experiments/your-private-render-review
runner="$HOME/ai-club-onboarding/setup/spark/run.sh"
out="$HOME/ai-club-tools/results/$TEAM_MEMBER/pale-early-review-new"
bash "$runner" isaac render_box_batch.py --asset-root "$HOME/ai-club-onboarding/datasets/mixing_twin/assets/csi_digital_twin_v1" --fixture fixture.json --out "$out" --limit 18 --subframes 4 --pt-spp 32
bash "$runner" data review_box_batch.py "$out"
```

For unstable VPN, put the render/validation commands in a shell script and
launch it with `nohup bash launch-review.sh > review.log 2>&1 < /dev/null &`.
Wait for complete `run.json` and `verification.json` with `passed: true`, not
just PNGs or a process ID. Preserve the log and any failed pilots separately.

Copy the complete result directory back to Windows. Generate local comparison
PNGs against the original verified GB10 renders:

```powershell
& $python setup/isaac/compare_pale_review.py --batch C:/path/to/copied/batch --prepared $prepared --legacy "$HOME/ai-club-tools/results/real-synthetic-detector-20261019/renders"
```

This revalidates native masks, raw semantic IDs, boxes/projection, absent states,
three-class YOLO export, RGB hashes and dimensions before making review sheets.
Additional revised checks cover pre-pool body absence, growing pool footprint
and a deliberate pale-versus-blue RGB response. The last is a stale-capture
check, not physical optical validation.

## Local review files and handoff

Austin's preparation root:
`C:/Users/Austin/ai-club-tools/results/pale-early-render-20261019/final-prepared/`.
Final GB10 output is copied alongside it to `../gb10-review-final/`. Open these PNGs
directly; no preview server/browser is needed:

- `early-pour-rgb.png`: empty → droplet/jet → first patch → low pool.
- `early-pour-boxes.png`: the same states with separate geometric targets.
- `pale-before-after.png`: Runs 1–3-guided revised appearance versus real pale
  references and old opacity proxies. Volumes/hues are illustrative, not paired.
- `clear-geometry-review.png`: RGB/boxes/native body mask for clear material;
  emphasizes the distinction between geometry and visible evidence.
- `runs123-reference-cues.png`: selected original RGB cues from all three runs.
- `contact-rgb.png`, `contact-boxes.png`, `contact-masks.png`: all 18 cases.
- `overlays/<case>-review.png`: full-size RGB, boxes and geometric masks.

`appearance-review.json` schema 1 lists reference IDs/runs/hashes and each
case's RGB/hash, native body annotation and native stream annotation. The
annotations retain mask path/hash, half-open XYXY bbox, projected geometry,
visible-pixel count, semantic identity and label-pass provenance. Paths are
relative to the batch. `review_status` stays pending; `training_eligible` and
`human_appearance_approved` stay false. `run.json` additionally records the MDL
source/hash, material policy, helper/fixture/renderer hashes, native runtime,
sampling settings, camera/source preservation and reference boundary.

Cameron/Hayden should review named cases against the supplied RGB references:
does pale fluid transmit rather than look milky, is the tiny footprint plausible,
does the stream meet the pool, and which clear boundaries remain unobservable?
Record objections and parameter changes by case ID before scaling. The label
team still owns reviewed real body/stream boxes and visibility states.

Opaque label passes still describe **geometric visibility through transparent
shells**, not exact refracted RGB segmentation or verified color pixels.
Keep body/stream separate even when the geometric boxes overlap at impact.
The raw export includes all visible vials (class 0), accumulated body (1), and
stream/drop (2); a future training plan must explicitly reconcile active-vial
scope with the real labels. No existing training plan admits this new renderer
automatically: preserve its frozen source/hash checks and make a new reviewed
plan only after appearance/label review. No transfer improvement is claimed.

## Executed final batch

The final **18** cases ran on `spark8` / GB10 / Isaac **6.0.0.1** / PathTracing
at 640x480, 32 SPP and four subframes, in **87.92 seconds** including startup.
Remote and copied-local native validation passed: **54 vial, 14 body, 13 stream
boxes**, zero missing required masks and zero annotation synchronization retries.
All absent bodies/streams have empty masks. The first patch is 155 native
pixels with bbox `[303,393,320,407]` (17x14), versus the old 2%-cylinder's
1,158 pixels / 44x36 box. This demonstrates the geometry change, not equivalent
physical volumes or improved real detection.

The pale/blue material-response check passed; actual source, fixture, helper and
archive hashes were verified after download. The final log has no error entries,
corrupt-normal warnings or traceback. Eleven local tests pass, covering closed
oriented geometry, reference exclusions, pre-Kit imports and annotation integrity.
The source USD/camera hashes stay unchanged. `source/` and `render.log` preserve
the executed code/fixture/CAD inspection; `references/` contains the nine RGB
references so Cameron/Hayden can review the bundle without Austin's Windows paths.

**Appearance remains pending review.** The earliest pale pools and thin jets
are faint, and the clear/mid-fill rim highlights still differ from the real
camera. Do not interpret geometric visibility as successful RGB observability,
or native export validation as training readiness. The locally saved comparisons
make these remaining gaps visible without changing or suppressing their labels.

Use the `gb10-review-final` folder, not the earlier `pilot-v1` or `gb10-review`
iterations. The final remote source is
`~/ai-club-tools/results/austin/pale-early-v2-review-final-20261019`.
Published execution evidence is `setup/verification/pale-early-gb10.json`.
