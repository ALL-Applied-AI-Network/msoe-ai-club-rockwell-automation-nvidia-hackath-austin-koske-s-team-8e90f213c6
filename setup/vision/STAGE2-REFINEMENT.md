# Stage 2 refinement: which components are useful?

October 20, 2026. Development branch: `austin-work`. This follows the
[first Stage 2 experiment](STAGE2.md). The selected next-pilot architecture is:

**Stage 1 sampling → background-referenced sRGB color → current recipe, phase and
metered-volume reference → source/camera/geometry checks → three-sample deviation
→ advisory hold/review and the existing proposed recovery coordinator.**

No new detector training, approved labels, recipe truth or industrial connection
was introduced. The deployed replay defaults and measurement QA remain unchanged.
The new modules are opt-in experiments, not a commissioned inspection system.

## Findings

| Question | Measured answer |
| --- | --- |
| Can background references reduce camera-induced feature changes? | Yes, in controlled real-pixel tests. Encoded green gain: p95 paired dE 5.22 raw → 0.68 sRGB-normalized. |
| Does normalization erase a liquid-only change? | Most signal remains: 96.3% median paired-distance retention for the stronger controlled shift. Fixed-threshold sensitivity changes and needs commissioning. |
| Is elapsed time sufficient for the expected mixture? | No. Simulated ±20%/±40% flow variation causes 20/24 and 22/24 null warnings with the clock template. Metered lookup gives 0/24 in both. |
| Is one endpoint color per ingredient sufficient? | No. It flags 24/24 null episodes because the mixture changes while an ingredient is added. |
| Does the combined candidate remain useful under multiple nuisances? | On fresh simulator seeds: 0/24 null warnings, 72/72 two-mL injection warnings, 45/72 half-mL injection warnings; all 168 episodes have assessable samples. |
| Can this establish a confirmed real off-recipe vial? | No. The scoped local inventory has no documented acceptable reference preparation, completed physical composition trial or non-demo review result. |

The simulation counts are **not real defect accuracy**. Nonzero simulated residue
is an injected condition, not a product tolerance. The real-pixel tests measure
invariance conditional on frozen unreviewed geometry, not absolute color accuracy.

## Real-image camera experiment

Nine fixed 48×48 background candidates were scored using 1,478 one-Hz frames from
Runs 1–3. Selection uses within-run camera-feature stability, clipping and darkness,
never liquid labels or Run 5. The selected patches are `[192,224,240,272]`,
`[112,224,160,272]`, `[112,160,160,208]` in native source XYXY pixels.

Run 4's first five seconds provide the per-camera anchor baseline (25 samples,
before accepted liquid sampling). All 849 frozen QA-passing Run 4 liquid samples
were reconstructed exactly from their source RGB/masks; 874 development frames,
including startup, passed source-pixel/native-PTS checks.

The final experiment has **11 conditions × 3 methods × 849 samples = 28,017 paired
evaluations**. Perturbations are applied to actual pixel values, with eight-bit
quantization and 0.4 RGB-count noise on changed conditions. Source images, masks
and labels are unchanged; detector/SAM inference is not rerun. Each method is
compared with its own unperturbed same-frame feature.

| Controlled whole-frame change | Raw p95 paired dE | sRGB gain correction | Linear-light gain correction |
| --- | ---: | ---: | ---: |
| Linear green +12% | 3.11 | 0.98 | 0.68 |
| Linear white-balance gain | 1.84 | 0.89 | 0.86 |
| Linear exposure −30% | 4.49 | 1.14 | 0.93 |
| Linear exposure +30% | 3.80 | 0.92 | 0.79 |
| Encoded sRGB green +10% | 5.22 | 0.68 | 1.37 |
| Encoded black offset +3 | 1.27 | 0.92 | 0.90 |

sRGB correction has the lower worst-case p95 across these six tested nuisance
families (1.14 versus 1.37). Linear correction is better when the change is truly
multiplicative in linear light. **sRGB correction is selected for the next pilot**
as a development choice; camera-response commissioning can change that selection.

The health check is identical for all methods and uses sRGB anchor-ratio agreement.
Its training-informed threshold is p99 plus 0.005, bounded to [0.01, 0.05]. Training
p99 is 0.0897, so the engineering cap of **0.05** is active; this is not a claim of
99% training acceptance. All 849 nominal Run 4 liquid samples pass this gate.
One- and two-anchor shadow tests cause 0/849 to remain eligible, explicitly
withholding correction-based decisions when anchors disagree.

The initial `camera-reference-v1` used unquantized transforms and a three-percent
gate in each correction domain, making health coverage incomparable (sRGB versus
linear). It is retained as exploratory evidence; `v2` adds quantization/noise and
one shared, training-informed gate. The old measurements were not overwritten.

For the stronger liquid-only perturbation, raw feature distance exceeds 5 in
832/849 frames. sRGB correction retains 783/832 crossings and 96.3% median paired
signal; linear correction retains 769/832 and 97.1% median signal. The smaller
shift retains 10/10 crossings with either method. These are controlled pixel
changes, not independently adjudicated defects. Absolute thresholds/references
must be commissioned under the selected feature profile; do not reuse the old bank.

Background anchoring cannot distinguish an actual composition change from a
lighting change confined to the vial. Reflections, local illumination, dark/pale
liquid, and absolute color/composition accuracy remain unvalidated.

## Reference progression and input health

The first follow-up decision study contains 1,344 simulated episodes and 6,720
policy evaluations, with nuisance seeds 11/12 (the earlier study used 0/1).
There are 24 null plus 144 injected episodes per condition. Threshold 5 and three
samples remain fixed engineering settings.

| Condition | Raw null warnings | Guarded sRGB null warnings | Guarded sRGB injected warnings / assessability |
| --- | ---: | ---: | --- |
| Nominal | 0/24 | 0/24 | 119/144; all assessable |
| Linear white balance | 2/24 | 0/24 | 119/144; all assessable |
| Linear green gain | 15/24 | 0/24 | 119/144; all assessable |
| Encoded green gain | 24/24 | 0/24 | 119/144; all assessable |
| Encoded black offset | 0/24 | 0/24 | 120/144; all assessable |
| Reference-anchor shadow | 0/24 | 0/24 | 0/144; all blocked |
| Truthfully timestamped one-second stale process input | 24/24 | 0/24 | 0/144; all blocked |
| Wrong progression falsely labeled current | 24/24 | 24/24 | 144/144; all assessable |

Abstention is not detection: the shadow/stale rows withhold all 168 episodes.
The last row remains a failure and requires verification of actual PLC-to-video
alignment; metadata checks alone cannot establish that the process data are true.
Missing/null/nonfinite timestamps now produce unavailable results rather than
crashing. A blocked sample breaks consecutive evidence while preserving a
previously latched review requirement. Normalization cannot approve unreviewed
geometry or supply an unknown recipe.

A separate progression experiment (288 episodes / 864 evaluations) varies
individual flow rates while preserving metered ingredient quantities; pauses
vary from one to four seconds in the variable-flow conditions. A reference bank
is keyed by recipe, ingredient phase, camera profile and cumulative metered volume,
with 81 predeclared Lab knots per phase. It never aligns by observed liquid color.
Missing/incompatible/future profiles and out-of-range meter values are unavailable.

| Simulated flow variation | Clock-template null warnings | Phase-endpoint null warnings | Metered-bank null warnings |
| --- | ---: | ---: | ---: |
| None | 0/24 | 24/24 | 0/24 |
| ±20% | 20/24 | 24/24 | 0/24 |
| ±40% | 22/24 | 24/24 | 0/24 |

All three strategies warn on 72/72 two-mL injected cases per condition; the
clock/endpoint strategies also warn on normal trajectories. The bank contains
only simulated profiles: no actual recipe has been commissioned into it.

## Combined candidate on fresh simulator seeds

Seeds 31/32 combine ±30% flow variation, one-to-four-second pauses, per-channel
camera gains [0.85,1.15] in linear or encoded space, five-percent measurement
dropouts and periodic reference-anchor occlusion. The two compared strategies use
the same appearances/noise. All 168 episodes remain assessable at some point.

| Strategy | Null warnings | Two-mL injections warned | Half-mL injections warned |
| --- | ---: | ---: | ---: |
| Raw camera + clock template | 23/24 | 72/72 | 70/72 |
| sRGB anchors + metered bank + guards | **0/24** | **72/72** | **45/72** |

The candidate warns on 117/144 injections overall (81.25%). The baseline's high
injection warning count is accompanied by 23/24 null warnings. Do not interpret it
as superior defect detection. The candidate blocks 343 unhealthy-camera samples
and 1,813 failed-measurement samples while retaining assessable evidence in every
episode. The same analytic family generates references and observations, and the
simulator provides ideal truthful process cues. New seeds are not a real-world
generalization test or a measured production false-hold rate.

## Evidence needed for the next real test

The scoped audit found nine unfilled standards worksheets and 18 unfilled physical
trial rows. No row records preparation plus a camera file, or a measured-composition
method plus image/time. Four existing review databases were opened read-only:
three demonstration revisions and zero eligible spatial revisions were present.
This does not establish that data are absent from every possible user directory.
Spatial reviews would still not establish product acceptance or chemical purity.

A concrete CSI request can now be small and specific:

1. Identify the actual Run 4 recipes and any recipe-selection, per-tank metering,
   stage and completion records. Confirm clock relationships and physical vial IDs.
2. Supply independently prepared/verified acceptable mixtures for two existing
   approved recipes, imaged under the same camera conditions. Record preparation
   method, actual volumes, uncertainties and camera settings.
3. Observe a CSI-approved A→B changeover and an approved flush comparison with
   fresh conditioning before each arm. Record actual vial identity, Teledyne
   color/weight verdict and routing deadline, plus intervention receipts. CSI
   chooses recipes, conditioning criteria, flush settings and permitted feed.

The first objective is one independently confirmed mismatch and its decision timing;
later independent sequences estimate missed-unacceptable-vial and unnecessary-hold
rates. The earlier local worksheet is a proposed experiment, not an executed one.
No request was sent or hardware access attempted by these experiments.

## Artifacts and checks

Local root: `C:/Users/Austin/ai-club-tools/results/stage2-refinement-20261020`.

- `reference-evidence-audit-v1/`: read-only preparation/review inventory.
- `camera-reference-v2/`: source identities, selected anchors, calibration, 28,017
  paired results and `camera-invariance.png`; `v1` remains preserved.
- `policy-refinement-v1/`: component ablation, guarded abstentions and plot.
- `progress-reference-v1/`: clock/phase/meter comparisons and simulated bank.
- `joint-candidate-v1/`: fresh-seed combined candidate and baseline results.
- `camera-demo-v1/camera-normalization-test.mp4`: 23-second single-panel real-pixel
  test with explicitly injected +10% green, frozen masks and current-frame paired
  errors. It demonstrates correction, not a real recipe mismatch. All 115 frames
  fully decode and match source RGB/native PTS.
- `verification/`: independently recomputed counts and artifact hashes.

Twenty Stage 2 contract tests pass. All experiment helpers reject existing output
paths. Frozen executed helper bytes/hashes remain in their experiment directories.
No original source labels, reviewer workspaces, deployed defaults or `main` changed.
Run 5 values were not loaded or used for this development. See
[verification](../verification/stage2-refinement-20261020.json).
