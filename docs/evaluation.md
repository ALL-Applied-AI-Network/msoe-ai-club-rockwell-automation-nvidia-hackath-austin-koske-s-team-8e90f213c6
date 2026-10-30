# Evaluation

The liquid-model results support a development candidate. Reviewed real mask
and first-visible onset accuracy are not yet available.

## Matched recorded-frame comparison

The latest GB10 comparison uses 264 inspected Run4 frames from six correlated
clips, identical source/reference pairs, fresh vial inference, a full warm-up
and two measured passes per arm. Timings include vial detection, liquid
processing and sampling QA. They exclude decoding, transport and display.

| Liquid branch | Median CV | p95 CV | QA-eligible samples |
| --- | ---: | ---: | ---: |
| Direct temporal U-Net | 16.37 ms | 24.86 ms | 105 / 264 |
| Paired DINOv3 with mask/presence heads | 29.43 ms | 31.63 ms | 180 / 264 |

[Recorded comparison and verification](../setup/verification/dinov3-direct-mask-20261024.json).

QA eligibility means that sampling checks passed. It is not segmentation or
recipe accuracy. All 77 paired-DINO gains coincide with U-Net containment
rejection, and 76 already had U-Net proposals. Taller masks can satisfy a box
gate while covering glass. Visual inspection favored the tighter U-Net pools;
this is an engineering selection, not proof of superior real accuracy.

An earlier temporal-prompted SAM2.1 study produced 185 / 264 eligible samples.
Its 72.62 ms figure measures incremental refinement/QA with cached upstream
outputs, not full pipeline latency. Do not compare it directly with the table's
complete CV timing. [Study record](../setup/verification/sam-temporal-comparison-20261022.json).

## Synthetic development

The selected U-Net's synthetic development result is 48 / 48 presence decisions
and 0.8933 mean positive geometric IoU. The 48 images belong to 12 families and
were used for checkpoint selection. They are neither an independent synthetic
test nor real optical ground truth.

Stress tests show sensitivity to reference selection and alignment. A later
reference changed 126 / 264 real proposal decisions. This measures prediction
sensitivity, not an error rate against human labels.
[Training and stress record](../setup/verification/temporal-candidate-checks-20261022.json).

## Reproducibility

The publication check replays two frozen machine examples on CPU. Raw and
component masks match exactly; presence agrees within 1e-4. Sixteen focused
tests cover geometry/sampling, review integrity and onboarding file checks.
[Publication record](../setup/verification/temporal-cnn-release-20261026.json).

These checks validate bytes, implementation behavior and contract handling.
They do not establish independent model accuracy, reference suitability or a
working camera-to-dashboard deployment. Experiment JSON files retain their
original dates, paths and hashes; old local paths are provenance, not run commands.

## Remaining acceptance measurements

| Measurement | Evidence needed |
| --- | --- |
| Pool presence precision/recall and mask IoU/Dice | Independent source-matched human review, including empty, stream-only, faint and uncertain cases |
| First-visible delay | Reviewed absent/present intervals on native video clocks |
| Recipe warning false alarms and misses | Independently accepted recipe references and attributed observed outcomes |
| Reference robustness | Capture/reset/alignment tests on fresh recordings and supervised camera conditions |
| End-to-end latency | Source capture/PTS through inference, QA, transport and visible display |
| Transfer to another camera/container | Separate camera profile and evaluation before claiming reuse of this checkpoint |

Use a fresh evaluation set without tuning the frozen candidate on its results.
Report uncertain/excluded coverage and missed proposals alongside successful
cases. See [the test procedure](operations.md#supervised-test).

## Operator application rehearsal

The CPU Docker path on `austin-work` processes a re-encoded eight-second Run4 visual
excerpt through the released YOLO/U-Net, unchanged QA, FastAPI and React/Nginx.
The browser samples 41 frames: 29 have a model proposal and 14 have an available
sample. These counts describe this integration clip, not segmentation accuracy,
independent test examples, manufacturing defect accuracy or production throughput.
Original Run4 source-frame hashes/PTS are retained separately from the browser
seek clock and re-encoded demo clip. No Run5 data or model fitting is involved.

The browser rehearsal covers original source display, mask artifacts, an attributed
operator response, stale-run response rejection, capture orientation, stopped and
offline states, and absence of invented progress. SQLite events, the response and
an exact rendered artifact survive an API restart. See the compact
[verification record](../setup/verification/demo-integration-20261027.json) and
[laptop instructions](demo.md). Physical camera/reference/recipe qualification and
performance on the target laptop/CSI host remain separate tests.

## GB10 native run

On spark8 (NVIDIA GB10, aarch64, Ubuntu 24.04), the [native kit](../setup/gb10/README.md)
installed torch 2.14.1+cu130 and served the app on `cuda:0` without Docker. The same 40
extracted Run4 excerpt frames were posted sequentially to the API:

| Path | Time per frame | Proposals | QA available |
| --- | ---: | ---: | ---: |
| x86 laptop, CPU Docker | ~300 ms | 28 / 40 | 14 / 40 |
| GB10 native, CUDA, first pass after start | 138–145 ms | 28 / 40 | 14 / 40 |
| GB10 native, CUDA, warm pass | 86 ms | 28 / 40 | 14 / 40 |

Times are client-measured HTTP round trips, including decoding, inference, QA, rendering
and SQLite/artifact persistence. They are not camera-to-display latency. Against the x86 CPU
path, proposal and QA decisions agreed on 40/40 frames; mask areas differed by at most one
pixel and Lab by at most 0.59. Two GB10 passes were identical. This shows device consistency,
not accuracy. The browser rehearsal on the GB10 (headless Chromium on the device) reproduced
41/29/14 and passed the response, artifact, reset, offline and Investigation checks.

The reproducible kit adds a bundled CPython 3.12.15 and hash-locked packages. From a fresh
extract on spark8, `START.sh --selftest` matched the frozen x86 reference outputs on 40/40 frames
with CUDA (max ΔLab 0.59, 90 ms median) and on CPU (ΔLab below 1e-13, 171 ms median). The GB10
CPU path also matched an offline wheelhouse install. A different GB10 should rerun the self-test
before the demo.

The GitHub path was tested from a fresh clone of `austin-work` on the x86 laptop with an NVIDIA
GPU: `setup/gb10/onboard.sh` downloaded and checksum-verified the model release, built the
`compose.gpu.yaml` image, confirmed CUDA inside the container and passed the self-test (40/40,
max ΔLab 0.59, 115 ms median). The aarch64 Docker GPU image has not been built; on the GB10,
onboarding falls back to Docker CPU and then to the validated native launcher.
[Record](../setup/verification/gb10-native-20261027.json).

## Run5 retrospective recipe check

The organizers' Run5 recording labels 30 vials (10 recipes × 3) with fill start/end frames.
The released YOLO11n/U-Net and unchanged QA ran per vial with a fresh tracker; the temporal
reference was the frame at the labelled fill start. The endpoint rule (median Lab of QA-valid
samples in the last 6 s of the fill) was selected on Runs 1–4 only: 22/32 detected fills
covered with 0.49 median within-vial spread, against 18/32 and 4.77 for a 3 s after-fill window.
A first Run5 pass with the after-fill window (15/30) prompted the comparison; the rule was then
fixed on Runs 1–4 and applied to Run5 once.

| Result | Value |
| --- | --- |
| Vials with a QA-valid endpoint | 20 / 30 |
| Recipes with a proxy reference (3rd vial, 2nd as fallback) | 7 / 10 |
| First vials after a change that could be compared | 5 / 10 |
| …above ΔE2000 5 | 5 / 5 (7.8, 24.5, 17.1, 13.1, 6.0) |
| Second vials within ΔE2000 5 | 4 / 6 |
| Vial 4 (recipe 5 → 6) | ΔE 24.5 to recipe 6, 1.6 to recipe 5 |

Run5 is diagnostic-only, not a blind test, and the references are proxies from the same
recording. There are no delivery records or per-vial outcomes, so the strongest claim is
"diverges from the intended recipe, consistent with carryover"; dosing and optical causes are
not excluded. In the operator app, the same clips reproduce these verdicts through browser
decoding (vial 2: ΔE 3.4 in spec; vial 4: 24.8 out of spec, 1.4 to recipe 5; vial 5: 10.1).
Scripts: `replay/run5_endpoints.py`, `replay/export_run5_recipe_check.py`,
`replay/make_run5_vial_clips.py`.

## Line runner on Run 5 (v1)

`replay/line.py` ran the whole Run 5 video with only the recipe schedule: automatic vial
segmentation (carrier still periods, empty-vial reference taken as the carrier settles), the
released models and unchanged QA, then `replay/decide.py`. Stage 2 parameters were set before
scoring and sanity-checked on Run 4; Run 5 was scored once. 50,941 frames took 3 min 18 s on an
RTX 4080 Super. [Record](../setup/verification/line-runner-run5-20261027.json).

| Result | Value |
| --- | --- |
| Vials detected / labelled | 30 / 30, 0 false positives |
| Fill start / end error (median) | 1.2 s / 4.9 s |
| Schedule alignment | 30 / 30 vials assigned the right recipe |
| Endpoint coverage | 24 / 30 |
| Repeatability (median ΔE, non-first vials of a recipe) | 2.6 |
| First vials after a change | 5 OUT OF SPEC, 5 CAN'T VERIFY (recipes 8, 9, 13 have no reference) |
| Second / third vials | 4 / 6 IN SPEC; 3 / 1 OUT OF SPEC; 3 / 3 CAN'T VERIFY |
| Actions | PASS 10, SCRAP 9, FLUSH 5, FLUSH RECOMMENDED 4, HOLD 11, STOP 0 |
| Timeliness | SCRAP 9 / 9 before the 120 s deadline; FLUSH 5 / 5 before the next fill; median decision 8.4 s after fill end |
| Counterfactual | Vial 5 (still carryover) likely saved by the FLUSH issued after vial 4 |

Limits: references are Run 5 proxies (3rd vial, from labelled windows), so second/third-vial
verdicts are partly circular; there is no per-vial outcome truth, so these are descriptive, not
accuracy. Vial 1 starts mid-fill at frame 0, so its reference is not empty. Vial 30's fill window
is truncated at the end of the video.

## Early warning during the fill (Run 5, 2026-10-30)

`replay/early.py` compares each QA-valid sample during a fill with the recipe's reference vial at
the same time since fill start (causal: first of three samples with ≥ 300 liquid px). Three
samples in a row more than the line band (ΔE2000 5) away raise one `EARLY_WARNING` action, only
within the reference's fill duration. Reference trajectories come from CREATE mode or, for Run 5,
from its reference vials (`client/public/data/run5/reference-trajectories.json`; recipe 6's vial 6
has too few QA-valid samples, so recipe 6 has none). Run through the live engine (Review of the
stored samples), reference vials excluded:

| Result | Value |
| --- | --- |
| Out-of-spec vials warned during the fill | 6 / 8 (the other 2 are recipe 6, without a trajectory); 6 / 6 with one |
| Warning lead before the endpoint decision | 16.5, 16.9, 17.3, 19.1, 23.8, 32.6 s (median 18.2 s) |
| Warning lead before the fill ends | 4.0–21.0 s; as early as 2.5 s into the fill (vial 1) |
| In-spec vials warned (false warnings) | 1 / 4 (vial 2, ΔE 2.84 at the end) |

The early warning is not a model. The learned models (vial detector, liquid segmenter) produce the
per-sample colour, and the warning is a fixed comparison with one reference vial's fill
([architecture](architecture.md#learned-perception-deterministic-decisions)).

Not a blind result: the rule was changed twice after looking at Run 5. Comparing with the final
reference colour warned on 3 / 4 in-spec vials (mid-fill colour differs because tanks dispense in
sequence), so the reference trajectory replaced it; then a warning after the reference's fill had
ended (vial 17) was excluded as not early. Band and sample count were not changed. The sample is
small and the references are Run 5 proxies; it needs fresh CSI recordings with CREATE references.
[Per-vial record](../setup/verification/early-warning-run5-20261030.json).

## Run 5 evaluation protocol (from 2026-10-29)

Run 5 is the only labelled run (30 vials, fill windows and recipe identity, no outcome or cause
labels), so recipe references have to come from its own vials, as an operator would capture a
known-good vial per recipe on the line. To keep that from scoring a vial against itself:

1. Settings are frozen before scoring with `python -m replay.eval_freeze` (thresholds current or
   tuned on synthetic train families by `replay.carryover_synthetic`, the recipe library by
   SHA-256, the reference vials, the residue estimator). `replay.line --freeze F` refuses a
   different library and takes the thresholds from the record.
2. Reference vials (3, 6, 9, 18, 20, 24, 30; from each reference's `from` note) are listed and left
   out of the verdict counts. The headline is the first vial after each recipe change.
3. Each first vial is re-scored against every repeat vial of its recipe; `verdict_stable` says
   whether the in/out-of-band verdict depends on which vial supplied the reference.
4. Run 5 is scored once. Carryover detection and false-alarm rates come only from the synthetic
   held-out families; Run 5 cannot measure them.

Plumbing check: re-deriving the v1 run above from its saved samples (current thresholds, no
models, nothing tuned) reproduces segmentation, alignment, actions and timeliness; with reference
vials excluded the second / third vials are 3 / 1 IN SPEC, 3 / 0 OUT OF SPEC, 3 / 3 CAN'T VERIFY,
and all 7 checkable first vials keep their verdict under every alternative reference.

## Synthetic carryover: FLUSH rule and residue estimate (2026-10-29)

14,400 Isaac renders on Brev (v2: residue 0–30%; v3: 0–100%), 4,800 changeovers of three
renders each (intended mix, previous mix, vial), 450 train / 150 dev scene families. Every
render was measurable. Numbers are dev families; no real data or Run 5.
[Record](../setup/verification/carryover-synthetic-20261029.json); method in
`replay/carryover_synthetic.py` and `setup/vision/train_carryover_estimator.py`.

**FLUSH rule** (out of band and toward the previous recipe), detection by residue:

| Thresholds | 0–10% | 10–30% | 30–50% | 50–100% | False alarms (clean / light drift) |
| --- | --- | --- | --- | --- | --- |
| Current: ΔE2000 5, cos 0.8 | 0.5% | 22% | 44% | 75% | 0% / 0% |
| Selected on train with false alarms ≤ 5% under light drift: ΔE 1, cos 0.65 | 41% | 74% | 86% | 93% | 2.4% / 2.7% |

Clean vials sit 0.67 ΔE (median) from their reference; 5–15% lighting drift alone moves them about
2 ΔE, as much as 10–20% residue, so the direction test does most of the work. The synthetic noise
floor is far below real repeat-vial variation, so the selected band is **not frozen**: a band
picked here would flag many real good vials unless synthetic measurement noise is added first.

**Under realistic measurement noise** (`replay/carryover_noise.py`, 2026-10-30). Independent
Gaussian noise (σ per Lab axis) is added to every render's measured colour, with 3 noise seeds, and
thresholds are re-selected on the train families (false alarms ≤ 5%). The real level comes from
Runs 1–4, not Run 5: consecutive vials closer than ΔE 5 (assumed repeats of one recipe; no recipe
labels exist) differ by a median of 1.8 ΔE (0.3–3.6, 9 pairs). σ = 1.0 gives clean synthetic vials
a median of 1.69 ΔE from their reference. Dev families, seed 0:

| σ (clean median ΔE) | Rule | False alarms | 0–5% | 5–10% | 10–20% | 20–30% | 30–50% | 50–100% |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1.0 (1.69) | Current: ΔE 5, cos 0.8 | 0% | 0% | 3% | 13% | 36% | 51% | 75% |
| 1.0 (1.69) | **Selected on train: ΔE 2.0, cos 0.75** | 4.8% | 13% | 30% | 50% | 68% | 67% | 89% |
| 1.5 (2.38) | Selected on train: ΔE 3.0, cos 0.75 | 5.1% | 8% | 19% | 41% | 59% | 58% | 86% |
| 0 (0.67) | Selected on train: ΔE 1.0, cos 0.5 | 4.0% | 26% | 60% | 68% | 84% | 88% | 94% |

At the noise level matched to Runs 1–4, **ΔE 2.0 / cos 0.75** keeps about 5% false alarms and
flags about 4× more 10–20% residue than the current rule. That is the candidate to freeze before a
fresh CSI run; the noise-free selection (ΔE 1.0 / cos 0.5) would over-alarm on real vials. The
noise model is independent per render (no lighting or camera drift), and the Runs 1–4 level rests
on 9 unlabelled pairs. [Record](../setup/verification/carryover-noise-20261030.json).

**Direction in optical density.** Dyes mix linearly in optical density (−ln linear RGB), not in Lab.
For very different recipes a pure mix bends in Lab: the mixing model predicts a Lab cosine of
0.47–0.59 for Run 5 recipe 7 contaminated by recipe 6, and the recoloured footage below measured
0.45–0.53. `toward_previous(..., "density")` (line setting `direction_space`) measures it where the
mix is straight. On the synthetic renders under noise, density alone is worse (σ = 1: 11/38/62% at
5–10/10–20/20–30% residue, as the log amplifies dark-channel noise), but **either space** (the larger
of the two cosines), selected on train, matches Lab alone: ΔE 2.0, cos 0.85, 5.3% false alarms,
9/29/46/68% detection. [Record](../setup/verification/carryover-noise-either-20261030.json).

**Known carryover on real footage** (`replay/recolour_eval.py`). Cosmos Transfer cannot inject a known
fraction: our October 21 trial on the GB10 took 26 min for a 6 s clip and changed liquid hues and
filled empty vials. Instead the real Run 5 liquid (the segmenter's own mask) is shifted in optical
density toward the previous recipe by a known f, and the unchanged models, QA, segmentation and
decision run on the result. Targets: the 6 in-spec vials whose recipe and previous recipe have
references (6, 8, 9, 20, 21, 24). Vials flagged as carryover (out of band and toward the previous
recipe):

| f | Current (ΔE 5, Lab 0.8) | Noise-calibrated (ΔE 2.0, Lab 0.75) | Calibrated, either space (ΔE 2.0, 0.85) | Measurable |
| --- | --- | --- | --- | --- |
| 0 | 0 | 0 | 0 | 6 / 6 |
| 0.1 | 0 | 1 | 2 | 6 / 6 |
| 0.2 | 1 | 2 | 3 | 5 / 6 |
| 0.3 | 1 | 3 | 4 | 4 / 6 |

At f = 0 the harness reproduces the original verdicts (6/6 in spec), and no rule flags carryover on
the untouched vials. Recipes 11 after 10 (vials 20, 21) barely move (the two references are close),
as expected. Heavy recolouring of dark vials can make QA refuse the colour (vials 6 and 9), so
"measurable" falls. Six vials, one run, and the thresholds were chosen on synthetic data before this
test; the recolouring is a physical approximation (no scattering, no change in the fill dynamics), and
the segmenter's mask spills onto glass and background in some frames, which get tinted too (a
visible halo; `screenshots/11-known-carryover-recolour.png`, local).
[Record](../setup/verification/recolour-run5-20261030.json).

**Residue estimate** (mean absolute error in residue fraction):

| Residue | n | Lab projection | CNN, uniform (seeds 29 / 30 / 31) | CNN, balanced (seeds 29 / 30 / 31) |
| --- | --- | --- | --- | --- |
| 0 | 376 | 0.040 | 0.019 / 0.014 / 0.013 | 0.021 / 0.018 / 0.020 |
| 0–10% | 203 | 0.035–0.073 | 0.015–0.023 | 0.012–0.021 |
| 10–30% | 483 | 0.056–0.062 | 0.020–0.037 | 0.021–0.027 |
| 30–50% | 43 | 0.085 | 0.075–0.085 | 0.065–0.069 |
| 50–100% | 95 | **0.052** | 0.080–0.131 | 0.074–0.076 |
| All | 1,200 | 0.053 | 0.036 / 0.026 / 0.029 | 0.027 / 0.026 / 0.028 |

The CNN (9-channel liquid crops of the three renders, about 3–4 minutes per seed on an RTX 4080
SUPER) is 2–3× better up to 30% residue and worse above 50%, where training data is thin (v3
only). Uniform sampling, 60 epochs: best epochs 52–56. Balanced (`--balance`, each epoch draws
train changeovers equally per residue bin; 120 epochs, best 117–119): heavier residue improves and
the seeds agree closely, at a small cost on clean vials, but 50–100% is still worse than the
projection. Epochs are selected on the dev families. It reads renderer liquid
masks, not the line's segmentation, and the app stores only reference colours, not reference
crops, so the line keeps the projection. Above about 50% a vial is mostly the previous recipe and
is reported as a possible schedule or switching problem rather than a carryover percentage.

**Hybrid** (`setup/vision/eval_carryover_hybrid.py`, balanced checkpoints, same dev changeovers):
report the projection when a gate reads at or above a cutoff fixed in advance (0.3 or 0.5),
otherwise the CNN. Gated by the projection's own estimate it is worse than the CNN alone (all
vials 0.039–0.044), because the projection over-reads many light vials (209 switched at 0.3, against
138 true vials above 30%). Gated by the CNN's estimate at 0.5 (86–88 vials switched) it is the
best of the three: all vials 0.025–0.026, 50–100% 0.058–0.061, no change below 50%. A 0.3 cutoff
loses at 30–50% (0.078–0.083). Still synthetic only, with the CNN epoch and the comparison both on
the dev families, so it does not change the line.

**Carryover forecast and recipe order** (synthetic, no real input). `replay/carryover_forecast.py`
forecasts a changeover's first vial from the two references and the true f, with no fitted
parameters. On the 1,195 dev changeovers the optical-density mix is within 0.43 ΔE2000 (median;
90th percentile 1.42) of the measured vial and agrees on out of band (ΔE 5) for 98.2%; the Lab
mix gives 0.48 / 1.64 / 97.9%. The error is about the clean-vial noise floor (0.67). The fixture
mixes dyes the same way, so real dyes may fit worse. `replay/schedule.py` uses that forecast as a
line model (each vial is (1 - f) recipe + f line contents; a water vial costs half a scrap) to choose
flushes and order recipe runs, or to re-plan from the last vial's measured colour.
`replay/schedule_synthetic.py`: 300 random queues (6 runs, 18 vials each), scored with the same
line model, so these are predicted savings:

| f | Requested, no flush | Requested, current rule (flush if references > 15 ΔE apart) | Requested, best flushes | Reordered within 2 | Reordered freely |
| --- | --- | --- | --- | --- | --- |
| 0.1 | 0.33 scraps | 0 scraps, 2.62 flushes | 0 scraps, 0.33 flushes | 0, 0.03 | 0, 0 |
| 0.2 | 1.52 | 0.20, 2.62 | 0.14, 1.38 | 0.03, 0.48 | 0, 0.16 |
| 0.3 | 2.56 | 1.59, 2.62 | 1.31, 1.25 | 0.49, 0.75 | 0.21, 0.58 |

**Lag line model** (`--model lag`, 2026-10-30). CSI's flush clips suggest each vial holds mostly
the previous *dose* rather than a fraction of the mixed line contents. In the lag model a vial is
(1 − f) its dose + f the previous dose, so a water flush dilutes the next vial instead of clearing
it. On the same 300 queues:

| f (lag) | Requested, no flush | Current rule | Best flushes | Reordered within 2 | Reordered freely |
| --- | --- | --- | --- | --- | --- |
| 0.3 | 2.39 scraps | 1.49 + 2.62 flushes | 1.29 + 1.10 | 0.50 + 0.61 | 0.24 + 0.49 |
| 0.6 | 3.53 | 3.53 + 2.62 | 3.53 + 0 | 2.26 + 0 | **2.07 + 0** |
| 0.9 | 3.99 | 3.99 + 2.62 | 3.99 + 0 | 2.94 + 0 | **2.82 + 0** |

Under a strong lag, flushes do not help (the current rule spends 2.62 water vials per queue for no
fewer scraps) and reordering is the only lever (−29 to −41%). Which model fits CSI's line, and its
f, needs one recorded changeover with repeat vials and known contents.

The line's f is unknown and is not fitted to Run 5; whether CSI may reorder runs is a question for
CSI. Neither is in the app yet.

Annotated video: `python -m replay.annotate --video V --run DIR --library L --out annotated.mp4`
redraws the run from its saved outputs (about 2 min for Run 5, no models): the vial region,
fill phase, measured vs reference colour with live ΔE, the verdict, an action log with deadline
countdowns and a strip of all scheduled vials. Verdicts and actions appear at their causal issue
time; fill phase uses the finalized segmentation. Local renders are in the results directory
(`run5-v1/annotated.mp4` and an 8x `annotated-8x.mp4`), not in Git.

RCA explainers: `python -m replay.explain --run DIR --backend rule` or `--backend openai --url
http://HOST:8000/v1 --model M --video V` (any OpenAI-compatible server, e.g. vLLM serving Cosmos
Reason2-2B with the team LoRA; `--video` attaches empty / mid-fill / fill-end crops). Output that is
not one JSON object with a taxonomy cause, cited packet sections and an operator explanation is
recorded as `unknown` with the problem listed. Reported per backend: schema-valid and grounded
counts, cause distribution, agreement with the stage-4 pattern where it determines a cause, latency,
and cause accuracy when `--truth seq,cause` exists (synthetic sessions). Rule baseline on the 20
Run 5 packets: 20/20 valid and grounded, 10/10 agreement (5 carryover, 5 no reference); the other
10 are dose error 4 / unknown 6, which Run 5 cannot check because it has no cause labels.

## RCA explainers on Run 5 (v1)

The 20 non-PASS Run 5 evidence packets went to each explainer unchanged. Cosmos-Reason1-7B and
Qwen2.5-VL-7B ran zero-shot on the GB10 (bf16, greedy) with the same prompt, written before the
run and not changed after. Images are the vial-region crops at the empty, mid-fill and fill-end
frames. [Record](../setup/verification/rca-explainers-run5-20261028.json).

| Explainer | Schema-valid | Cited real evidence | Agrees where stage 4 fixes the cause | Consistent with packet | Median s/vial | Causes |
| --- | --- | --- | --- | --- | --- | --- |
| Rule baseline | 20 | 20 | 10/10 | 20 | 0 | carryover 5, no reference 5, dose error 4, unknown 6 |
| Cosmos-Reason1-7B + images | 18 | 19 | 5/10 | 6 | 17.0 | carryover 19, unknown 1 |
| Cosmos-Reason1-7B, numbers only | 16 | 18 | 7/10 | 9 | 16.4 | carryover 15, no reference 3, unknown 2 |
| Qwen2.5-VL-7B + images | 20 | 20 | 5/10 | 5 | 10.6 | carryover 20 |
| Jev 1.14 (TypeSafe API) + images | n/a¹ | 20 | 10/10 | 17 | 0.24 | carryover 8, occlusion 6, no reference 5, dose error 1 |
| Jev 1.14 (TypeSafe API), numbers only | n/a¹ | 20 | 10/10 | 17 | 0.19 | carryover 8, no reference 5, short fill 3, occlusion 3, dose error 1 |
| OpenAI Decisions API (gpt-6-luna) + images | n/a¹ | 20 | 10/10 | 20 | 0.34 | carryover 5, no reference 5, unknown 4, dose error 3, bubbles 2, occlusion 1 |
| OpenAI Decisions API (gpt-6-luna), numbers only | n/a¹ | 20 | 10/10 | 20 | 0.26 | unknown 7, carryover 5, no reference 5, occlusion 3 |

"Consistent with packet" fails a claim the packet rules out: carryover on the first vial of the
run, carryover with no colour comparison (no measurement or no reference), carryover when colour
moved away from the previous recipe, or "no reference" when a reference was used. Both VLMs find
all 5 real carryovers but call almost everything carryover with 0.7–0.95 confidence, and the
images make Cosmos worse, not better. Zero-shot VLM causes are not usable; verdicts and actions stay
numeric, and the VLM's job (if any) is wording plus visual causes after a LoRA on synthetic packets.
Run 5 has no cause labels, so none of this is cause accuracy.

**Hosted on build.nvidia.com** (2026-10-30, same 20 packets, prompt and scoring; zero-shot):

| Explainer | Schema-valid | Agrees where stage 4 fixes the cause | Consistent with packet | Median s/vial |
| --- | --- | --- | --- | --- |
| Nemotron 3 Ultra 550B, numbers only | 20 | 10/10 | 18 | 21.8 |
| Nemotron 3.5 Super VL 120B + images | 18 | 10/10 | 14 | 20.5 |
| Llama 3.2 90B Vision + fill-end image (one image only) | 18 | 9/10 | 12 | 25.3 |
| Nemotron 3 Nano Omni 30B + images | 18 | 5/10 | 8 | 9.3 |

Cosmos-Reason2-8B is listed but returned "function not found for this account" (not enabled for
the key). The large text model is close to the rules; the smaller vision models still over-call
carryover. Records: `rca-nim-*` folders beside the Run 5 line-runner results.

¹ Jev answers a `choice` question over the same 11 causes (with a description per cause) and
returns the cause with per-cause probabilities, not prose, so it has no explanation or next check
to validate. It was run zero-shot with the same packets and system text, from a hosted API (vial
crops padded to its 280 px minimum). It matches the rules on every determined vial (carryover at
0.98–1.00, no reference at 0.87–0.99) and puts unmeasured vials on occlusion or short fill at low
confidence (0.3–0.75) instead of carryover. Its three contradictions are vials 16, 29 and 30
(deviation with no colour comparison) called carryover at 0.23–0.47. The OpenAI Decisions API (same
packets, system text and cause descriptions, `--backend decisions`) also matches the rules on every
determined vial at 0.89–1.00 and has no contradictions. With images it picks dose error on the three
deviation vials the rules call dose error (0.45–0.64) and suggests bubbles on two unmeasured vials;
numbers only, it answers unknown wherever the packet does not settle the cause. Both decision APIs
are far better cause-pickers than the zero-shot VLMs, but they need the network; the rule baseline
still decides.

## Finished-vial tray on Run 5 (v1)

`replay/tray.py` reads the Run 5 cap_seal video (same clock as the fill camera; burned-in clocks
agree within ~20 ms). Good-tray slots are occupied when their patch differs from the empty tray at
frame 0 for 5 s with no arm over the trays; a slot without yellow is a missing or crooked cap. The
Bad tray is counted by yellow caps only. Run 4 with unchanged settings: 3 arrivals matched (transit
118–148 s), the remaining vials in transit when its 5-minute video ends. Arrivals are aligned in order to the line runner's vials,
allowing a vial that never arrives. Slot grid and thresholds were set by looking at Run 5
(start/end frames and the 1215–1290 s window), so this is a demonstration, not a blind test.

| Result | Value |
| --- | --- |
| Vials that left the fill station | 30 |
| Good-tray arrivals / matched | 26 / 26 (Bad tray: 0) |
| Still in transit at video end | vials 27–30; none missing |
| Transit, fill-station departure → tray | median 128 s, range 100–166 s (CSI: 2–3 min) |
| Fill order | column 6 → 1, top to bottom, matching vial order |
| Cap not detected on the slot | vials 17 (R2C3, cap visible beside the slot: likely crooked) and 20 (R5C3, open vial) — both colour PASS |
| Reached Good although we decided SCRAP / HOLD | 7 (1, 4, 5, 7, 16, 19, 23) / 9 |

The yellow-cap count alone gives 25 Good-tray caps and misses vial 20; the slot view recovers it.
Run 5 had no sorting, so "reached Good" is what the line did, not an error by the line. Decisions are
issued at fill-station departure, so every SCRAP had at least 100 s before its vial reached the tray.
`python -m replay.annotate ... --tray-video cap_seal_video_5.mp4 --tray DIR` adds the tray view.

## Sequence-diagnosis pilot

The investigation module was exercised on 1,620 analytic four-vial sequences across
ten scenario families (162 each), frozen before scoring. No fitting, rendered
Isaac images, real-data threshold selection or Run5 data was involved.

| Scenario | Observed software behavior |
| --- | --- |
| Nominal | 162/162 inside the experimental color band |
| Persistent wrong recipe | The legacy first-vial rule suggests carryover in 162/162; sequence logic reports persistent deviation with cause undetermined |
| Unhealthy camera / unqualified recipe reference | 162/162 of each withhold diagnosis |
| Attributed delivery discrepancy | All 162 remain visible, including 54 inside the color band |
| Carryover and constructed dosing/optical alternatives | 162 identical-input triplets receive identical diagnoses;76/162 in each family receive a carryover hypothesis, with alternatives retained |

These are feature-model scenario counts. Analytic counterexamples cover 54 process
settings and 216 endpoints where alternative mechanisms share the same ideal color.
They do not establish physical feasibility of the optical construction or real
causal accuracy. The audited Run4 probe has 264 frames/six correlated clips and 105
QA-eligible samples, but supplies zero qualified completed-vial RCA inputs.

The [Investigation panel](investigation.md) exposes this uncertainty, source
bindings and missing evidence. Its examples stay synthetic and separate from run
records. Cause confirmation, product quality and hardware commands remain unset.

## CSI live-camera transfer check (2026-10-29)

The completed live-camera recording `live-20261029-230248/fill.mp4` is transfer
evidence only; it was not used to train or fit either model. On all 944 saved
model samples, the released vial YOLO returned zero boxes. It also returned no
box on three representative vial frames with confidence reduced to 0.001, so a
confidence adjustment does not recover this camera domain.

The station profile now supplies candidate vial geometry `[278, 75, 358, 420]`
in native 640x480 coordinates only when YOLO returns no box. The UI and saved
samples identify this as `station_profile_candidate`; it is not presented as a
model detection. On a source-matched transfer replay using the 45 s empty-vial
reference, all 61 half-second samples from 50--80 s received profile geometry,
59 produced a liquid proposal, and 59 passed measurement QA. The replay output
and four visual overlays are under
`results/station-onboarding/csi-live-20261029/transfer-01/`.

This is an integration check, not detection accuracy: the recording has no
source-matched human vial boxes or liquid masks. Operator review of the profile
box and human transfer labels remain required before approving it as a station
profile. The MP4 uses the recorder's generated constant-frame time grid; native
USB capture timestamps are unavailable.

### Vial detector retrained on CSI-view renders (candidate, 2026-10-30)

The released YOLO11n was trained on renders framed like Run 5 and found no vial on CSI's current
camera. The Brev carryover renders use the station's own FillCam, whose fill-position vial box
([279, 66, 341, 420]) almost matches the CSI station region ([278, 75, 358, 420]). The detector
was fine-tuned from the released checkpoint on 4,000 of those renders (train families; 800 dev
renders for validation; fill-position vial and liquid only, as in the release), 12 epochs, with
the fixed final epoch. No real image was used for training or selection.

| Frames (every 15th; Run 5 every 60th) | Released | Retrained | In the station region |
| --- | --- | --- | --- |
| CSI live 230248 (956) | 0 | 91 | 91 |
| CSI live 231953 (2,021) | 7 | 387 | 387 |
| CSI live 233940 (1,043) | 0 | 368 | 368 |
| CSI live 000510 (1,312) | 15 | 971 | 971 |
| Run 5 (850) | 817 | 831 | 827; median IoU with the released boxes 0.87 |

Every retrained detection on CSI's camera lies on the station vial position, and Run 5 detection
is unchanged. The CSI frames have no human vial labels and include frames with no vial under the
nozzle, so these are detection counts, not recall or precision. The candidate is not shipped:
the runtime pins the released checkpoint, and adopting it needs a new release identity and a
labelled CSI check. [Record](../setup/verification/vial-csi-view-20261030.json).

### Dual finished-sort profile candidate

The current tray-camera view contains two physically separate 5×6 grids. The previous live path
used slot occupancy only for the Good grid and cap blobs for the Bad region, so an uncapped Bad
vial had no occupancy observation. The station profile now contains 30 independently measured
slot centers for each grid. A solid UI box means the corresponding slot changed from its clear
session baseline for five arm-free seconds; thin outer rectangles are labeled profile regions,
not detections.

On the real evaluation-only recording `live-20261029-230248/tray.mp4`, the candidate established
a baseline, produced 1,434 samples, held 12 arm-occluded samples, and localized the visible capped
Good-tray placement at 468.3 s to R3C6. No Bad-tray placement occurs in the available recordings,
so Bad positive accuracy is unavailable. Synthetic unit observations exercise capped and uncapped
arrivals in arbitrary slots on both grids, but they are not physical accuracy evidence. The
review package is under `results/station-onboarding/csi-live-20261029/tray-dual-grid-05/`. A second
recording with the robot remaining over the grids never qualifies a baseline and correctly emits
no slot observations instead of turning the later clear holes into false vial boxes.

The inventory-baseline update was also replayed on the live MJPEG frame that previously remained
in `WAITING FOR CLEAR VIEW`. It established the robot-free baseline, reserved 8 Good and 6 Bad
starting slots and emitted zero arrival events. This is a profile-behavior check without human
slot labels, so occupancy accuracy remains unavailable. Evidence is under
`results/station-onboarding/csi-live-20261029/tray-inventory-01/`.

The subsequent open-slot update removes that profile's global-clear dependency entirely. Each
slot is classified against its empty-hole signature; changed slots initialize as inventory and
open slots initialize as available. Five seconds of stable per-slot evidence are still required
for later arrival/vacancy events, so ordinary short gripper visits do not become vial arrivals.
The live 2026-10-30 transfer exposed a further camera auto-exposure failure mode: a common grey
shift made unchanged empty holes cross the occupancy threshold and appear as uncapped arrivals.
The profile comparison now subtracts the median signed shift across each 30-slot grid. Replay of
the two affected sessions removed the spurious Good-grid arrivals while retaining localized slot
changes; this is a targeted transfer check, not a labelled physical-accuracy result.
The operator then identified Good R1C6 and R2C6 as empty in a live frame; their previously bright
empty signatures were corrected from that attributed evidence. Replay of the same frame changed
Good from two boxes to zero while retaining the two visible Bad inventory boxes. This is a narrow
profile correction, not a full human-labeled accuracy evaluation.

### LINE sessions on the CSI GB10 (2026-10-29)

Ten sessions were run on CSI's GB10 with the station's fill and tray cameras and saved by the app
(`Linewatch-OMEGA-20261030` handoff, checksums verified; two are simulated replays). Replayed
offline through the same engine: 5,859 model samples, 746 with a QA-valid colour, 14 vials
segmented and decided, 4 tray arrivals. The sessions followed a demo order of Run 5 recipe ids
(7, 11, 5) against Run 5 references while the line ran other recipes, and the physical contents
were not logged, so the verdicts (7 OUT OF SPEC, 7 CAN'T VERIFY) show the path works on
the hardware, not accuracy. The kit's self-test passed 40/40 twice on the same GB10.

## CSI capture: water flushes after red and yellow (2026-10-28 clips, measured 2026-10-30)

Six fill-camera clips recorded by CSI with mL metadata (`shamar-capture/` in the
`Linewatch-OMEGA-20261030` handoff, checksums verified): a water flush, red 40 mL, a water
flush after red, yellow 40 mL, a water flush after yellow, blue 40 mL. Evaluation only. Colour is
the median of a fixed liquid patch (x 296–336, y 330–385) at 90% of each clip, with optical
density taken against the same clip's empty-vial frame.

| Clip label | Looks | Lab | Density R / G / B |
| --- | --- | --- | --- |
| Water flush (first) | brown | 22.7, 8.3, 20.3 | 0.35 / 1.20 / 2.12 |
| Red 40 mL | brown | 26.6, 2.2, 15.1 | 0.38 / 0.78 / 1.33 |
| Water flush after red | strong red | 18.3, 24.6, 24.5 | 0.15 / 2.23 / 3.58 |
| Yellow 40 mL | orange | 22.9, 10.4, 25.3 | 0.15 / 1.07 / 2.26 |
| Water flush after yellow | amber | 23.4, 6.9, 30.2 | 0.37 / 1.20 / 3.76 |
| Blue 40 mL | green | 23.2, −8.4, 20.8 | 1.08 / 0.93 / 2.34 |

Both water flushes hold **more** dye than the vial labelled with the dye before them (2.7–2.9× in
green/blue after red, 1.1–2.4× after yellow). A residue fraction f ≤ 1 cannot produce that, so the
proportional carryover model (`replay/schedule.py`) does not fit these clips and no f was set from
them. They look like a one-vial lag: each vial holds mostly the previous dose (the line holding
about one dose), or the metadata is one row off. Run 5's first vials after a change point the same
way (vial 4: ΔE 1.6 to the previous recipe). Until CSI confirms which, expect the first vial after
every change to read as the previous recipe, and do not assume one water vial clears the line
(the yellow vial after the red flush is orange). One patch, six clips, no repeats: indicative only.
