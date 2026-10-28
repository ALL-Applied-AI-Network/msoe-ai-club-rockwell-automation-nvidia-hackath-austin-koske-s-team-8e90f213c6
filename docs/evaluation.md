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

"Consistent with packet" fails a claim the packet rules out: carryover on the first vial of the
run, carryover with no colour comparison (no measurement or no reference), carryover when colour
moved away from the previous recipe, or "no reference" when a reference was used. Both VLMs find
all 5 real carryovers but call almost everything carryover with 0.7–0.95 confidence, and the
images make Cosmos worse, not better. Zero-shot VLM causes are not usable; verdicts and actions stay
numeric, and the VLM's job (if any) is wording plus visual causes after a LoRA on synthetic packets.
Run 5 has no cause labels, so none of this is cause accuracy.

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
