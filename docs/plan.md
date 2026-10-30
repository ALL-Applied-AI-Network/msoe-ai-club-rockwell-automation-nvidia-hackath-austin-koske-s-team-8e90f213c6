# Linewatch plan: one pipeline to submission

October 27, 2026 · submission November 1, 11:59 p.m. CT · finals November 5

## The product

**Video in, recipe schedule in → a decision for every vial, the action to take, when to take
it, and why.** Linewatch is a recipe-aware replacement for the line's Teledyne colour check.
Today that check compares each vial against a static colour database and the sorter is not
recipe-aware, so a vial in the wrong recipe passes as long as its colour is a "known" colour.

```text
INPUTS                        PIPELINE                                     OUTPUTS
video (file or camera) ─┐
                        ├─► 1 Perceive ─► 2 Segment vials ─► 3 Measure ─┐
recipe schedule ────────┤                                               ├─► 4 Decide ─► 5 Explain (RCA)
recipe library ─────────┘                                               │
                         (references or expected colour per recipe) ────┘

                        ─► vial ledger · action log · RCA reports · annotated video · metrics
```

Constraint from CSI (Derek Riley): we cannot talk to sensors, the PLC or the sorter. Every
action is **issued to the operator** (and logged as the command a PLC integration would send).
The Teledyne database is static; our recipe library is ours.

**Success bar:** the full pipeline works on synthetic line video, where every vial, fault and
cause is known, and works well enough on the organizers' real line video for a convincing demo. We train on
simulation only, evaluate against Run 5 as real ground truth, and measure the sim-to-real gap directly.

## Inputs

| Input | Format | Notes |
| --- | --- | --- |
| Video | MP4/MKV file or live 640×480 UVC camera (same fill-station view) | Native frame times retained |
| Recipe schedule | CSV `seq,recipe_id[,dye amounts]`: the vials that will be made, in order | Stands in for the line's production order; the n-th detected vial is schedule row n |
| Recipe library | JSON: per recipe, dye amounts and reference Lab (or "none") | One library per dye set (organizer clear/orange/blue) |
| Line settings | JSON: SCRAP deadline, flush cost, persistence count | Defaults from CSI: vial reaches the sorter 2–3 min after fill end (deadline 120 s); a flush is one water vial (one fill cycle) |

## Pipeline stages

| # | Stage | What it does | Status |
| --- | --- | --- | --- |
| 1 | Perceive | YOLO11n vial geometry, paired temporal U-Net liquid mask/presence, unchanged measurement QA → per-frame Lab or "unavailable" with reasons | **Done** (released models, GB10 validated) |
| 2 | Segment vials | Detect vial arrival → fill start → fill end → departure from the video, with no manual FINISH VIAL. Empty-vial frame becomes that vial's temporal reference | **Done v1** (`replay/line.py`): Run 5 30/30 vials, 0 false positives |
| 3 | Measure | Endpoint = median Lab of QA-valid samples in the last 6 s of the fill (rule fixed on Runs 1–4). Also keep the in-fill Lab trajectory | **Done** (`replay/run5_endpoints.py`, `client/src/services/recipeCheck.ts`) |
| 4 | Decide | Deterministic rules turn measurements + schedule + library into a verdict, an action and a deadline (below) | **Done v1** (`replay/decide.py`): PASS/SCRAP/FLUSH/STOP/HOLD + proactive flush, with deadlines |
| 5 | Explain (RCA) | Evidence packet → cause from a fixed taxonomy, confidence, cited evidence, next check | **Partial**: evidence packets per non-PASS vial (`rca/vial-NN.json`, with QA flags and 3 keyframes); `replay/explain.py` runs the rule baseline or any OpenAI-compatible endpoint (vLLM + Cosmos, text LLM) with keyframe crops, validates the schema and scores. Run 5 on the GB10: zero-shot Cosmos-Reason1-7B and Qwen2.5-VL-7B call ~everything carryover (consistent with the packet on 5–9/20 vs rule 20/20); Jev 1.14 via the TypeSafe API (`--backend jev`, cause as a choice) agrees 10/10 and is consistent on 17/20 at 0.2 s/vial but needs the network; the OpenAI Decisions API (gpt-6-luna, `--backend decisions`) agrees 10/10 and is consistent on 20/20 at 0.3 s/vial; LoRA on synthetic packets next |
| 6 | Track to tray | Cap_seal camera on the same clock: per-slot arrivals in the Good tray (and Bad-tray caps), matched in order to the vials that left the fill station; flags vials we would divert that reached Good, and missing/crooked caps | **Done v1** (`replay/tray.py`): Run 5 26/26 arrivals matched, 4 in transit at end, 0 missing; transit 100–166 s |

Stages 1–4 are deterministic and numeric. The VLM/LLM explains; it never changes a verdict or action.

### Stage 4: actions and when to take them

| Action | Trigger | When it must be issued |
| --- | --- | --- |
| **PASS** | ΔE2000 to the intended recipe ≤ band (5) | At vial decision time |
| **SCRAP** | ΔE > band | Before the vial reaches the sorter: fill end + 120 s (CSI: 2–3 min) |
| **FLUSH LINE** (order 1 water vial) | Scrapped vial whose colour shifted toward the previous recipe (direction cosine ≥ 0.8) → carryover. Also *proactive*: at a scheduled changeover whose previous→next recipe distance is large | Before the next vial's fill starts (reactive) or before the changeover (proactive). Costs one fill cycle |
| **STOP & CHECK DOSING** | ≥ 3 consecutive out-of-spec vials **not** toward the previous recipe → persistent deviation (wrong recipe loaded, pump/dose fault) | Immediately; hold subsequent vials |
| **HOLD / RECHECK** | No QA-valid endpoint, camera unhealthy, or no reference for this recipe | At vial decision time; operator inspects |
| *(stretch)* **EARLY WARNING** | In-fill trajectory already outside the band | During the fill, before fill end; measured as lead time |

Thresholds are research settings, frozen before scoring on held-out data. CSI has no colour
baseline or per-recipe tolerance today, only upper thresholds, so **per-recipe baselines and
tolerances are a deliverable**: we propose them from reference vials and report the evidence.

### Expected colour per recipe

1. **Reference vials** (primary): the first qualified vials of each recipe in a commissioning pass.
2. **Mixing model** (stretch): predict Lab for any dye mix from pure-dye baselines (absorbance-space
   mixing). This removes the reference requirement for new/custom recipes. Fit and test it on synthetic
   sessions; check it on Runs 1–4 recipes.

### Stage 5: RCA design

- **Evidence packet** (JSON, built by stages 1–4): schedule context (previous/intended/next recipe),
  vial Lab endpoint and trajectory, ΔE to intended and previous, direction cosine, neighbouring vials,
  QA flags, and 3 keyframes (empty, mid-fill, end).
- **Cause taxonomy** (the synthetic fault labels): carryover, wrong recipe, dose error,
  missed dose, short fill, occlusion, vial position, lighting, bubbles, unknown.
- **Models**, compared on the same packets:
  1. rule baseline (`replay/diagnostics.py`);
  2. text LLM over the packet's numbers;
  3. Cosmos Reason2-2B (+ team LoRA) over keyframes + packet.

  Expectation from the team's Cosmos trial: VLMs read colour poorly, so colour causes come from the
  numbers and the VLM adds visual causes (short fill, occlusion, bubbles, position).
- **Output schema**: `{cause, confidence, evidence_ids, explanation, next_check}`, validated; anything
  else is rejected and logged as "unknown".

## Outputs

| Output | Contents |
| --- | --- |
| Vial ledger (CSV + JSON) | Per vial: seq, intended recipe, fill start/end times, endpoint Lab, ΔE intended/previous, verdict, action, issue time vs deadline, RCA cause |
| Action log | Timestamped actions as an operator would receive them (and as PLC commands, simulated) |
| RCA reports | One per non-PASS vial, with evidence and keyframes |
| Annotated video | Overlays, live verdict and actions; the demo artifact (`replay/annotate.py`, done v1) |
| Metrics report | The tables below, per dataset |

One entry point runs the same pipeline offline and in the app:
`python -m replay.line --video X --schedule S --library L --assets A --out DIR [--labels GT]`
(built; `--samples frames.jsonl` re-decides without the models). The app will read the same ledger.

## Training: simulation only, Run 5 as ground truth

| Step | What | Existing starting point |
| --- | --- | --- |
| 1. Synthetic line video | Render whole sessions from a recipe schedule with injected faults (carryover, wrong recipe, dose error, missed dose, short fill) and nuisances (lighting, white balance, occlusion, bubbles, position). Every frame carries vial ID, fill phase, mask, true recipe and fault cause | Isaac Sim 6.0 FillCam renderer and liquid geometry (`setup/isaac/`); digital twin simulator |
| 2. Domain randomization | Randomize dye colours (organizer clear/orange/blue plus other hues), concentration, lighting, camera gain/white balance, noise, JPEG/H.264 compression, small camera shifts | Renderer already varies colour, opacity, fill height, illumination |
| 3. Train on synthetic | YOLO vial geometry, U-Net liquid mask/presence, vial-event segmentation (stage 2), RCA LoRA on synthetic evidence packets | Synthetic-only U-Net is the released model |
| 4. Check transfer on real development runs | Run Runs 1–4 through the pipeline to find sim-to-real failures, then fix them with more randomization or renderer changes, not by training on real frames | Matched Run 4 comparison, Run 1–4 endpoint study |
| 5. Measure the gap | Same metrics on synthetic test and on Run 5 | Run 5 retrospective (`replay/run5_endpoints.py`) |

Ground truth: Run 5's organizer labels give recipe and fill start/end frames for 30 vials (10 recipes ×
3), which define the schedule, vial segmentation truth and recipe changeovers. There is no per-vial
fault label; real faults are the first vials after a changeover, judged against proxy references.
Masks on Run 5 frames can be spot-checked with `annotation_review/`. The CSI 10/27 visit produced no
usable data.
Optional: Cosmos Transfer for photorealism. Our trial showed it invents liquid and shifts colours,
so use it only with mask/edge control, and re-check labels.

## Data splits

Whole sessions/batches stay in one split; never split frames of one vial across train and test.

| Set | Train | Validation (select checkpoints/thresholds) | Test (scored once, after freeze) |
| --- | --- | --- | --- |
| Synthetic | Most rendered sessions | Held-out sessions | Held-out sessions with unseen fault/nuisance combinations |
| Organizer Runs 1–4 | — | Real development: transfer checks, rule sanity | — |
| Organizer Run 5 | — | — | **Ground truth** (30 labelled vials, 10 recipes; not blind, inspected in earlier diagnostics) |

Run 5 references stay proxies (a later vial of the same recipe), as in the existing retrospective.

## Metrics

Targets are proposed goals, not results. Report every metric with its count (n), on **synthetic
test** and **Run 5**; their difference is the sim-to-real gap.

| Area | Metric | Target: synthetic | Target: real |
| --- | --- | --- | --- |
| Perception | Mask IoU (synthetic exact; real on spot-checked frames) | ≥ 0.85 | ≥ 0.6 |
| | QA-valid endpoint coverage (vials with an endpoint / all vials) | ≥ 95% | ≥ 70% |
| | Repeatability: median ΔE between nominal vials of the same recipe | ≤ 1 | ≤ 3 |
| Vial segmentation | Vial count errors; fill start/end timing error | 0; ≤ 0.5 s | ≤ 1 per session; ≤ 1.5 s |
| Decision | Out-of-spec recall / precision | ≥ 95% / 95% | ≥ 80% / 80% |
| | False SCRAP rate on nominal vials | ≤ 2% | ≤ 10% |
| | Action accuracy (PASS/SCRAP/FLUSH/STOP/HOLD vs labelled fault) | ≥ 90% | ≥ 70% |
| | Minimum detectable dose error (mL) | Report | — (no real dose ground truth) |
| | Decisions before deadline; early-warning lead time | 100%; report s | 100%; report s |
| RCA | Top-1 cause accuracy; rule vs LLM vs Cosmos | Beat rule baseline | Report |
| | Evidence faithfulness; invalid-output rate | 100%; ≤ 2% | 100%; ≤ 5% |
| System | GB10 frames/s; decision latency after fill end | ≥ 10; ≤ 2 s | same |
| Business | Faulty vials a recipe-unaware check would pass that we flag | Report | Report |

## Schedule

| Day | Deliverable |
| --- | --- |
| Tue 10/27 | Plan agreed; synthetic session spec (schedule + faults + nuisances) agreed; Run 5 schedule file written from organizer labels |
| Wed 10/28 | First synthetic sessions rendered; stage 2 vial segmentation; offline runner writes the ledger; evidence-packet schema |
| Thu 10/29 | Train on synthetic; transfer checks on Runs 1–4; stage 4 actions + deadlines; metrics harness on all splits; RCA comparison |
| Fri 10/30 | **Freeze models and thresholds.** Score synthetic test + real test once; GB10 run; app shows ledger and RCA |
| Sat 10/31 | Demo video (synthetic session with every fault caught, then real Run 5), metrics tables, write-up |
| Sun 11/1 | Submit |

## Workstreams (proposed owners)

| Stream | Scope | Proposed |
| --- | --- | --- |
| Synthetic sessions | Fault/nuisance session renderer, domain randomization, labels | Cameron, Hayden |
| Training + sim-to-real | Synthetic training, transfer checks on Runs 1–4, gap measurements | Srijan, Aleyna, Reagan |
| Pipeline + integration | Stages 2–4, `replay.line` runner, ledger, metrics harness, GB10 | Austin |
| App | Upload video + schedule, live ledger, actions, RCA panel | Josiah |
| RCA | Evidence packets, synthetic RCA training data, LLM + Cosmos comparison | Zoe, Jev |

## Demo

1. A synthetic session from a schedule with injected faults: every vial gets the right action at the
   right time, with the RCA explaining each fault. This shows the system is correct where truth is known.
2. Real Run 5 video with its recipe schedule: the same pipeline on the GB10 flags the first vials
   after recipe changes (e.g. vial 4: ΔE 24.5 to its recipe, 1.6 to the previous one) as
   OUT OF SPEC → SCRAP + FLUSH, with the RCA naming carryover.
3. The sim-to-real table: the same metrics on synthetic test and on Run 5.

## Not in scope before submission

SAM/Faster R-CNN comparators, Dataset 1, TensorRT, the live digital-twin UI. Code stays in the
repo; see [branches (archived)](archive/branches-20261029.md).

## From CSI (October 27)

- No colour baseline or per-recipe tolerance exists; the line has upper thresholds only.
- A vial reaches the sorter 2–3 minutes after its fill ends (SCRAP deadline).
- A flush is ordering one vial of water: it takes one fill cycle.

**Requested reference recording** (emailed to Shamar Webster, due Thursday 10/29): 10 red/yellow/blue
recipes of 40 mL, each with a water-vial flush before and after, recorded continuously on the vial
camera; optionally 2–3 deliberately unflushed changeovers (blue → yellow ×2, red → green). If it
arrives, it is evaluation/reference data only: independent per-recipe baselines and proposed tolerances,
water-vial clear baselines, and real carryover examples with a known cause. It is not training data.
