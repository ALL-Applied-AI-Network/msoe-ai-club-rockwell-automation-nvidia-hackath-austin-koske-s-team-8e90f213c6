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

## Inputs

| Input | Format | Notes |
| --- | --- | --- |
| Video | MP4/MKV file or live 640×480 UVC camera (same fill-station view) | Native frame times retained |
| Recipe schedule | CSV `seq,recipe_id[,dye amounts]`: the vials that will be made, in order | Stands in for the line's production order; the n-th detected vial is schedule row n |
| Recipe library | JSON: per recipe, dye amounts and reference Lab (or "none") | One library per dye set (organizer clear/orange/blue; CSI 10/27 red/yellow/blue) |

## Pipeline stages

| # | Stage | What it does | Status |
| --- | --- | --- | --- |
| 1 | Perceive | YOLO11n vial geometry, paired temporal U-Net liquid mask/presence, unchanged measurement QA → per-frame Lab or "unavailable" with reasons | **Done** (released models, GB10 validated) |
| 2 | Segment vials | Detect vial arrival → fill start → fill end → departure from the video, with no manual FINISH VIAL. Empty-vial frame becomes that vial's temporal reference | **Gap**: today the operator marks vials or we use labelled frames |
| 3 | Measure | Endpoint = median Lab of QA-valid samples in the last 6 s of the fill (rule fixed on Runs 1–4). Also keep the in-fill Lab trajectory | **Done** (`replay/run5_endpoints.py`, `client/src/services/recipeCheck.ts`) |
| 4 | Decide | Deterministic rules turn measurements + schedule + library into a verdict, an action and a deadline (below) | **Partial**: per-vial verdict exists; flush/stop logic and timing do not |
| 5 | Explain (RCA) | Evidence packet → cause from a fixed taxonomy, confidence, cited evidence, next check | **Partial**: rule-based `replay/diagnostics.py`; Cosmos Reason2 LoRA on `zoe_jev_cosmos` not yet reliable |

Stages 1–4 are deterministic and numeric. The VLM/LLM explains; it never changes a verdict or action.

### Stage 4: actions and when to take them

| Action | Trigger | When it must be issued |
| --- | --- | --- |
| **PASS** | ΔE2000 to the intended recipe ≤ band (5) | At vial decision time |
| **SCRAP** | ΔE > band | Before the vial reaches the sorter (deadline = fill end + transfer time; ask CSI for transfer time) |
| **FLUSH LINE** | Scrapped vial whose colour shifted toward the previous recipe (direction cosine ≥ 0.8) → carryover. Also *proactive*: at a scheduled changeover whose previous→next recipe distance is large | Before the next vial's fill starts (reactive) or before the changeover (proactive) |
| **STOP & CHECK DOSING** | ≥ 3 consecutive out-of-spec vials **not** toward the previous recipe → persistent deviation (wrong recipe loaded, pump/dose fault) | Immediately; hold subsequent vials |
| **HOLD / RECHECK** | No QA-valid endpoint, camera unhealthy, or no reference for this recipe | At vial decision time; operator inspects |
| *(stretch)* **EARLY WARNING** | In-fill trajectory already outside the band | During the fill, before fill end; measured as lead time |

Thresholds are research settings, frozen before scoring on held-out data, not product
tolerances. Ask CSI for their tolerance and compare.

### Expected colour per recipe

1. **Reference vials** (primary): the first qualified vials of each recipe in a commissioning pass.
2. **Mixing model** (stretch): predict Lab for any dye mix from pure-dye baselines (absorbance-space
   mixing). This removes the reference requirement for new/custom recipes. CSI 10/27 Tier 1 data
   (pure R/Y/B plus mixes) is designed to test it.

### Stage 5: RCA design

- **Evidence packet** (JSON, built by stages 1–4): schedule context (previous/intended/next recipe),
  vial Lab endpoint and trajectory, ΔE to intended and previous, direction cosine, neighbouring vials,
  QA flags, and 3 keyframes (empty, mid-fill, end).
- **Cause taxonomy** (matches the CSI session fault labels): carryover, wrong recipe, dose error,
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
| Annotated video | Overlays, live verdict and actions; the demo artifact |
| Metrics report | The tables below, per dataset |

One entry point runs the same pipeline offline and in the app:
`python -m replay.line --video X --schedule S --library L --out DIR` (to be built), and the app
reads the same ledger.

## Metrics

Targets are proposed goals, not results. Report every metric with its count (n).

| Area | Metric | Proposed target |
| --- | --- | --- |
| Perception | QA-valid endpoint coverage (vials with an endpoint / all vials) | ≥ 80% |
| | Repeatability: median ΔE between in-spec vials of the same recipe | ≤ 2 |
| Vial segmentation | Vial count error per session; fill start/end timing error (s) | 0 miscounts; ≤ 1 s median |
| Decision | Out-of-spec recall and precision vs ground truth | ≥ 90% recall, ≥ 90% precision |
| | False SCRAP rate on nominal vials | ≤ 5% |
| | Action accuracy (correct PASS/SCRAP/FLUSH/STOP vs labelled fault) | ≥ 85% |
| | Minimum detectable dose error (mL), from the dose-error ladder | Report value |
| | Timeliness: decisions before deadline; early-warning lead time | 100%; report seconds |
| RCA | Top-1 cause accuracy vs labelled fault type; rule vs LLM vs Cosmos | Beat rule baseline |
| | Evidence faithfulness (cited numbers/vials correct); invalid-output rate | 100%; ≤ 5% |
| System | GB10 frames/s and end-to-end decision latency | ≥ 10 frames/s; ≤ 2 s after fill end |
| Business | Faulty vials the recipe-unaware check would pass that we flag | Report count |

## Evaluation data

| Set | Use | Ground truth |
| --- | --- | --- |
| Runs 1–4 (organizer) | Development; rule selection | Machine labels, fill windows |
| Run 5 (organizer) | Held-out diagnostic, never tuned on | 30 labelled vials, 10 recipes; no per-vial outcome |
| **CSI 10/27 session** | **Primary real evaluation** of decisions and RCA | Our vial log: intended/actual R/Y/B, fault type, times |
| Synthetic/simulated | Fitting and threshold selection (policy) | Generated |

Policy: real imagery is evaluation-only and Run 5 stays out of fitting/tuning. **Decision needed**:
whether reference vials (and the 3-parameter mixing model) from a commissioning subset of the CSI
session count as commissioning rather than fitting. Proposed: yes, with the first vial of each Tier 1
batch as references and every other vial scored.

## Schedule

| Day | Deliverable |
| --- | --- |
| Tue 10/27 | Plan agreed; CSI data copied, checksummed, vial log reviewed |
| Wed 10/28 | Stage 2 automatic vial segmentation; offline runner writes the ledger; evidence-packet schema |
| Thu 10/29 | Stage 4 actions + deadlines; metrics harness on CSI + Run 5; RCA comparison (rule/LLM/Cosmos) |
| Fri 10/30 | **Freeze code and thresholds.** Scored run on GB10; app shows the ledger and RCA |
| Sat 10/31 | Demo video, metrics tables, write-up |
| Sun 11/1 | Submit |

## Workstreams (proposed owners)

| Stream | Scope | Proposed |
| --- | --- | --- |
| Pipeline + integration | Stages 2–4, `replay.line` runner, ledger, GB10 | Austin, Srijan |
| App | Upload video + schedule, live ledger, actions, RCA panel | Josiah |
| RCA | Evidence packets, LLM + Cosmos comparison, output schema | Zoe, Jev |
| Data + ground truth | CSI log QA, fill-window labels, fault labels | Aleyna, Reagan |
| Synthetic faults | Simulated carryover/dose cases for threshold freeze and RCA tests | Cameron |

## Not in scope before submission

Isaac/twin rendering, Cosmos Transfer synthetic video, SAM/Faster R-CNN comparators, Dataset 1,
TensorRT, new model training. Code stays in the repo; see [branches](branches.md).

## Open questions for CSI

- Their out-of-spec tolerance, and what Teledyne would have reported for our fault vials.
- Time from fill end to the sorter (sets the SCRAP deadline).
- Their flush procedure: what triggers it today and how long it takes.
- The most common real causes of bad vials, to prioritise the RCA taxonomy.
