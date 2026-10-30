# Linewatch: recipe-aware vial inspection for CSI Fluid Mixing Station 3

Rockwell Automation / NVIDIA Physical AI hackathon (MSOE, UWM CSI) · Austin Koske's team ·
submission due Sunday, November 1, 11:59 p.m. CT.

Linewatch watches the filling station with the station's own cameras. It knows which recipe each
vial should be, and decides for every vial whether it is that recipe and what to do: pass, scrap
before the sorter, flush the line, stop and check dosing, or hold. It also says when the action
is due and why. It runs on an NVIDIA GB10 at the line. Its vision models were trained only on
Isaac Sim renders of the station.

- Code: this repository (`main`). Every number below is reproduced in [evaluation](docs/evaluation.md).
- Kit: `linewatch-gb10` release plus `temporal-cnn-v1.0.0` models ([data](docs/data.md)).

---

## Executive summary (one page)

**The manufacturing problem.** At CSI the vial colour check (Teledyne) compares each vial against
a static colour database and is not told which recipe was ordered. A vial of the wrong recipe
passes whenever its colour is some known colour. The main way that happens is carryover: after a
recipe change, the first vial still holds the previous recipe. In the organizers' Run 5, every
first vial after a change that we could measure was 6–24.5 ΔE2000 away from its recipe. Vial 4
was 24.5 ΔE from recipe 6 but only 1.6 from recipe 5. Every vial that reached the trays went to
the Good tray. For any manufacturer, an off-recipe unit that reaches the customer means scrap, rework or a
return. Today, catching it takes a person looking at every vial.

**Our approach.** Video and the recipe order go in. For every vial the system:
1. **Finds the vial and its liquid:** a YOLO11n vial detector and a temporal U-Net liquid
   segmenter, followed by measurement QA that refuses to report a colour it cannot trust.
2. **Measures the finished colour,** then compares it with that recipe's reference. If the vial
   is out of band, it checks whether the colour moved toward the previous recipe (carryover).
3. **Issues an action with a deadline:** SCRAP before the vial reaches the sorter (about 2
   minutes), FLUSH before the next fill, or STOP & CHECK DOSING after 3 deviations.
4. **Explains the cause** with a rule baseline. Language and vision models were compared against
   it.
5. **Checks the result at the tray:** a second camera confirms where each vial landed.

CSI does not allow connections to the PLC or sorter. Actions therefore go to the operator, and
are logged as the commands an integration would send.

**How it uses physical AI.**
- **Synthetic training:** both vision models were trained on Isaac Sim renders of a digital twin
  of the station, with no real training labels.
- **Synthetic carryover data:** 14,400 more renders on Brev L40S GPUs to study carryover.
- **Edge inference:** the system runs on a GB10 at the line, on live cameras.
- **Cause explanation:** Cosmos-Reason1 and Qwen2.5-VL were evaluated as explainers.
- **Physical actions:** every decision becomes a physical action with a time limit.

**How we used the CSI equipment.**
- **Kit on CSI's GB10:** our release kit passed its self-test (40/40 frames, twice).
- **Live sessions:** we ran 10 sessions on the station's fill and tray cameras, producing
  5,859 model samples and 14 vials segmented and decided.
- **Reference recording:** CSI recorded water-flush sequences for us.
- **Camera transfer:** the station's camera broke the vial detector (0 of 944 detections). A fixed
  station vial region recovered 59 of 61 samples.

**Data.**
- **Real (evaluation only, never training):** CSI Dataset 2, Runs 1–5 (Run 5: 30 labelled vials,
  28 minutes), plus our 10 CSI live sessions (about 50 minutes) and 6 CSI flush clips.
- **Synthetic:** 192 renders for the liquid model; 14,400 carryover renders (600 scene families);
  1,620 analytic fault sequences; 300 simulated recipe queues.
- **Compute:** about 16 GPU-hours.

**Evidence that it works** (next section). The full Run 5 video was processed with only the
recipe order:
- **Vials:** 30/30 found and 30/30 assigned the right recipe.
- **First vials after a change:** all 5 measurable ones flagged.
- **Timing:** 9/9 SCRAPs issued before the sorter deadline.
- **Explanations:** the rule explainer was consistent with its evidence on 20/20 vials. The best
  zero-shot vision model managed 9/20.

It falls short in four places:
- **No outcome truth:** there is no per-vial outcome ground truth.
- **Proxy references:** recipe references are proxy vials.
- **Detector transfer:** the vial detector does not transfer to CSI's current camera.
- **One-vial lag:** CSI's own flush clips suggest the line lags one vial. Our forecast model does
  not yet capture that.

---

## Results against baselines

| Question | Baseline | Linewatch | Limits |
| --- | --- | --- | --- |
| Are off-recipe vials caught? (Run 5) | **The line as recorded:** a recipe-unaware colour check; all 26 vials that reached the trays went to Good, including first vials 6–24.5 ΔE off recipe | **5/5** measurable first vials after a change OUT OF SPEC; **9 SCRAP, all before the 120 s deadline** (each with ≥100 s to spare); FLUSH 5/5 before the next fill; vial 5 likely saved by the flush after vial 4 | No per-vial outcome labels. References are Run 5 proxy vials (3rd vial of each recipe; those vials are excluded from scoring). 5 first vials can't be checked: recipes 8, 9 and 13 have no reference |
| Can it follow the line on its own? (Run 5) | **Organizers' hand labels** | **30/30** vials, 0 false positives, **30/30** right recipe; fill start/end within 1.2 s / 4.9 s (median); decision 8.4 s after fill end (median); 50,941 frames in 3 min 18 s on an RTX 4080 | Run 5 is one run. Settings were frozen on Runs 1–4 before it was scored |
| Is the cause right? (20 Run 5 packets) | **Rule explainer** | Rule: 20/20 consistent with the evidence. Cosmos-Reason1-7B: 6–9/20; Qwen2.5-VL-7B: 5/20 (both call nearly everything carryover); OpenAI Decisions API: 20/20; Jev 1.14: 17/20 | No cause labels exist. The rules still decide; models only word the explanation |
| Does carryover get flagged? (synthetic, held-out families) | **Current rule** (ΔE 5, direction 0.8): flags 0.5% of vials at 0–10% residue, 22% at 10–30% | Thresholds tuned on train families: **41% / 74%**, and 86–93% above 30% residue; 2.4–2.7% false alarms | Synthetic noise is lower than real noise, so these thresholds are **not frozen** |
| How much residue? (synthetic) | **Lab colour projection:** 0.053 average error in residue fraction | **CNN (3 seeds): 0.026–0.028**; CNN switching to the projection above 50%: 0.025–0.026 | Worse than the projection above 50% residue (0.074 vs 0.052). Synthetic only |
| Can the order avoid scrap? (300 simulated queues, 18 vials each) | **Requested order plus current flush rule:** at f = 0.2, 0.20 scraps + 2.62 flushes per queue | **Reordered (no run moves more than 2 places): 0.03 scraps + 0.48 flushes**; at f = 0.3: 1.59 + 2.62 → 0.49 + 0.75 | Predicted with our own line model, not measured. f is the share of old liquid carried into the next vial; CSI's real f is unknown, and CSI's clips suggest a one-vial lag instead |
| Does the tray camera confirm sorting? (Run 5) | **Counting yellow caps:** 25 | **Slot tracking: 26/26** arrivals matched to vials; transit 128 s (median); 2 cap problems found | Slot grid set while looking at Run 5 (demonstration, not blind) |
| Does it run at the line? | **x86 CPU:** about 300 ms per frame | **GB10 CUDA:** 86 ms warm; self-test 40/40 on spark8 and twice on CSI's GB10; 24 ms per model sample on an RTX 4080; live mock path: 0 dropped frames, ≤0.25 s behind | Real USB camera drop-outs only partly exercised (10 CSI sessions) |
| Does the vial model transfer to CSI's camera? | **YOLO11n:** 0/944 vial boxes on the new camera | **Fixed station vial region:** 59/61 samples measured | No human labels on that recording; the vial region needs operator approval |

Full tables and records: [evaluation](docs/evaluation.md).

---

## Video plan (5 minutes: 3 talk + 2 demo)

### Talk (3:00)

| Time | Say | On screen |
| --- | --- | --- |
| 0:00 | "CSI's colour check doesn't know which recipe was ordered. The first vial after a recipe change can be the old recipe and still pass. In the organizers' Run 5, every measurable first vial after a change was off recipe, and every vial that reached the trays went to the Good tray." | Run 5 vial 4 next to its recipe swatch: ΔE 24.5 from recipe 6, 1.6 from recipe 5 |
| 0:35 | "Linewatch gives the inspection the missing input: the recipe order. Two cameras, a GB10 at the line, models trained only on Isaac Sim renders of this station. For every vial: is it the ordered recipe, and what do we do, by when?" | Pipeline diagram (README) |
| 1:10 | "Measured on the full Run 5 video with only the recipe order: all 30 vials found and assigned the right recipe. All 5 checkable first vials caught. All 9 scraps issued before the sorter, with at least 100 seconds to spare." | Results table, first two rows |
| 1:45 | "Physical AI end to end. 14,400 renders on Brev to learn carryover; our tuned rule flags 74% of 10–30% residue against 22% for the current rule. Cosmos and Qwen were compared against our rule explainer, and lost; the rules decide, models only word it." | Carryover and RCA rows |
| 2:20 | "What doesn't work yet. There's no outcome truth, so these are detection counts, not accuracy. The vial detector failed on CSI's new camera; a fixed vial region recovered it. And CSI's own flush clips show the line lags about one vial, so one water flush isn't enough. That changes how scrap and flush should be planned, which is what our scheduler does next." | Shortfalls list |
| 2:50 | "For any manufacturer: off-recipe units are stopped before the customer, flushes happen only when needed, and the operator gets a reason and a deadline instead of eyeballing every vial." | Title card |

### Demo (2:00)

Recorded on CSI's GB10 with the station cameras, or on MOCK (Run 5 played as the cameras)
if the line is not running. Open `http://127.0.0.1:8788`.

| Time | Do | Say |
| --- | --- | --- |
| 0:00 | LINE page, **Live**, **Webcams** (or **Mock**) → **Start** | "These are the station's two cameras, live on the GB10." |
| 0:15 | **MASKS ON** on the fill camera | "Green is the liquid the model sees; blue is what QA actually measures." |
| 0:30 | **Recipe setup:** **New recipe** "Blue 50" (50 mL clear, 50 mL blue) → **Add to order** → **CREATE reference from next vial** | "New recipe, no reference yet. The next vial becomes its reference, with a record of where it came from." |
| 0:55 | Vial decided → `REFERENCE SET` | "Every later vial of Blue 50 is judged against that colour." |
| 1:10 | Recipe change: first vial after it → **OUT OF SPEC**, **SCRAP** with countdown, **FLUSH** | "First vial after the change still looks like the old recipe: scrap before the sorter, flush before the next fill." |
| 1:35 | **DETECTIONS ON** on the tray camera; the arrival appears | "The tray camera confirms where it went." |
| 1:50 | **Stop** → open the run in **Review** | "Every session is saved and can be reviewed frame by frame." |

On MOCK, Run 5's recipe order is preloaded. **Start at (min)** 2 shows the 5 → 6 change at
vial 4 within a minute. Rehearse with `setup/gb10/e2e_mock.py` ([line view](docs/line-view.md)).
An on-site recording already exists in the CSI handoff
(`results/screen-recordings/linewatch-FINAL-20261030-000437.mp4`).

---

## Data and compute

| Data | Amount | Use |
| --- | --- | --- |
| CSI Dataset 2, Runs 1–4 | 4 runs | Rule settings (endpoint window, segmentation), sanity checks |
| CSI Dataset 2, Run 5 | 30 vials, 28 min, fill and tray cameras | Evaluation only, scored once with settings frozen |
| CSI live sessions on the GB10 (2026-10-29) | 10 sessions, about 50 min of fill video, 5,859 model samples, 14 vials | Evaluation and camera transfer |
| CSI flush recordings (2026-10-28) | 6 clips with mL metadata | Evaluation: carryover behaviour |
| Isaac Sim renders, liquid model | 192 (48 scene families) | U-Net training and selection |
| Isaac Sim renders, carryover (Brev) | 14,400 (4,800 changeovers, 600 families) | FLUSH thresholds, residue estimator |
| Analytic sequences / simulated queues | 1,620 / 300 | Diagnosis pilot / scheduler benchmark |

| Compute | GPU-hours |
| --- | --- |
| Rosie (H100, Slurm), 22 jobs: detector and liquid training, sweeps | 2.65 (exact, `sacct`) |
| GB10 (spark8): Isaac renders, Cosmos, VLM explainers, benchmarks | about 4 (from job logs; upper bound) |
| Brev (2× L40S): carryover renders | 9.7 billed (about $35) |
| Laptop RTX 4080 SUPER: replays, residue CNN, live tests | not logged (minutes per run) |
| **Total** | **about 16** |

Details: [compute](docs/compute.md).

---

## Where it falls short

1. **No outcome truth.** No dataset records what each vial actually contained or where it was
   sorted. Every real-data figure is a detection count against proxy references, not accuracy.
2. **References come from the run itself.** Run 5's references are its 3rd vial per recipe. At
   the line, CREATE mode makes the same choice explicit and records it.
3. **Camera transfer.** The vial detector found nothing on CSI's current camera. The fixed station
   vial region works but needs operator approval, and the detector needs retraining on renders
   that match the new camera.
4. **Carryover is not a simple fraction.** CSI's flush clips hold more dye than the vial before
   them, consistent with a one-vial lag. So the scheduler's residue model and the
   one-water-vial flush both need CSI's confirmation and a recorded changeover with repeats.
5. **Synthetic thresholds are not frozen.** Synthetic measurement noise is far below real
   repeat-vial noise (2.6 ΔE).

---

## Submission checklist

- [ ] Video, about 5 minutes: the 3-minute talk and 2-minute demo above, showing the CSI GB10 and
      cameras.
- [ ] Executive summary: the one-page section above, exported to PDF.
- [ ] Links: this repository (`main`), the [evaluation](docs/evaluation.md), the release kit,
      and `setup/verification/*.json` records.
- [ ] Submit at the [final submission form](https://dashboard.all-ai-network.org/s/V28NHK) by Sunday,
      November 1, 11:59 p.m. CT. Any teammate can edit until then.
- [ ] Finalist invitations: accept by Wednesday, November 4. Finals: Thursday, November 5, Diercks
      Hall (posters 4:00 p.m.).
