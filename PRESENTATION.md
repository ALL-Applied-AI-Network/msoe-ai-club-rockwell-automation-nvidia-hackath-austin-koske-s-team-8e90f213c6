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

**Learned perception, deterministic decisions.** The models (trained only in simulation) find the
vial and liquid and measure its colour. Everything on top of that is a fixed rule an operator can
audit, including the early warning, which compares each moment of the fill with one reference
vial's fill. A new recipe needs one accepted vial, not a training set.

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
- **Early warning:** off-recipe vials flagged *while still filling*, a median 18 s before the
  endpoint decision (6/6 with a reference trajectory; 1/4 good vials falsely warned).
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
| Can it warn before the fill ends? (Run 5) | **Decide after the fill:** median 8.4 s after the fill ends | **6/6** bad vials that have a reference trajectory were warned *during* the fill, **median 18.2 s earlier** (16.5–32.6 s), as early as 2.5 s into the fill; 1/4 good vials warned | Rule changed twice after seeing Run 5 (disclosed); small sample; recipe 6 has no trajectory. Needs fresh CSI data |
| Is the cause right? (20 Run 5 packets) | **Rule explainer** | Rule: 20/20 consistent with the evidence. Cosmos-Reason1-7B: 6–9/20; Qwen2.5-VL-7B: 5/20 (both call nearly everything carryover); OpenAI Decisions API: 20/20; Jev 1.14: 17/20 | No cause labels exist. The rules still decide; models only word the explanation |
| Does carryover get flagged? (synthetic, held-out families) | **Current rule** (ΔE 5, direction 0.8): flags 0.5% of vials at 0–10% residue, 22% at 10–30% | Thresholds tuned on train families: **41% / 74%**, and 86–93% above 30% residue; 2.4–2.7% false alarms | Synthetic noise is lower than real noise, so these thresholds are **not frozen** |
| How much residue? (synthetic) | **Lab colour projection:** 0.053 average error in residue fraction | **CNN (3 seeds): 0.026–0.028**; CNN switching to the projection above 50%: 0.025–0.026 | Worse than the projection above 50% residue (0.074 vs 0.052). Synthetic only |
| Can the order avoid scrap? (300 simulated queues, 18 vials each) | **Requested order plus current flush rule:** at f = 0.2, 0.20 scraps + 2.62 flushes per queue | **Reordered (no run moves more than 2 places): 0.03 scraps + 0.48 flushes**; at f = 0.3: 1.59 + 2.62 → 0.49 + 0.75 | Predicted with our own line model, not measured. f is the share of old liquid carried into the next vial; CSI's real f is unknown, and CSI's clips suggest a one-vial lag instead |
| Does the tray camera confirm sorting? (Run 5) | **Counting yellow caps:** 25 | **Slot tracking: 26/26** arrivals matched to vials; transit 128 s (median); 2 cap problems found | Slot grid set while looking at Run 5 (demonstration, not blind) |
| Does it run at the line? | **x86 CPU:** about 300 ms per frame | **GB10 CUDA:** 86 ms warm; self-test 40/40 on spark8 and twice on CSI's GB10; 24 ms per model sample on an RTX 4080; **full Run 5 through the live camera path in real time (MOCK, 28 min): 30/30 vials, 50,941/50,941 frames, 0 dropped, ≤0.2 s behind, 30/30 verdicts identical to the batch run, actions identical (9 SCRAP, 5 FLUSH, 4 FLUSH RECOMMENDED, 10 PASS, 11 HOLD)** | Real USB camera drop-outs only partly exercised (10 CSI sessions) |
| Does the vial model transfer to CSI's camera? | **YOLO11n:** 0/944 vial boxes on the new camera | **Fixed station vial region:** 59/61 samples measured | No human labels on that recording; the vial region needs operator approval |

Full tables and records: [evaluation](docs/evaluation.md).

---

## Video plan (5 minutes: 3 talk + 2 demo)

### Talk (3:00)

| Time | Say | On screen |
| --- | --- | --- |
| 0:00 (alt. opener) | "Poor quality costs manufacturers 15 to 20 percent of sales. The top cause of US food recalls, about 4 in 10, is the previous product ending up in the next one, and a recall costs about ten million dollars. It happens at changeovers, and lines change over more every year." | Two big numbers: 15–20% · $10M ([sources](docs/industry-research.md)) |
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
| 1:05 | Next vial of the recipe starts filling off-colour → amber **EARLY WARNING** on the fill camera | "It hasn't even finished filling, and Linewatch already says it's off recipe." |
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

## Impact numbers for the slides (sourced)

Full research with a trust rating per source: [docs/industry-research.md](docs/industry-research.md).
The strongest hooks:
- **Allergen recalls:** undeclared allergens cause 38.8% of US food recalls, and the usual
  mechanism is the previous product's residue (cross-contact).
- **Rockwell's own survey:** quality control is manufacturers' #1 AI use case.
- **Inspectors:** human inspection is benchmarked at about 80% accuracy.
- **Labour:** 1.9M US manufacturing jobs could go unfilled by 2033.

**The cost of the problem**
- **Cost of poor quality:** commonly put at about **15–20% of sales** for manufacturers, and up to
  40% at the worst performers. These are ASQ-attributed estimates, as quoted by
  [ReliaMag](https://reliamag.com/articles/cost-of-poor-quality-scrap-rework-returns/); an IISE
  article gives about 15% on average, in a 5–35% range
  ([IISE](https://iise.org/details.aspx?id=22118)).
- **Food and beverage recalls:** about **$10M in direct costs** on average
  ([FMI/GMA study, via Food Safety](https://www.food-safety.com/signature-series/recall-the-food-industrys-biggest-threat-to-profitability/)).
  Allianz puts a significant food and beverage recall at about €8M (about $9.5M), and **$12M** across
  all sectors, for 2012–2017
  ([Allianz via FoodNavigator](https://www.foodnavigator.com/Article/2017/12/13/Product-recall-risks-growing-in-size-and-number-Allianz),
  [Business Insurance](https://www.businessinsurance.com/Product-recalls-cost-companies-$12-million-on-average-Allianz-/)).
- **Pharma fill-finish:** a single failed biologic or parenteral batch is put at **$0.5M to several
  million**. That figure comes from vendor estimates, not a peer-reviewed benchmark
  ([iFactory](https://ifactoryapp.com/industries/manufacturing-plant/ai-pharma-batch-consistency-manufacturing)).
- **Market:** industrial machine vision is about **$7–18B in 2026**, depending on scope, and quality
  inspection is a main growth driver
  ([GMI](https://www.gminsights.com/industry-analysis/industrial-machine-vision-market),
  [Fortune Business Insights via GII](https://www.giiresearch.com/report/fbs2020111-machine-vision-market-size-share-growth-global.html)).

**Where Linewatch applies:** any line that changes recipes and has a visible product, such as
beverages and syrups, paints and coatings, chemical dosing, cosmetics, and pharma fill-finish.
It needs a camera at the fill point, the production order and one accepted unit per recipe.

**What our data says about the problem's size.** In Run 5, 10 of the 30 vials (1 in 3) were the
first of a new recipe. All 5 that could be measured were off recipe, and all 26 vials that
reached the trays went to Good.

**What it cost to build** (all sponsor-provided):

| Resource | Used | At list price |
| --- | --- | --- |
| Brev, 2× L40S (NVIDIA credits) | 9.7 instance-hours | about **$35** ($3.61/h) |
| Rosie H100 (MSOE) | 2.65 GPU-hours | about **$5–18** at $2.00–6.88 per H100-hour on demand ([IntuitionLabs](https://intuitionlabs.ai/articles/h100-rental-prices-cloud-comparison)) |
| GB10 / DGX Spark: spark8 plus CSI's unit at the line | about 4 GPU-hours, plus deployment | **$3,999 at launch, $6,950 now** per unit ([Tom's Hardware](https://www.tomshardware.com/desktops/mini-pcs/nvidia-dgx-spark-gets-18-percent-price-increase-as-memory-shortages-bite-founders-edition-now-usd4-699-up-from-usd3-999), [The Register](https://www.theregister.com/a/5300622)) |
| build.nvidia.com hosted models | Nemotron and Llama vision explainers | free API tier |
| **Total** | **about 16 GPU-hours** | **about $40–55 of cloud compute**, plus two GB10s worth about $14k at today's price |

The slide line: *"Trained on about $50 of GPU time, entirely in simulation, and running on one
desktop GB10 at the line."*

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

---

## Answers to the submission questions

Each question from the organizers' brief, answered directly. Numbers are from
[evaluation](docs/evaluation.md) and the records in `setup/verification/`.

### Executive summary questions

**What are your goals?**
Find off-recipe vials, mainly carryover after a recipe change, before they reach the sorter.
Tell the operator what to do and by when (scrap, flush, stop and check dosing, hold), and why.
Do it with the station's existing cameras and a GB10 at the line, with vision models trained only
on simulation, and measure honestly what works and what doesn't.

**What data did you collect or generate?**
- **Collected (evaluation only, never training):** 10 LINE sessions on CSI's GB10 with the
  station's fill and tray cameras (about 50 min of fill video, 5,859 model samples, 14 vials).
  6 CSI-recorded flush clips with mL metadata.
- **Supplied by the organizers:** CSI Dataset 2, Runs 1–5. Runs 1–4 set the rules; Run 5 (30
  labelled vials, 28 min) was scored once with frozen settings.
- **Generated:**
  - 192 Isaac Sim renders of the station digital twin for the liquid model.
  - 14,400 Isaac Sim carryover renders on Brev (4,800 changeovers, 600 scene families).
  - 1,620 analytic fault sequences for the diagnosis pilot.
  - 300 simulated recipe queues for the scheduler.
- **Compute:** about 16 GPU-hours (Rosie 2.65, GB10 about 4, Brev 9.7).

**What is your approach?**
A recipe-aware inspection pipeline. For each vial:
1. A YOLO11n vial detector and a temporal U-Net liquid segmenter find the liquid. Measurement QA
   then gives a colour only when it can trust one.
2. The finished colour is compared with the recipe's reference (ΔE2000). An out-of-band vial
   whose colour points toward the previous recipe is carryover.
3. Rules turn that into an action with a deadline: SCRAP before the sorter (120 s), FLUSH before
   the next fill, STOP & CHECK DOSING after 3 non-carryover deviations, or HOLD.
4. The tray camera confirms where each vial landed. The cause is explained by rules; Cosmos,
   Qwen, Jev and the OpenAI Decisions API were evaluated against them.

New recipes get a reference at the line through CREATE mode. Everything runs on a GB10 as a
self-tested kit.

**What manufacturing problem are you solving?**
The station's colour check is not told which recipe was ordered, so an off-recipe vial passes
whenever its colour is a known colour. The first vial after a recipe change often still holds the
previous recipe. In Run 5, all 5 measurable first vials after a change were 6–24.5 ΔE off recipe
(vial 4: 24.5 from recipe 6, 1.6 from recipe 5). All 26 vials that reached the trays went to Good.

**How does your project use physical AI?**
- **Simulation-only training:** both vision models were trained on Isaac Sim renders of a digital
  twin of the station, with no real training labels.
- **Simulated carryover:** 14,400 renders on Brev L40S GPUs to study carryover and set
  thresholds.
- **Edge inference at the line:** real-time on a GB10 with live cameras.
- **Physical reasoning models:** Cosmos-Reason1-7B and Qwen2.5-VL-7B were evaluated on the GB10
  as cause explainers.
- **Physical actions:** every perception result becomes a physical line action with a deadline.

**How did you use the CSI equipment?**
- **The GB10:** we installed our kit on CSI's GB10, and the self-test passed 40/40 twice.
- **Live sessions:** we ran 10 LINE sessions on the station's fill and tray cameras: 5,859 model
  samples, 14 vials decided, 4 tray arrivals.
- **Camera transfer:** we measured it on the new camera. The vial detector found 0 of 944
  vials; a fixed station vial region recovered 59/61 samples.
- **Tray profile:** we set up the dual Good/Bad tray profile on the live tray camera.
- **Flush recordings:** CSI staff recorded water-flush sequences for us. They show the line
  behaves like a one-vial lag.

**What evidence shows your solution works?**
- **Run 5 offline:** with only the recipe order, 30/30 vials found, 0 false positives, and 30/30
  assigned the right recipe. All 5 measurable first vials after a change were flagged; 9/9 SCRAPs
  went out before the 120 s deadline and 5/5 FLUSHes before the next fill.
- **Early warning (Run 5):** 6/6 off-recipe vials with a reference trajectory flagged during the
  fill, a median 18.2 s before the endpoint decision; 1/4 good vials falsely warned. The rule was
  adjusted twice after seeing Run 5, so this needs confirming on fresh CSI data.
- **Run 5 through the live camera path in real time (28 min):** the same 30/30 verdicts, 0 of
  50,941 frames dropped, never more than 0.2 s behind.
- **Explanations:** the rule explainer was consistent with the evidence on 20/20 vials, against
  6–9/20 for zero-shot Cosmos and 5/20 for Qwen.
- **Carryover (synthetic held-out families):** thresholds tuned on synthetic data flag 74% of
  10–30% residue, against 22% for the current rule, at 2.4–2.7% false alarms.
- **On CSI hardware:** self-test 40/40 and the 10 live sessions above.

### Video questions

**Show your solution running on the CSI hardware.**
The demo is filmed on the LINE page on CSI's GB10 with the station cameras (demo script above).
An on-site screen recording from 2026-10-30 already exists
(`results/screen-recordings/linewatch-FINAL-20261030-000437.mp4` in the CSI handoff). If the line
is idle, MOCK plays Run 5 through the same camera path on the same GB10.

**What specific problem does it solve, and why does it matter?**
It catches the vial that is the wrong recipe even though its colour looks normal, mostly the
first vial after a recipe change. It tells the operator to scrap it about 2 minutes before it
reaches the sorter, and to flush the line. Today that vial ships as Good. That costs a customer
complaint or a recall, and finding it later costs inspection labour.

**How would it help other manufacturers?**
It needs only a camera on the fill point, the production order and one accepted vial per recipe
(CREATE mode). No PLC integration is required. That fits any batch process with recipe changeovers
and a visible product: paint and coatings, beverages, chemical dosing, cosmetics, pharma fill-finish.

| Cost | How Linewatch reduces it | Our evidence |
| --- | --- | --- |
| **Scrap** | Flush only when the colour shows carryover, not after every change; order recipes so the next one tolerates what the line holds | Synthetic queues (predicted): the current flush rule uses 2.62 flushes per 18 vials; reordering needs 0.48 at f = 0.2, with fewer scraps (0.20 → 0.03) |
| **Rework and returns** | Off-recipe units are diverted before the sorter, not found downstream | Run 5: 9/9 SCRAPs issued at least 100 s before the vial reached the tray |
| **Downtime** | STOP & CHECK DOSING after 3 consecutive non-carryover deviations, instead of finding a dosing fault at the end of a batch | Synthetic diagnosis pilot: persistent wrong recipe reported as persistent deviation in 162/162 cases |
| **Labour** | Every vial is checked automatically, and the operator gets an action, a deadline and a reason instead of inspecting by eye | Live path: decision a median 8.4 s after fill end, 24–86 ms per model sample |

**What are your measured results against a baseline?**
See [Results against baselines](#results-against-baselines). In short:
- **The line as recorded** passed every vial to Good. Linewatch flagged all 5 measurable
  off-recipe first vials and scrapped 9 vials in time.
- **The current carryover rule** flags 22% of 10–30% residue; the tuned rule flags 74%.
- **Lab projection** residue error is 0.053; the CNN's is 0.026–0.028.
- **The rule explainer** is consistent with the evidence on 20/20 vials, against 5–9/20 for
  zero-shot vision models.
- **x86 CPU** takes about 300 ms per frame; the GB10 takes 86 ms.

**Where does it fall short?**
- **No outcome truth:** no dataset records what each vial actually was, so real-data results are
  detection counts against proxy references, not accuracy.
- **Camera transfer:** the vial detector did not transfer to CSI's current camera, and the fixed
  vial region needs operator approval.
- **One-vial lag:** CSI's flush clips suggest the line lags about one vial, which the scheduler's
  model doesn't capture yet, and one water vial may not clear a change.
- **Synthetic thresholds:** not frozen, because synthetic noise is below real noise.
- **Missing references:** 5 of Run 5's 10 first vials can't be checked, because those recipes have
  no reference.

### Links to supporting material

- Repository: <https://github.com/ALL-Applied-AI-Network/msoe-ai-club-rockwell-automation-nvidia-hackath-austin-koske-s-team-8e90f213c6> (`main`)
- [Evaluation](docs/evaluation.md), [architecture](docs/architecture.md), [line view](docs/line-view.md), [compute](docs/compute.md), [data](docs/data.md)
- Records:
  - Run 5 line runner: [`line-runner-run5-20261027.json`](setup/verification/line-runner-run5-20261027.json)
  - Tray reconciliation: [`tray-reconciliation-20261028.json`](setup/verification/tray-reconciliation-20261028.json)
  - Cause explainers: [`rca-explainers-run5-20261028.json`](setup/verification/rca-explainers-run5-20261028.json), [`rca-jev-run5-20261028.json`](setup/verification/rca-jev-run5-20261028.json), [`rca-openai-decisions-run5-20261028.json`](setup/verification/rca-openai-decisions-run5-20261028.json)
  - Synthetic carryover: [`carryover-synthetic-20261029.json`](setup/verification/carryover-synthetic-20261029.json)
  - GB10 native run: [`gb10-native-20261027.json`](setup/verification/gb10-native-20261027.json)
  - Model release: [`temporal-cnn-release-20261026.json`](setup/verification/temporal-cnn-release-20261026.json)
- Releases: [`temporal-cnn-v1.0.0`](https://github.com/ALL-Applied-AI-Network/msoe-ai-club-rockwell-automation-nvidia-hackath-austin-koske-s-team-8e90f213c6/releases/tag/temporal-cnn-v1.0.0) (models, training renders) and the `linewatch-gb10` kit
