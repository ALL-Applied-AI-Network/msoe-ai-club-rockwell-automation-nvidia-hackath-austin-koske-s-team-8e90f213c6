# Speaker notes for the slide figures

The figures carry almost no text on purpose; these notes carry the explanation. Numbers come from the
final CSI run (`live-20261030-193307`, 30 Oct, 4 blue → 4 red, live on CSI's GB10) and from the
Run 5 evaluation. "Say" lines are suggestions; "If asked" lines are backup.

---

## Problem: `fig-changeover.png`

**Say:**
- "This is a real vial from our run at CSI yesterday. We ordered four blue, then four red."
- "Vial 5 is the first red one. Look at it: it's green."
- "Blue was still in the line, and blue plus red makes green." *(point at the swatches)*
- "A colour check that doesn't know the order sees a perfectly valid colour and lets it through."
- "Linewatch knows the order. This vial is 32 away from red and only 10 from blue, so the colour
  moved toward the previous recipe. That's carryover."
- *(timeline)* "The operator got four messages:"
  - **4:41** "a flush recommendation before the vial was even filled, because blue to red is a big jump";
  - **5:19** "an early warning 9 seconds into the fill";
  - **5:44** "OUT OF SPEC, carryover";
  - "and a SCRAP with a 96-second deadline, before the vial reaches the sorter."

**If asked:**
- **ΔE** is the standard colour-difference score (CIEDE2000). Under 5 counts as the same recipe.
- **"cos 0.97 toward blue"** means the colour moved almost exactly in the blue direction.

---

## Live results: `fig-run.png`

**Say:**
- "Here's the whole run, all eight vials, photographed by the station's own camera at the end of
  each fill."
- "Red outline means Linewatch said no; the bar is how far off it was. The dashed line is the
  tolerance."
- "Right after the change to red, vials 5 and 6 still carry blue, both caught as carryover. SCRAP
  and FLUSH."
- "By vial 7 the line is clean, and it passes."
- "The same thing happened at the start: the line still held red from our earlier runs, so the
  first two 'blue' vials came out brown and teal. Also caught."

**Be careful:**
- Vials 1–2 are labelled "deviation" in the app, not carryover: that session had no previous recipe
  on record. Say "leftover from earlier runs" and don't count them as carryover catches.
- In this run every vial went to the Bad tray, because the vials had no caps. Don't claim tray
  sorting results from this run.

**Takeaway:** the line lags one to two vials behind each recipe change. A single water flush isn't
always enough, and that's what the FLUSH/SCRAP logic is built for.

---

## Solution / demo: `fig-live.png`

**Say:**
- "This is the app running on CSI's GB10, at the line, on the station cameras."
- Walk through the numbers on the figure:
  - **1. Fill camera.** "Green is the liquid our model sees, blue is the vial. The colour is measured
    while it fills; right now it's 5.5 off."
  - **2. Decision.** "OUT OF SPEC, carryover, SCRAP due in 96 seconds, flush one water vial, and a
    reason for every action."
  - **3. Tray camera.** "A second camera confirms which tray and slot each vial lands in."
  - **4. Explain with AI.** "Nemotron 3 Ultra explains the cause in plain language. The rules
    decide; the AI explains."
- "It doesn't touch the PLC. CSI doesn't allow that, so it tells the operator what to do and by
  when."

---

## Physical AI: `fig-sim2real.png`

**Say:**
- "On the left is not a photo. It's an Isaac Sim render of a digital twin of the station."
- "Both vision models, the YOLO vial detector and a temporal U-Net for the liquid, were trained
  only on renders like this. No real labelled images."
- "On the right are the same models on CSI's real camera, live on the GB10, finding the vial and
  the liquid."
- "We generated 14,400 extra carryover renders on Brev to study how carryover looks."

**If asked:**
- **Detector transfer.** The released detector struggled on CSI's new camera angle (0/944 on the
  first session). A fixed station vial region plus a detector retrained on CSI-view renders fixed
  it. Still synthetic-only training.
- **Run 5.** It's the organizers' labelled run and evaluation only: no training or tuning on it.

---

## Architecture: `fig-architecture.png`

**Say:**
- "Read it left to right: two cameras go into the GB10."
- "The green parts are learned: Isaac-trained perception. Measurement QA throws away colour it can't
  trust."
- "Everything after that is fixed, auditable rules: compare to the recipe's reference, check the
  carryover direction, issue actions with deadlines."
- "Nemotron on NVIDIA's API only explains. It never decides."
- "A new recipe needs one good vial as its reference, not a training set."

---

## Results: `fig-results.png`

**Say:**
- "Two kinds of evidence."
- **Left, the organizers' Run 5:** "28 minutes of recorded video, with only the recipe order given.
  All 30 vials found and assigned the right recipe. Every measurable first vial after a change was
  caught, and the line itself had passed all of them. All 9 scraps landed before the sorter. The
  early warning fires a median 18 seconds before the end-of-fill decision."
- **Right, live at CSI yesterday:** "8 of 8 vials decided live. 2 of 2 carryover vials caught,
  all 4 off-spec vials scrapped, and a warning 25 seconds before the decision."

**Be honest if pushed:**
- No per-vial ground truth exists, so these are detection counts, not accuracy.
- The live run is one 8-vial session.

---

## Suggested order (3-minute talk)

1. Title
2. `fig-changeover` (the green vial)
3. `fig-architecture`
4. `fig-sim2real`
5. `fig-live`
6. `fig-run`
7. `fig-results`
8. Impact / close
