# One-hour CSI visit: three recipes, references, then a locked run

Tanks available: T100 clear, T200 orange, T400 blue. Setup follows the
[demo-day checklist](csi-session.md#demo-day-checklist); this page is the run sheet.

## Recipes

Shipped library recipes, so nothing new has to be typed in, and the same three Run 5 started with
(the evaluation numbers apply to them directly):

| Recipe | T100 clear | T200 orange | T400 blue | Dispense order | Looks |
| --- | --- | --- | --- | --- | --- |
| **5** | 10 mL | 10 mL | 10 mL | T200 → T400 → T100 | dark mixed (green-brown) |
| **6** | 30 mL | 10 mL | — | T200 → T100 | pale orange |
| **7** | 20 mL | — | 50 mL | T400 → T100 | blue |

Distance between their Run 5 references (ΔE2000; band is 5): 5↔6 25.8, 6↔7 33.5, 5↔7 15.6.
So 6↔7 is the easy changeover and 7→5 the hard one.

## Timeline (about 56 s per vial at Run 5's pace)

| Time | Step | Vials |
| --- | --- | --- |
| 0:00–0:15 | Set up the GB10, close CSI's recorder, cameras in, overlays cover vial and tray | — |
| 0:15–0:25 | **A. References**: 5 ×3, 6 ×3, 7 ×3 | 9 |
| 0:25–0:27 | **Lock**: CREATE off, check the three references, no more changes | — |
| 0:27–0:40 | **B. Locked test run**: 6 ×3, 7 ×3, 5 ×3, 6 ×3 | 12 |
| 0:40–1:00 | Buffer: redo a missed reference, or the optional wrong-recipe vial, then save and copy data | ≤ 6 |

## A. References (9 vials)

1. LINE → **Recipe setup**: recipe 5 × 3, recipe 6 × 3, recipe 7 × 3 → **Save & use on line**.
   CSI programs the station with the same order.
2. **Start** on the station cameras.
3. Arm **RECREATE reference from next vial** for the recipe just before the **third** vial of each
   recipe (vials 3, 6 and 9). The first vial after a change still holds the previous recipe
   (Run 5: vial 5, the second of recipe 6, was still carryover), so the third is the clean one,
   the same rule as Run 5's references.
4. Check that each one took: the recipe shows a new reference with this session's vial number. If
   QA withheld the colour, CREATE stays armed; run one more vial of that recipe.

Vials 1–2, 4–5 and 7–8 are judged against the shipped (Run 5) references until each is replaced.
That is fine; they are not the test.

## Lock in

From here nothing changes: CREATE stays off, thresholds stay at band 5 / direction 0.8, models
unchanged. Write the three reference vials and the time down. If a reference looks wrong (bubble,
glare, half fill), redo it now, not during B.

## B. Locked test run (12 vials)

Order: **6 6 6 · 7 7 7 · 5 5 5 · 6 6 6** (four changeovers: 7→6, 6→7, 7→5, 5→6). New Recipe setup
with these batches → **Save & use on line**, CSI programs the same, **Start**.

Expected, if the line behaves like Run 5:
- the first vial of each new recipe is **OUT OF SPEC**, carryover, with **SCRAP + FLUSH**;
- the second may still be off, the third **IN SPEC** / PASS;
- the 7→5 change is the closest pair, so the weakest signal.

Watch **Recipe check** fill in live during the run; that page is the demo screenshot.

## Optional: wrong recipe (1 vial)

If time allows: the order says 6 but CSI fills 7 (or the reverse). Expect **OUT OF SPEC** with a
deviation or carryover pattern. Say on the recording that it was deliberate; it is not a real
defect.

## Before leaving

- Stop the session so it is saved (a run named `live-…`), then copy `~/linewatch-data`
  (`line-recordings/` and `site-recipes.json`).
- Note on paper: start time, the reference vials, anything CSI did differently (flushes, refills).
- The recordings are evaluation data only: no training or threshold changes on them.

## Rehearse now

Block A is exactly Run 5's first nine vials, so it can be rehearsed with the mock cameras:
`bash START.sh --mock-fill vial_fill_video_5.mp4 --mock-tray cap_seal_video_5.mp4`, enter the A
order, **MOCK** → **Start**, and arm RECREATE before vials 3, 6 and 9.
