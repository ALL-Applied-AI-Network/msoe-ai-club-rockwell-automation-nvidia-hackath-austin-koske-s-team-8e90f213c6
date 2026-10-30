# Slide visuals

Real artifacts for the 8-slide deck ([PRESENTATION.md](../PRESENTATION.md)). Nothing here is staged.

| Slide | File | What it is |
| --- | --- | --- |
| 1 Title | `01b-fill-station-csi-live.png` | CSI fill camera, live session on the GB10 (2026-10-30) |
| 1 Title (alt) | `01-fill-station-run5.png` | Run 5 fill camera, vial 4 mid-fill |
| 2 Problem | `02-vial4-vs-references.png` | Run 5 vial 4 between the recipe 5 and recipe 6 reference vials, fill-end frames |
| 2 Problem | `02b-vial{3,4,6}-crop.png` | The three crops alone, to lay out yourself |
| 3 Opportunity | `03-tray-camera-run5.png`, `03b-tray-camera-csi-two-grids.png` | Where the vial ends up: Run 5 tray; CSI Good/Bad grids with the slot profile |
| 4 Solution | `06-csi-gb10-line-page-1440s.png` | **The LINE page live on CSI's GB10**: masks, OUT OF SPEC, SCRAP "due 61 s", tray camera |
| 4 Solution | `04-architecture-system.png`, `04-architecture-line-runner.png`, `04-line-view-live.png` | Pipeline diagrams and the LINE page in live mode |
| 5 Physical AI | `05-synthetic-vs-real.png` | Isaac Sim training render next to the CSI camera |
| 5 Physical AI | `05b-synthetic-contact-sheet.png` | A sheet of carryover renders (Brev) |
| 6 Results | `09-early-warning-vial7.png` | **EARLY WARNING while vial 7 is still filling** (ΔE 24.0 at 7.5 s in) |
| 6 Results | `09b-early-warning-vial1.png`, `09c-scrap-vial4.png` | Early warning on vial 1; SCRAP + FLUSH on vial 4 |
| 6 Results | `07-annotated-run5-vial4-scrap.png` | Annotated Run 5 video at vial 4's decision |
| 7 Validation | `06-csi-gb10-line-page-{200,280,1400,1520}s.png` | More frames of the app running on the CSI GB10 |
| 5 or 7 | `10-vial-detector-before-after-csi.png` | **Same CSI frame: released detector finds 0 vials; retrained on CSI-view Isaac renders finds the vial (0.88) and liquid (0.97)** |
| 6 or 7 | `11-known-carryover-recolour.png` | Real Run 5 vial with 0/10/20/30% of the previous recipe injected (known-answer test). The orange halo is the liquid mask spilling onto glass; mention it if shown |
| 4 or 6 | `12-ai-explanation-vial4.png` | **Ledger row for vial 4 opened: rule cause, each action's reason, and Nemotron 3 Ultra's advisory explanation (carryover 95%, next check: was the flush run)** |
| 4 or 6 | `12b-panel-actions-with-reasons.png` | Side panel: every action with its reason (FLUSH, SCRAP, EARLY WARNING, FLUSH RECOMMENDED) |
| 8 Impact | `08-repo-qr.png` | QR code to the repository |

One number to keep consistent: for vial 4, the line runner (and these images) measure ΔE 24.0
to recipe 6 and 8.2 to recipe 5. The earlier hand-windowed Run 5 check gave 24.5 and 1.6.
Quote one pair per slide, labelled with where it comes from.

## Slide figures in the deck's theme (`slides/`)

White-background PNGs (2×) in Century Gothic, charcoal and olive green, kept low on text; talk track in `slides/SPEAKER-NOTES.md`. Built in React
from `slides/src/` (`slides.src.html?fig=…`, `arch-slide.html`; render with `render.py` on Windows/Edge).
Real footage is the final CSI run, `live-20261030-193307` (4 blue → 4 red) from
`Linewatch-FINAL-DAY2-20261030.zip`; numbers are that session's ledger.

| File | Use |
| --- | --- |
| `fig-architecture.png` | Architecture overview, recoloured to the deck |
| `fig-run.png` | **The final CSI run**: 8 vial photos under the ordered recipe, verdict, ΔE bar vs band 5, action |
| `fig-changeover.png` | Vial 5 (green, ordered red) between the blue and red vials, with the operator's timeline: FLUSH RECOMMENDED → EARLY WARNING → OUT OF SPEC carryover → SCRAP due 96 s |
| `fig-live.png` | LINE page live on CSI's GB10 (masks on, vial 5 decision) with 4 numbered callouts |
| `fig-sim2real.png` | Isaac Sim render → the same models on the CSI camera |
| `fig-results.png` | Stat tiles: Run 5 and live at CSI |
