# Synthetic training and digital twin presentation

**October24 presentation direction:** start with
[lessons, evidence-backed claims and five-minute outline](LESSONS-AND-CLAIMS-20261024.md).
The main adoption lesson is that promising perception needs a repeatable camera,
reference, measurement-QA and operator commissioning workflow on new data.
[AI-guided onboarding](../integration/AI-NATIVE-ONBOARDING.md) now has an initial
project skill and a tested read-only asset checker; end-to-end autonomous
training/deployment is proposed, not demonstrated. The
[current architecture](../../setup/vision/CURRENT-ARCHITECTURE.md) selects direct
U-Net with optional SAM refinement. Older PDFs below remain dated snapshots.

**October23 research update:**
[eight-fit fixture investigation](../../setup/vision/DATASET1-FIVECLASS.md) now
documents five-class limitations, normal-current improvements and a separately
scoped cap-state model. Its three-page code-free PDF is local:
`C:/Users/Austin/ai-club-tools/results/dataset1-followups-20261023/summary-v1/brief-v2/Digital-Twin-Inspection-Results.pdf`.
The same summary has an actual single-panel synthetic cap-proposal video, full
metrics and source verification. Fresh cap-state accuracy98.75% is synthetic;
body damage remains unassessed, authored lying-S poses are uncovered and reviewed
real accuracy is unavailable. Preserve the older PDF edition as dated history.

**Latest real probe:** the frozen synthetic RGB classifier has completed480
G slot-frame predictions on16 real cap_seal stills. See
[the actual transfer protocol and review gate](../../setup/vision/DATASET1-REAL-TRANSFER.md).
No reviewed real accuracy is available yet; the original PDF/ZIP edition remains
a dated snapshot and does not contain these subsequent experiments.

**Real footage clarification:** Dataset2's `cap_seal` camera does show the real
final-fixture station. Its 1181 stills and five videos include335 stills from
Runs1–4. The missing item is reviewed per-slot defect labels/coordinate mapping,
not raw footage. See `setup/vision/DATASET1-PILOT.md` for the source-exact audit.

**Later experiment update:** the [Dataset 1 three-class synthetic pilot](../../setup/vision/DATASET1-PILOT.md)
has now completed after this presentation edition was prepared. It scores360/360
targets on12 held-out synthetic scenes for both RGB/reference classifiers.
The PDF/ZIP retain their original dated status; real Dataset1 transfer and full
integration are still unimplemented. Use the new pilot protocol/results when
updating presentation content rather than treating its earlier status as current.

October 22, 2026 development snapshot. This is an eight-page presentation brief
with editable text, speaker notes, an architecture figure and the actual
single-panel real-video demonstration. It documents completed experiments; it
does not promote a model or establish physical acceptance.

## Prepared artifacts

Local release directory:
`C:/Users/Austin/ai-club-tools/results/final-presentation-20261022/release-v1`

- **Synthetic-Training-and-Digital-Twin-Presentation.pdf** — start here.
- **Presentation-Brief.md** — editable presentation text and tables.
- **Presenter-Notes.md** — one speaking note per page, including likely questions.
- **architecture.svg / architecture.png** — actual tested composition; dashed SAM branch is exploratory.
- **synthetic-states.png** — four source-hash-checked training renders from one family.
- **temporal-pool-candidate.mp4** — single panel; 240 decoded output frames at 5 fps,
  48 seconds from six real development clips. This is frozen-model output, not
  a fresh live camera demonstration.
- **measured-comparison.json / liquid-model-provenance.json** — exact measured
  counts and frozen liquid-model identity, with no implementation source/weights.
- **completed.json** — source/evidence identities and artifact SHA256 values.

The portable, code-free sharing ZIP is the sibling `release-v1.zip`. Rendered
page previews are retained in the local directory and excluded from the ZIP.
The repository's editable document source is
`team_handoff/presentation/brief-20261022.json`.

## Key figures and how to describe them

| Figure | Safe interpretation |
| --- | --- |
| 144 train / 48 development | Synthetic images; 36/12 disjoint four-state families. |
| 48/48 presence; 0.8933 positive IoU | Synthetic development score used for model selection; geometric mask targets, not real optical truth. |
| 16.61 / 25.08 ms median / p95 | Matched cached-input direct CV processing on GB10; excludes decode, rendering, network and browser. |
| Approximately 6× faster | Direct temporal versus matched detector→SAM comparators; not the refined hybrid. |
| 105 → 185 / 264; 80 gains, 0 losses | QA-eligible raw-image samples on the same frames; not segmentation or recipe accuracy. |
| 72.62 ms median refinement | Incremental prompt/SAM/QA with cached upstream inference; full hybrid timing unavailable. |
| 0/24 normal; 72/72 two-mL; 45/72 half-mL | Separate process-simulation warning outcomes, with ideal process cues; not physical trial metrics. |

Six correlated clips are not 264 independent physical trials. Real development
imagery informed diagnosis and hypothesis formation, but the new liquid weights,
checkpoint and operating thresholds were fitted/selected using synthetic data.
Run 5 was excluded from this round. Human-reviewed real mask, presence and
first-visible timing accuracy are pending; no production accuracy is established.

## Training provenance and model identities

| Component | Identity | Domain fitting |
| --- | --- | --- |
| Vial detector | YOLO11n | Synthetic CSI imagery; generic COCO initialization. |
| Liquid model | Small GroupNorm U-Net 16/32/64/128, mask + presence | Random initialization; 144 synthetic images; epoch 40 selected from 60 by synthetic criteria. |
| Optional refinement | SAM2.1 Hiera Tiny | Generic pretrained model; no CSI fine-tuning. |
| Legacy R-CNN comparator | Faster R-CNN ResNet50-FPN | Real CSI Runs 1–3; selected on Run 4. Explicitly a legacy comparator. |
| Synthetic R-CNN comparator | Faster R-CNN ResNet50-FPN | Synthetic CSI imagery; generic COCO initialization, retained fixed epoch-3 control. |

SHA256 identities:

```text
liquid  6714fe32191d8eba230aa87246f38a25c2906c619d60d7ba30c9e60205352eef
vial    0bfac07a256eaf190416b632ad1f094a6b534309df1bb11a8c175e9d811b56bc
SAM2.1  7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69
```

## Dataset 1 status

**Not integrated, trained or evaluated.** We located the tray-inspection twin
and camera configuration and wrote the second-station design. Dataset 1 real
labeled fixture crops are absent from the known local inputs. We can begin
synthetic rendering/training now; real examples are needed for independent real
validation. Cross-station physical vial identity also remains unimplemented.

The two stations would provide separate observations to the operator. Link fill
evidence to a final fixture slot only with confirmed physical association. An
unlinked observation remains useful; a matching color or sequential track number
does not establish that link. Existing sensor/actuator control is outside the
hackathon interface.

## Evidence in the repository

- `setup/vision/SYNTHETIC-ITERATIONS.md`: render/training/selection, negative results and stress checks.
- `setup/vision/TARGETED-SYNTHETIC-TRANSFER.md`: paired render construction and geometric versus optical target boundary.
- `setup/vision/TEMPORAL-POOL-CANDIDATE.md`: exact tested input/reference/model contract.
- `setup/vision/SAM-TEMPORAL-COMPARISON.md`: matched timing and actual SAM refinement.
- `setup/vision/STAGE2-REFINEMENT.md`, section “Combined candidate on fresh simulator seeds”: separate process simulation.
- `team_handoff/part2/OPERATOR-CLARIFICATION.md`: operator-only scope and organizer clarification.
- `team_handoff/integration/dataset1-extension-20261022.json`: proposed Dataset 1 integration; explicitly unimplemented.
- `setup/verification/sam-temporal-comparison-20261022.json`: completed comparison evidence.
- `annotation_review/VISIBLE-POOL-REVIEW.md`: pending human review workflow; drafts are not approved labels.

Full model-comparison visuals and the refinement video remain under:
`C:/Users/Austin/ai-club-tools/results/synthetic-iterations-20261022/`

```text
sam-temporal-summary-v1/matched-comparison.png
sam-temporal-summary-v1/early-later-mask-review.png
sam-temporal-summary-v1/temporal-sam-refinement-review.png
gb10-results/temporal-sam-refinement-v1/temporal-vs-sam-refinement.mp4
```

The presentation release was prepared locally; it has not been pushed to main
or published as a GitHub release. No new training or industrial connection was
needed to prepare it.
