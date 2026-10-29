# Matched SAM2 / temporal comparison — October 22

**October 24 decision:** retain the direct temporal U-Net as the leading
SAM-free integration candidate. The SAM refinement below remains optional; its
higher sampling availability does not establish better liquid boundaries. See
[the current architecture decision](CURRENT-ARCHITECTURE.md).

The temporal candidate is about **6× faster** in the matched cached-input test.
Guided SAM supplies more QA-eligible samples, but can still produce extensive
glass-shaped masks in the inspected early frames. A follow-up uses temporal
proposals to prompt SAM, without R-CNN, and raises qualified sample availability
from **105/264 to 185/264: 80 gains, zero losses**. These are execution and
availability results; no reviewed mask/presence/onset accuracy is established.

## Matched experiment

All arms use the same 264 exact native Run 4 RGB frames from six predeclared
clips, the same synthetic-trained YOLO11n vial model, same QA code/configuration
and same GB10. Every arm runs fresh vial inference and equality of those
proposals is checked. Full 264-frame warm-up, then two measured passes per arm;
arm order reverses in the second pass. SAM is invoked only with eligible prompts;
invoked-frame timing is retained separately so missing prompts cannot hide cost.

| System | Training provenance | CV median / p95 | Mask proposals / 264 | Color QA passes / 264 |
| --- | --- | ---: | ---: | ---: |
| Legacy R-CNN → guided SAM2.1 | R-CNN trained on real Runs 1–3, selected on Run 4; COCO/SAM generic pretraining | 99.24 / 101.09 ms | 195 | 185 |
| Synthetic R-CNN → guided SAM2.1 | Existing fixed epoch-3 GB10 synthetic-only R-CNN control, 614 images; COCO/SAM generic pretraining | 99.84 / 101.38 ms | 257 | 210 |
| Temporal direct | Frozen epoch-40 U-Net, 144 synthetic training images, random initialization | **16.61 / 25.08 ms** | 189 | 105 |

Each arm has 528 measured pipeline samples. Timings exclude decode, rendering,
serialization, transport and browser. They are not camera-to-dashboard latency.
The earlier isolated temporal benchmark was 15.47 ms; this matched mixed-model
trial is the appropriate source for the approximately 6× speed comparison.

The synthetic R-CNN checkpoint is the retained predeclared epoch-3 duration
control, not a model chosen from these clips. This is a practical system
comparison: fitting sets, generic pretraining and selection criteria differ.
The real-trained legacy arm is explicitly a development comparator, not the
synthetic-trained competition solution. No fitting or threshold tuning occurred.

## SAM policy and source integrity

Both SAM arms use the existing colored-pool box/point guidance and causal box
bridge, with unchanged thresholds and a state reset per clip. No SAM video mask
propagation or future-frame prompting occurs. The original October 20 executed
prompt implementation was replayed on the new predictions: **all 528 prompt and
bridge outputs match exactly**. The current source differs only by the previously
documented constant refactor. Prompt source frames are current/past only.

Source RGB hashes, video hash, native PTS/time base, model and QA hashes are
checked. All 792 source groups across warm-up/two passes have identical vial
proposals in all three arms. Within each arm the repeated passes reproduce raw
masks, final sampling masks, colors and QA outcomes exactly. Both comparison
videos fully decode to 240 frames at 5 fps (48 seconds).

The first run finished all inference and media, then its manifest writer tried
to hash `helpers/__pycache__`. That failed directory/log remains intact. A fresh
`sam-comparison-verified-v2` copy independently redecoded every input and output
video, recomputed timing/counts, checked model/helper hashes and verified repeated
outputs. No GPU inference was repeated to repair packaging. The source writer
now filters files for future runs; the original executed helper is preserved.

## Exploratory temporal → SAM refinement

The matched comparison motivated one additional, explicitly exploratory arm:

```text
same-frame cached YOLO vial + frozen temporal liquid proposal
  → unchanged pool box/point guidance and causal box bridge
  → actual SAM2.1 Hiera Tiny inference
  → unchanged sampling QA on raw source RGB
```

There is no R-CNN input to this branch. The temporal presence threshold and
proposal mask stay frozen; no proposal is created where the temporal branch
supplied none. References retain their unreviewed status. This does not repair
the temporal model's dependence on a suitable, aligned reference.

| Result | Temporal direct | Temporal-prompted SAM |
| --- | ---: | ---: |
| Source frames | 264 | 264 |
| Upstream temporal proposals | 189 | 189 |
| Actual SAM calls / refined masks | — | 188 |
| QA-eligible samples | 105 | **185** |
| Paired QA gains / losses | — | **80 / 0** |

The same 264-frame warm-up and two measured passes are used. Repeated raw/final
masks, colors and QA decisions match exactly. Incremental prompt/SAM/QA took
72.62 ms median / 73.55 ms p95 across all 528 measured frames, **excluding cached
vial and temporal inference**. Do not interpret that as full hybrid pipeline
latency or add medians to claim a measured full-pipeline result. The direct
temporal path remains the measured fast option; a complete live scheduling and
transport benchmark for the refinement composition has not been run.

## Decision and remaining work

Keep both capabilities available for integration testing: direct temporal output
for fast presence/localization, optional SAM refinement for sampling. The new
refinement experiment is promising; it is not automatically deployed or a proven
accuracy winner. Srijan owns the integrated pipeline and Josiah/Hayden own its
operator/backend implementation. Reference capture/reset/alignment still needs
validation. A changed sampling method also needs compatible, qualified color
references before recipe comparisons.

Higher QA availability does not establish correct fluid masks: the inspected
synthetic-R-CNN/SAM early overlays often cover much of the lower glass while the
temporal model is silent. This demonstrates why neither QA counts nor the
zero-tall-mask heuristic can rank segmentation accuracy. Those frames are not
human-approved empty examples. Shared visible-pool labels are the next gate.

The existing 72-frame review packet can score all three arms and the refinement
using one source-matched export; no duplicate labeling is needed. Use a raw-first
independent subset to reduce draft anchoring. Demo exports are refused and
missing masks on reviewed positive frames count as missed proposals. Masks are
scored before sampling erosion; sampling interiors intentionally have another
target scope. Reviewed real data is reserved for evaluation, never fitting.

```text
python -m annotation_review.score_sam_temporal_review --comparison PATH/gb10-results/sam-comparison-verified-v2 --source-probe PATH/gb10-results/mask-probe-v1 --refinement PATH/gb10-results/temporal-sam-refinement-v1 --packet PATH/visible-pool-review-v2 --review PATH/human-export.json --out PATH/fresh-reviewed-comparison
```

## Local evidence

Experiment root:
`C:/Users/Austin/ai-club-tools/results/synthetic-iterations-20261022`.

- `sam-temporal-summary-v1/README.md`, `comparison.json`, `matched-comparison.png`.
- `sam-temporal-summary-v1/early-later-mask-review.png`: all six clips, same early
  and later source frames for every model, with raw masks.
- `sam-temporal-summary-v1/temporal-sam-refinement-review.png`: direct/refined masks
  and the actual final sampling interior, all six clips.
- `gb10-results/sam-comparison-verified-v2/sam-vs-temporal.mp4`: matched three-arm video.
- `gb10-results/temporal-sam-refinement-v1/temporal-vs-sam-refinement.mp4`: actual
  temporal proposal and SAM refinement side by side.
- `gb10-results/sam-prompt-equivalence.json`: independent original-policy replay.

Verification: [sam-temporal-comparison-20261022.json](../verification/sam-temporal-comparison-20261022.json).
All new GPU work ran on GB10. Run 5, source labels, scene assets and application
defaults were not changed. No approved human labels or industrial connections
were created. All 24 annotation/scoring tests passed using CPU-only fixtures;
no test fixture is an eligible team annotation.
