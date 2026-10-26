# Srijan pipeline review — October 26, 2026

Reviewed freshly fetched `origin/three-model-integration` at
`d5a143c9b0a30376bfb1ad39c885c86d9d725906` (Srijan Kumar, October 26).
The new scripts connect the original YOLO vial detector → Faster R-CNN liquid
box → SAM2.1 Tiny mask architecture for single images and image directories.
The batch runner loads models once and exports overlays, binary masks and a
per-image CSV with explicit proposal/mask statuses. This is concrete integration
progress. It is not yet a video worker, QA-qualified measurement or operator app.

No mentee branch was changed, checked out or merged. No push, training, GPU
inference, server or industrial connection was performed. Source snapshots and
CPU evidence are retained outside the repository at
`C:/Users/Austin/ai-club-tools/results/srijan-pipeline-review-20261026-v1`.

## Findings

### P1: YOLO receives RGB NumPy input through a BGR interface

`integration/batch_integration.py:81–92` converts PIL to RGB NumPy and passes
`source=rgb`; `integration/integration.py:66–79` does the same. Installed
Ultralytics 8.4.182 treats a three-channel NumPy image as BGR. Its later model
preprocessing reverses channels, so this route swaps red and blue relative to
the intended image. A CPU probe of the actual library loader confirms the
difference using a red pixel. No weights or model inference were used.

Pass the PIL image directly to YOLO, or convert only its input to a contiguous
BGR array. Keep RGB for Faster R-CNN, SAM and downstream color measurement.
The size of any detector-quality effect has not been measured here; Srijan's
installed package/version also needs pinning.

### P1: a previous binary mask survives a new no-proposal result

`integration/integration.py:141` writes a fixed `sam_mask.png` only after a
successful match, while `result.jpg` is overwritten at line157 on every run.
A positive fixture followed by a no-proposal fixture leaves the previous mask
unchanged next to the new frame's result. A downstream consumer reading that
filename could apply stale evidence to a different image.

The batch runner similarly reuses `integration/batch_results` and writes masks
only on success, using image stems as identities (`batch_integration.py:60–61`,
`168–173`, `196–224`). Repeated runs or same-stem inputs can leave/collide outputs.
Use a fresh run directory and source-bound records that explicitly state mask
availability; downstream consumers must honor the current record. Preserve old
evidence rather than overwriting it.

### P2: matching any vial does not identify the actively filling vial

`batch_integration.py:139–142` and `integration.py:115–118` accept a liquid box
when its center falls inside any detected vial. There is no active-vial input,
fill-position selection or ambiguity handling. A two-vial CPU fixture accepts
a liquid proposal on the neighboring vial even when the fixture's designated
filling position is the other one. This establishes controller behavior, not a
measured wrong-vial rate on real recordings.

Associate proposals with the explicitly selected filling vial/camera profile,
retain that identity in the output, and mark ambiguous association unavailable.
Center containment alone must not replace measurement QA or mask containment.

### P2: first-candidate selection suppresses usable later proposals

`batch_integration.py:122–145` and `integration.py:105–118` take candidate0 before
validity/association checks. If it does not match a vial, the runner stops even
when a lower-scoring proposal matches. A CPU fixture with two valid in-frame
boxes reproduces `unmatched_liquid_box`, zero SAM calls, and an ignored matching
second candidate. A separate invalid-first-box fixture reproduces the same
failure to fall back.

Filter and associate all eligible proposals before ranking. Keep explicit
ambiguity handling; selecting the next candidate alone does not solve vial
identity. Do not tune thresholds from Run5.

### P2: the asset/runtime handoff is not reproducible from this commit

Both loaders require `~/fluid-mixing-cnn/integration/models/yolo_best.pt`,
`~/fluid-mixing-cnn/liquid_pool_model_v5_10epochs.pth` and
`~/fluid-mixing-pipeline/checkpoints/sam2.1_hiera_tiny.pt`. Those two checkout roots
are hardcoded. The commit adds no model manifest, dependency lock, launch
instructions or executed result bundle. `test_models.py` is a model-load smoke
script, not evidence of a passed inference or deployment test.

Supply configurable input/model/output/device paths, dependency versions,
checkpoint hashes, class mapping and training provenance. Keep large assets in
the external artifact handoff. This review could not read the GB10 checkpoints
or results: `spark8.ai.msoe.edu` did not resolve. Missing committed weights do
not establish that no training or successful local inference occurred.

## Align the architecture before further integration

This commit implements the earlier team assignment. The current local decision
in [CURRENT-ARCHITECTURE.md](../../setup/vision/CURRENT-ARCHITECTURE.md) instead
uses YOLO geometry plus current/reference RGB differences → direct temporal
U-Net mask/presence → unchanged measurement QA. SAM2.1 and Faster R-CNN remain
comparators/optional paths. The new pipeline has no U-Net/reference lifecycle,
measurement QA, qualified recipe decision, video/native-clock observation
contract or app adapter.

Our updated architecture and runtime assets remain local/unpublished; do not
treat Srijan's implementation of the original assignment as failure to follow a
delivered replacement handoff. Share the actual frozen U-Net assets and input/
output contract before asking him to switch the liquid branch. Reuse his model
loading/orchestration and per-image status work. Do not train a replacement or
force U-Net masks through SAM merely to satisfy the earlier interface.

For Tuesday, Srijan and Josiah/Hayden need one saved-source rehearsal:

1. Load the selected, hash-pinned assets once and preserve source geometry/clock.
2. Produce current-frame masks/presence plus vial and reference identities.
3. Pass those observations through the existing measurement QA.
4. Display source-aligned masks and separate proposal, QA and qualified-recipe
   states; unusable/unqualified evidence stays unavailable.
5. Demonstrate positive, no-proposal, neighboring-vial, stale-source and reset
   cases, retain outputs, and record a fallback clip.

New CSI-domain fitting/selection remains synthetic-only. This review did not
verify the R-CNN or YOLO checkpoint training provenance, human-reviewed accuracy,
early-pool onset, segmentation quality or latency.

## Adjacent dashboard update

Fresh `origin/IntegrationDashboard` at
`3894e9f7f3fcb4d2cf032a4f1755cfdb992d6f2a` includes the API/client starter and a
new React camera prototype under `protoype-Hayden/client/`. Its
`src/main.tsx:3` imports `./pages/OperatorDashboard`, but that module is absent
from the pushed tree. TypeScript 5.9.3 reproduces TS2307 for this import; this
was a targeted compiler check, not a full frontend build. Ask Josiah/Hayden to
push the missing page before the shared rehearsal. It is a separate blocker
from Srijan's model pipeline.

## Verification boundary

All three new Python sources compile. CPU controller probes execute the
unchanged `main`/`valid_box` function ASTs with model doubles and real NumPy,
Pillow and filesystem writes: one positive control plus four edge cases.
The actual installed Ultralytics loader was exercised without model weights.
No model inference, real accuracy measurement or GPU benchmark was run. Evidence
and hashes are indexed in
[the verification record](../../setup/verification/srijan-pipeline-review-20261026.json).
