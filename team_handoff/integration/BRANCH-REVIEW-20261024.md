# Mentee branch review and Tuesday demo priorities — October 24

Fetched `origin`, checked all 11 remote branch heads and reviewed six mentee
branches at the commits below. No open PRs were found; PR1 is the already merged
frontend/API starter. No branch was merged, pushed or checked out. The dirty
primary `austin-work` checkout and all mentee source remain intact.

The delivered pieces include a trained vial detector, an API/WebSocket camera
bridge, a browser UI, a SAM pilot and a new liquid detector training script.
**The pushed mentee branches do not yet contain a demonstrated end-to-end direct
temporal U-Net → measurement QA → operator application.** Unpushed local work is
outside this review. The original assignments/PDFs predate the current SAM-free
decision and synthetic-only fitting boundary; align the team on those changes
before asking them to continue the original detector/SAM training plan.

## Delivered work

| Branch / latest commit | What is present | Demo readiness |
| --- | --- | --- |
| `IntegrationDashboard` — `5ed6668`, Oct 22, Fonsh19 | FastAPI frame upload, per-camera WebSocket channels, same-app HTML dashboard, optional SAM wrapper, color-distance calculation, recipe context, short sequence history, reference CSVs | Useful transport/UI starting point. CPU health/homepage/ingest/WebSocket probes pass. No integrated detector or temporal U-Net; decision gating and mask display need work. |
| `alayna-yolo` — `406b544`, Oct 22, ostlera | YOLO11n training notebook, 50-epoch CSV, validation plots, `weights/best.pt` and `last.pt`, manual box correction tool | Actual fitted artifact and saved evaluation outputs. Notebook revalidation reports mAP50 0.995 / mAP50–95 0.92063 on 157 images. This is real-CSI development/provisional-label evidence, not a synthetic-trained competition model or verified physical accuracy. |
| `liquid-pool-model-v5` — `dca8cdf`, Oct 23, Srijan Kumar | Faster R-CNN ResNet50-FPN trainer and random-30-image prediction script | New scripts, not the selected temporal pipeline. No checkpoint, supplied input annotation file, evaluation metrics, mask output or integrated adapter is committed. |
| `team-3` — `5d15423`, Oct 22, Cameron Delpage | SAM2.1 Tiny four-frame runner, rough-mask evaluator and progress PDF | PDF reports five prompted masks on three Run4 frames plus one no-proposal frame. Useful optional branch/provenance work; no complete recording integration. Original result/mask exports remain outside Git. |
| `cameron/data_pipeline` — `53a6a2e`, Oct 19 | Isaac scene/rendering script, appearance/carryover helper, schema exporter and launch scripts | Supporting synthetic-data implementation. Referenced configuration is absent from the branch; launch paths expect a different folder layout. No new executed dataset verification is committed. |
| `hayden-work` — `c3535ea`, Oct 19, Fonsh19 | Manifest CSV and contact sheet | Earlier supporting material; the dashboard branch is the newer application delivery. |

No separate Josiah implementation branch or newer Zoe/Reagan NVIDIA-tool report
is visible among the fetched heads. This does not establish that no local work
exists. The dashboard branch predates the Oct23 `api/` / `client/` starter merge
and has not incorporated that layout.

## Findings that affect Tuesday

### P1: color decisions bypass sample QA and reference qualification

`IntegrationDashboard:outputs/backend/app/main.py:244` compares supplied RGB values
at a hard-coded ΔE2000 threshold of 2.3. `run_analysis` returns that decision at
line400 independently of its `sample_usability` result, mask availability,
completion, accepted-reference status or physical-vial identity. The UI at
`outputs/backend/web/index.html:64` highlights `within_threshold` as “IN RANGE.”

CPU reproduction against unchanged source: submit `sample_usable=false`, no
detections and identical observed/expected RGB. HTTP200 returns
`sample_usability.usable=false`, `status=no_liquid_proposal`, **and**
`color_inspection.status=within_threshold`. Supplying only a rectangular ROI and
matching RGB reference also returns a comparison without a liquid proposal.

For the integrated candidate, consume the existing measurement QA and independent
reference/completion/identity qualification. An unusable/unqualified sample must
show unavailable with reasons. Do not substitute this rectangular-ROI median and
2.3 threshold for the existing QA or a commissioned recipe acceptance rule.

### P1: the selected U-Net masks cannot enter the current backend contract

The API expects class1 boxes and calls its optional SAM engine. It has no typed
upstream liquid-mask/reference/model/QA observation input. A CPU fixture supplies
a valid mask field plus a class1 detection with SAM disabled: the field is ignored
and the result is `segmentation_unavailable`.

Connect the frozen direct U-Net through the
[current architecture and evidence contract](../../setup/vision/CURRENT-ARCHITECTURE.md).
Carry raw proposal, presence, exact source/reference identity, geometry and actual
measurement QA separately. Do not run U-Net → SAM just to satisfy the old API.

### P1: browser camera mode supplies no detections and draws no mask overlay

`outputs/backend/web/index.html:63` always posts `detections_json=[]`; the service
does not run a vial/liquid detector itself. `onFrameEvent` displays the received
image and color metadata, but never draws `combined_mask_png_base64` or individual
proposal masks. Starting this UI alone cannot produce the proposed CV demo.

Add the actual inference producer and source-aligned raw-mask overlay. Show
presence/proposal, sampling QA, qualified recipe comparison and source age as
distinct states. The current DOM event list is not a retained operator review,
acknowledgement or response workflow.

### P2: frame preprocessing breaks the existing fixed camera profile

The browser capture horizontally mirrors pixels, can resize them to width960,
posts at one frame per second, and uses browser elapsed time as `pts_seconds`.
The tested U-Net expects native640×480 pixels with a fixed112×192 ROI and an
aligned reference. These inputs are not interchangeable without an explicit,
validated adapter. Reset does not automatically follow browser source/context
changes. Preserve native recording clocks and geometry for file replay; label
browser capture timestamps honestly rather than calling them native video PTS.

### P2: Srijan's trainer is not reproducible from the supplied handoff

`train_v5.py:63` requires `starter/coco/likely_good.json`, which is absent from
the branch and the local published v1 package. The supplied exports use names
such as `legacy-liquid-train-originals.json` and explicitly retain provisional
status. The loader at lines23–31 drops every image without annotations: a CPU
fixture containing one positive and one annotation-free image retains only the
positive. It also maps every supplied box to liquid without filtering category.
Mixed-category inputs would therefore need explicit validation/filtering.

The script has no separate development evaluation, synthetic provenance checks
or checkpoint selection; prediction draws the highest scoring box, not a mask.
For Tuesday, Srijan's priority should be the already available synthetic-trained
U-Net adapter. Keep this R-CNN experiment as a comparator; do not infer completed
training, compliance or accuracy from the script name. Real starter imagery must
not enter new CSI-domain fitting.

### P2: YOLO packaging and provenance need a small handoff

Alayna's notebook trains on1442 images and validates on157, with original CSI
filenames; the split matches the real starter. The recorded metrics are useful
development results. The committed `yolo.yaml` defines class0 vial, while notebook
calls refer to missing `data.yaml` and `.../yolo_weights/best.pt`; the actual
committed trained artifact is `.../weights/best.pt`. Supply a working relative-path
inference entry point, checkpoint identity, class map and dataset provenance.

The manual editor overwrites original label TXT files and skips images with
anything other than one box. Use copied/versioned corrections with reviewer and
source provenance; do not overwrite the released labels or treat edits as approved
without explicit human decisions. For the synthetic-trained competition path,
the existing frozen synthetic YOLO is already available; Alayna's real-trained
checkpoint remains a separately labeled comparator.

### Lower-priority input handling and older guidance

Malformed RGB types currently produce HTTP500 instead of a client-validation
response: metadata construction is outside the endpoint's parsing exception
handler. SAM configuration errors are hidden behind an unconfigured engine.
These should become explicit diagnostics. Cameron's progress PDF still recommends
real Runs1–3 training; that dated recommendation is superseded by the current
synthetic-only boundary. Its rough IoU0.599 versus0.085 compares prompts on tiny,
manually prepared draft samples and must not become a general mask-accuracy claim.

## Recommended work allocation before Tuesday, October 27

| Owner | Concrete deliverable |
| --- | --- |
| Srijan | One frozen YOLO + direct U-Net inference adapter. Same source pixels/clock in image, proposals and QA; qualified reference capture/reset; explicit unavailable output. No SAM dependency in this path. |
| Josiah / Hayden | Wire that observation into one chosen dashboard/API surface. Draw masks at source coordinates; gate color/recipe status; show source age and QA reasons; retain a source-pinned operator acknowledgement/response. Avoid spending Monday on a framework rewrite. |
| Alayna / Zoe / Reagan | Package YOLO artifact/provenance and compare its role with the synthetic baseline. Complete independent early-pool/empty/stream-only review for evaluation; preserve the synthetic-only fitting boundary. Keep new NVIDIA experiments off the Tuesday critical path. |
| Cameron | Help verify source/mask coordinates and reviewed pool-versus-stream definitions. Keep SAM as an optional side-by-side comparator and supply the original rough-mask/result exports separately from approved labels. |
| Austin / CSI | Confirm supervised feed/test arrangement, known empty reference procedure, intended recipe/completion context and an independent acceptable recipe reference where available. No booking or industrial connection is implied by this review. |

**Sunday:** agree the observation contract and model hashes, then make a saved
recording drive the actual inference producer and mentee dashboard together.
**Monday:** rehearse empty/early/established fills, bad reference, camera/source
reset and dropped/late frames; fix blockers and record a fallback video.
**Tuesday:** show the same operator loop on an approved supervised feed if
available. Preserve file replay as an explicitly labeled fallback.

The minimum useful demonstration is frame → actual mask/presence → qualified
sample or unavailable → operator review/response. A recipe-quality claim needs
independent recipe/reference/completion/identity evidence; if those inputs are
missing, demonstrate the unavailable state rather than a fabricated pass.
Dataset1 remains an optional separately qualified extension to this core demo.

## What was actually checked

- Fresh `git fetch origin`, full remote-head comparison, commit authors/dates,
  changed-file inventory and connected GitHub PR search.
- Read the six branches' changed code/docs and Alayna's saved notebook/CSV.
- CPU in-process FastAPI: health, homepage, frame upload and WebSocket frame
  delivery work; QA bypass, missing upstream-mask input and malformed-RGB failure
  reproduced with artificial PNG fixtures. No port/server/browser/camera opened.
- Positive/annotation-free constructor fixture confirms the R-CNN loader's
  exclusion. All10 changed Python modules compile; dashboard JavaScript syntax
  passes `node --check`.
- No new fitting, GPU use, YOLO/SAM inference, Isaac render, physical equipment
  test, branch merge or push. Model metrics are committed reports, not fresh
  inference/evaluation results from this review.

Evidence, exact source snapshots and checks:
`C:/Users/Austin/ai-club-tools/results/team-branch-review-20261024-v1/`.
Start with `inventory.json`, `backend-runtime-checks.json`,
`yolo-recorded-metrics.json` and `review-verification.json`.
