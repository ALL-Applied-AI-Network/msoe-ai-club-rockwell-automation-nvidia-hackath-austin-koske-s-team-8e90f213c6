# Consolidated handoff — October 21, 2026

## Current state and next actions

**October26 publication:** Austin authorized the scoped temporal CNN and current
architecture handoff on `main`. Start at
[temporal model release](team_handoff/temporal_cnn/README.md): frozen model/assets,
exact synthetic training inputs, current diagram, CPU example and source/QA
contract. Other experimental/application work remains on `austin-work`. This
publication adds no training, live deployment or real-accuracy claim.

**October26 Srijan pipeline review:**
[fresh push and reproducible findings](team_handoff/integration/SRIJAN-PIPELINE-REVIEW-20261026.md).
`three-model-integration` at `d5a143c` connects the original YOLO→R-CNN→SAM2.1
image/batch path. CPU probes confirm RGB/BGR mismatch, stale mask retention,
any-vial association and first-candidate rejection hiding a matching proposal.
Three sources compile; no actual inference/accuracy/latency claim. GB10 hostname
did not resolve. Align the original assignment with the locally selected U-Net
assets/contract, then rehearse inference→unchanged-QA→operator. Josiah's refreshed
`IntegrationDashboard` at `3894e9f` adds a React prototype whose imported
`OperatorDashboard` page is absent; targeted TypeScript check confirms TS2307.
Evidence: `results/srijan-pipeline-review-20261026-v1`. No source-branch edits,
merge, push, new worktree, GPU job or industrial connection.

**October24 Tuesday visit guide ready:**
[October27 guide](team_handoff/integration/CSI-VISIT-20261027.md) covers owners,
camera/source→inference→operator tie-ins, workstation/preflight, a90-minute run
sheet, native-clock metrics, failure cases and evidence/backup. Six-page PDF and
10 CSV field sheets:`C:/Users/Austin/ai-club-tools/results/csi-tuesday-20261027/guide-v3`;
portable`guide-v3.zip`. All PDF text bounds,14 artifact hashes, blank decision
fields and ZIP integrity verified. Earlier v1/v2 renders are preserved.
Austin confirmed no time/assigned host/feed arrangement yet. Published compute
source lists Windows11 /RTX PRO5000 Blackwell atCSI; RTX PRO6000 advice concerns
Brev. Verify actual unit onarrival. No CSI GPU benchmark, runtime installation,
booking or camera connection was performed. The packet contains guide/forms,
not runtime or weights; integrated U-Net start/stop package is Monday's missing
deliverable. Source documents were refreshed through the GitHub connector.


**October24 presentation/onboarding direction:**
[lessons, measured claims and five-minute outline](team_handoff/presentation/LESSONS-AND-CLAIMS-20261024.md)
now capture why promising models require camera/reference qualification, retained
measurement QA and an operator workflow on new data. The
[AI-guided onboarding design](team_handoff/integration/AI-NATIVE-ONBOARDING.md)
adds one reusable project skill at`.agents/skills/commission-vision-station/`
and a read-only`setup/vision/onboarding_readiness.py` asset/hash checker.
Skill-format validation and four CPU tests pass. On the actual U-Net assets,
four supplied files verify; temporal-reference, independent-real-evaluation and
live-rehearsal inputs remain unsupplied. All-file success still cannot authorize
deployment or claim real accuracy. Outputs:`results/ai-onboarding-20261024-v1/`,
current`readiness-v2`. No autonomous agent full-workflow or physical deployment
was demonstrated; no training/GPU/main change. Next scope is wrapping the same
mentee inference/dashboard interfaces, not adding a parallel runtime application.


**October24 mentee branch review:** [delivered work, reproduced blockers and
Tuesday task allocation](team_handoff/integration/BRANCH-REVIEW-20261024.md).
Fresh origin fetch/all11 remote heads and six mentee branches reviewed; no open
PRs found. Dashboard ingest/WebSocket works in CPU fixtures but color comparison
bypasses unusable-sample status, the UI sends no detections/renders no masks, and
no direct U-Net adapter is connected. Alayna supplies real-trained YOLO weights
and recorded99.5% AP50 /92.06% AP50–95 on157 provisional development images;
retain as comparator. Srijan's new Oct23 R-CNN branch has scripts only, missing
input/checkpoint/evaluation, and is not the selected U-Net pipeline. Cameron's
SAM pilot remains optional. No merge, push, new worktree or GPU job was started.
Evidence:`results/team-branch-review-20261024-v1`; Tuesday priority is one actual
U-Net→unchanged-QA→operator loop, independent reference qualification and a
recorded fallback, not more model-family experiments.


**October 24 workspace cleanup:** this repository now has one registered worktree,
the primary `austin-work` checkout. The clean app-starter checkout was retired
using ordinary `git worktree remove`; its tracked tree matched published
`origin/main`, and branch `team/app-starter` at `edd4fb6` remains available.
Ten ignored dependency/cache directories were moved intact to
`C:/Users/Austin/ai-club-tools/recovery/workspace-cleanup-20261024-v1/generated/`.
That recovery folder also contains the before/after inventory and restoration
instructions. No datasets, results, reviewer workspaces, pending primary edits,
branches or main refs were deleted/changed. Other projects were not cleaned.


**October 24 architecture decision:** [direct temporal U-Net is the leading
SAM-free integration candidate](setup/vision/CURRENT-ARCHITECTURE.md). The liquid
branch uses current/reference RGB differences, U-Net mask/presence and unchanged
measurement QA. YOLO supplies vial geometry. SAM2.1 refinement remains optional;
DINOv3 and earlier R-CNN/SAM paths are comparators. Preserve existing app defaults
and frame/reference contracts; Srijan owns live integration. No new training or
physical-validation claim follows from this decision.

**October24 SAM-free DINOv3 pilot complete:**
[architecture, matched metrics and visual failure analysis](setup/vision/DINOV3-DIRECT-MASK.md).
Two new mask/presence heads use the frozen generic DINOv3 ConvNeXt Small backbone,
with144 synthetic training images and48 development images; both select epoch50.
Paired DINO supplies180/264 unchanged-QA samples at29.43/31.63ms median/p95 versus
U-Net105/264 at16.37/24.86ms. RGB-only supplies172/264 but proposes on every frame.
These are sampling availability and compute, not reviewed segmentation accuracy.

Visual inspection finds paired DINO extending into glass/stream in some fills;
U-Net looks tighter around several pools. All77 paired QA gains coincide with
U-Net containment rejection, and76 already had U-Net mask proposals. The90% box
containment gate can favor taller masks without proving correct fluid pixels.
Do not present this as77 newly detected liquid frames or a validated replacement.
Both raw and final sampling masks are retained for human review. Keep the fast
U-Net baseline and paired DINO candidate; source-matched pool/onset review and
synthetic boundary/stream/glass coverage are the next useful gates.

Local results:`C:/Users/Austin/ai-club-tools/results/dinov3-direct-mask-20261024`.
`summary-v1` has48-second three-panel and single paired-DINO videos, each fully
verified240 frames, plus a12-source-frame comparison board. Both measured passes
reproduce masks, presence and QA exactly;792 vial comparisons match.280 remote
artifact hashes,264 source RGB hashes and792 reference joins were checked.
Two fits and matched inference finished onGB10; it is idle. Local CPU smoke
checks shape/freeze/exact checkpoint reload; all8 helper scripts compile.
Preserve frozen helpers/results. No Run5, approved labels, QA/default changes,
main push or industrial connections. No new job is pending for this pilot.


**October23 cap-demo correction after Austin's visual feedback:**
[audit and current app behavior](replay/operator/FULL-APP-DEMO.md#cap-display--metric-correction).
The first demo misrepresented fixed fixture-hole coordinates as cap-top markers.
273/646 present-cap marker centers fall outside exact-CAD projected cap bounds;
displacement reaches28.71px. The app now shows the selected96px slot input crop,
raw zoom and provisional class, without colored cap-location dots. CAD is used
only in the separate audit, not to fix predictions. All720 predictions and source
images are unchanged. Actual fresh cap-class results are nominal608/608,
missing36/38, crooked31/38, empty36/36:98.75% overall hides81.58% crooked recall.
UI warning gating misses one crooked target classified empty; earlier score-only
0/38 warning misses did not include that display suppression. Correct this
distinction in presentation claims; no model accuracy gain is claimed here.
Interactive8870 now uses`bundle-v3` /`operator-workspace-v2`, with the old
workspace preserved. Refresh the page. Audit board, nine genuine errors and
updated full-app recording are under`results/cap-overlay-audit-20261023`.

**October23 full operator-app replay demo complete:**
[full pipeline and run instructions](replay/operator/FULL-APP-DEMO.md).
A separate clean interactive session runs at`http://127.0.0.1:8870` (Windows
loopback HTTP200 verified). Actual47.6-second browser/API recording:
`C:/Users/Austin/ai-club-tools/results/full-app-demo-20261023/First-Vial-Full-App-Demo.mp4`.
The opt-in app shows two independent camera/evidence streams, temporal/SAM branch
switch, source/mask views, clickable cap targets, operator recipe context, pinned
review, acknowledgement/response/closure, append-only journal and JSON export.
Its pipeline tab renders offline synthetic training and both model paths.

Inputs reuse72 exact real Run4 filling frames and12 synthetic cap states with
actual completed model outputs. Direct/SAM subset QA availability is27/72 and
53/72; these are not accuracy figures. The UI/API workflow runs now; GPU inference
is cached. Recipe/product acceptance stays unavailable, body condition unassessed,
physical vial links absent and hardware commands zero. Existing app defaults,
measurement QA and mentees'`api/`/`client/` are unchanged; no main change or push.
Four backend tests and actual browser controls/export/reload/mobile checks pass.
The entire MP4 decodes1,190 frames at25fps. Sources/recording/verification live
under`results/full-app-demo-20261023`, with completed`bundle-v2` and`browser-v3`.
Earlier partial preparation/recordings are preserved. Recording-only servers
8871/8872 are stopped; interactive8870 remains running with no demonstration
recording actions in its clean workspace. No new training/inference was needed.

**October23 overnight fixture investigation complete:**
[full design, results and limitations](setup/vision/DATASET1-FIVECLASS.md).
Eight synthetic-only fits on GB10 completed: three five-class baselines, two
detail/reference arms, two normal-current coverage arms and a dedicated cap-state
model. Normal-current training improves matched fresh five-class exact accuracy
78.33%→93.82% and macro recall84.43%→89.35%; at separately frozen95%-pooled-defect
rules, normal false warnings fall867/1,140→90/1,140 while damaged misses rise
8/76→15/76. Dense classification worsens; sparse/normal behavior improves.
Do not turn that aggregate gain into a full five-class success claim.

Dedicated cap state scores99.38% on1,440 synthetic targets, with body condition
explicitly unassessed. A separate six-family post-freeze confirmation scores
98.75% on720 targets: the frozen99%-development-cap-recall warning rule misses
0/38 missing and0/38 crooked caps, with4/608 nominal-cap false warnings. Label-free
scene gain correction preserves this under reference brightness0.83/1.17;
independent digital-noise/JPEG90 cases score98.47–98.75%. These remain correlated
synthetic tests, not real camera accuracy. One-pixel reference shift produces
many false warnings; automatic registration-validity detection is not established.
Source CAD also exposes lying S vials at80.88–83.79° versus roughly0–30° generated
poses. Body-damage changes affect only1–42 central-crop pixels above the chosen
digital threshold, versus939–1,229 for cap controls. Full body damage and original
S-pose coverage are unresolved; no experiment was promoted into app defaults.

Verified local results: `results/dataset1-followups-20261023/summary-v1`.
Start with `brief-v2/Digital-Twin-Inspection-Results.pdf`, `report.json` and
`cap-preview-v1/cap-state-predictions.mp4`. The video shows24 static synthetic
states/576 fully decoded frames and actual colored cap proposals, one panel.
All34,560 follow-up parent-source/label/prediction joins and reported metrics
were recomputed. All GPU jobs have exited; GB10 is idle. Inspect markers before
any follow-up; do not repeat fitting or these evaluations.

The organizer participant package is manual, not a trained autolabeler. All79
package hashes match. Its two native1920×1080 examples now have120 unresolved
manual-review targets in `results/dataset1-fiveclass-20261023/native-review-v1`.
Their run identity is unspecified and they never enter fitting/selection. Our
separate640×480 packet has960 unreviewed targets/192 priority cases in `review-v1`;
native coordinates must not be silently scaled into it. Browser round-trip and
source-integrity checks pass; no actual human review/real accuracy was produced.
960 frozen five-class observations export through the existing candidate evidence
contract with null product decisions and unapproved geometry/visibility.

Next useful work: independently review real targets, qualify real golden/reset
handling and registration, reconstruct S coverage from authored lying geometry,
and validate optically visible body defects. Srijan owns live adapters; Josiah/
Hayden own app integration. The cap-only output must never be renamed whole-vial
`pass`. Research remains on `austin-work`; no research was pushed to main.
Separately authorized app-starter PR1 was merged into main as`1ec7cbc`, under
root`api/` and`client/`, without database/Terraform or a`team_app/` directory.

**October 23 two-station engineering rehearsal complete:**
[contract, results and team integration notes](replay/operator/COMMISSIONING.md).
Opt-in common evidence adapter ingested264 filling frames and480 fixture targets
from completed frozen outputs. It retains189 mask proposals/105 unchanged-QA
color samples and41 capped-looking/439 empty fixture proposals while reference,
recipe and physical-vial qualifications remain unavailable. Nineteen synthetic
reference/identity guard cases passed, including demonstration refusal and reset
on context/profile/camera/discontinuity/rewind. The757-event demonstration journal
retains exact source metadata for two review/response flows; no actual operator
action, quality release or equipment command occurred. Grid diagnostic flags0/16
sampled stills and both artificial8px shifts; mapping remains unapproved.
Thirteen focused CPU tests pass. One-command real scoring is ready but reports
null accuracy until actual human exports arrive. All GPU inference was reused.
Local root: `results/two-station-rehearsal-20261023`, with current `rehearsal-v2`,
`geometry-v1`, `evaluation-v1` and three-page PDF/worksheets in `handoff-v1`.
Exact portable source coverage:72 filling PNGs +16 fixture JPEGs (552/744 target
observations). Strict video re-extraction detected an unresolved local/frozen RGB
byte difference and refused substitution; `sources-v1` failure retained,
`sources-v2` explicitly lists192 unavailable filling images. No live integration,
real accuracy, same-vial association, reference approval, app/default/main change
or push is claimed. Next: actual human reviews, physical reference/context
commissioning, Srijan's live model adapters and Josiah/Hayden's operator integration.

**October 22 frozen Dataset1 RGB model on real footage complete:**
[protocol and media](setup/vision/DATASET1-REAL-TRANSFER.md).
Sixteen deterministic cap_seal stills from Runs1–4 yield480 G slot-frame proposals:
41 capped-looking,439 empty,zero uncapped. These are counts, not accuracy or distinct
vials; no reviewed truth/rare-defect recall is established. The exact synthetic
epoch5 checkpoint is unchanged; all GPU inference ran on GB10. A geometric Run4
anchor fit has0.659px RMS/1.414px max aperture residual, but the slot mapping and
occlusion remain human-unreviewed. Other-run overlays and30 source crops inspected.
No qualified capped real golden exists, so the paired model is explicitly unavailable.
Source-verified single-panel MP4/PNG and observations: `results/dataset1-real-transfer-20261022/summary-v1`.
The MP4 presents16 sampled stills at1/s, not native/live playback. Stills are not
joined to decoded video clocks or filling-vial identities. [Offline review](annotation_review/SLOT-REVIEW.md)
has96 priority/480 total targets; initial labels and drafts stay unreviewed.
Actual headless demo save/download/reopen and conflict checks passed; demo export
is rejected by scoring. Ten new focused integrity tests pass. No training, real
threshold selection, Run5 pixels, app/default/main change or push. Next: explicit
human review to measure real per-class/scene performance and uncertainty coverage.

**October 22 real final-station imagery confirmed:** Dataset2's `cap_seal` is the
overhead robot/two-fixture camera required for Dataset1-style transfer inspection.
Actual Run2/4 stills match the supplied tray twin layout; the twin README documents
CapSealCam's real clip calibration. The folder has1181 stills/5 videos, including
335 indexed stills from Runs1–4. Missing: reviewed per-slot defect labels and a
validated real target mapping, not raw station imagery. Run5 pixels excluded.
No inference/fitting was run in this audit. Source-exact real/twin PNG, four-still
contact sheet and audit JSON: `results/dataset1-cap-camera-audit-20261022`.
Next: validate slot crops and prepare evaluation-only human review, then probe
the already-frozen synthetic models. No reviewed golden/identity link is assumed.

**October 22 Dataset 1 synthetic pilot complete:**
[protocol/results](setup/vision/DATASET1-PILOT.md).
GB10 rendered60 scene families with matched all-capped golden/current RGB, then
fit two random-initialized classifiers for capped/uncapped/empty positions in the
G fixture. Whole-scene split:36 train/12 development/12 test (1080/360/360 crops).
RGB epoch5 and paired-reference epoch7 were selected by development macro recall
from30-epoch fits before test. Both score360/360 synthetic targets, including120/120
per class, with12/12 all30-correct scenes. No paired advantage, real Dataset1 accuracy,
crooked/damaged-class coverage or physical-vial association is established.
Local evidence: `results/dataset1-pilot-20261022/gb10-results-v1`; source-exact
single-panel demo/PNG/metrics/30 unlinked observations in `summary-v1`.
All120 rendered RGB hashes and720 test-prediction source/label/metric joins verified.
GPU is free, fits complete, no local GPU/app/default/main/push changes.
Initial Kit launch stalled before images; original failure/log retained, direct
render recovery succeeded under the same frozen plan, and training ran once.
The executed helper snapshots remain intact; repository helper now resolves paths
and isolates experiment arguments from Kit. Four critical CPU integrity checks pass.
Next gate: validated crops/review labels from the available real cap_seal footage
and approved reference handling, then transfer scoring; full cross-station
integration remains planned.

**October 22 presentation documentation prepared:**
[presentation brief and evidence guide](team_handoff/presentation/README.md).
Eight-page PDF, editable text, presenter notes, actual training-state images,
architecture figure and single-panel demo are bundled locally under
`results/final-presentation-20261022/release-v1` (sibling code-free ZIP).
All pages rendered/inspected; counts tied to the completed matched comparison.
That dated presentation edition records Dataset 1's earlier proposal-only state;
the subsequent three-class synthetic pilot above now supersedes that status.
Real evaluation and physical vial-to-slot linkage remain unimplemented.
No new experiment, app deployment or push was needed to prepare that brief.

**October 22 matched SAM/temporal comparison complete:**
[protocol and actual results](setup/vision/SAM-TEMPORAL-COMPARISON.md).
Same264 exact Run4 frames, fixed synthetic vial detector/QA, full-sequence warm-up
and two measured passes. Legacy/Synthetic R-CNN→guided SAM take 99.24/99.84 ms
median; direct temporal takes 16.61 ms (~6× faster). QA availability is 185/210/105
frames respectively, not mask accuracy. Actual temporal-proposal→SAM refinement
recovers 80 QA samples with zero losses (105→185/264), without R-CNN. Its 72.62 ms
timing is incremental with cached upstream inference, not full hybrid latency.
All raw/final masks, colors and QA repeat exactly. Independent replay matches the
original Oct20 prompt/bridge algorithm on528SAM inputs. A manifest-only cache
directory error was recovered into a fresh verified directory without rerunning
GPU inference; originals retained. Videos/PNG boards and tables are under
`results/synthetic-iterations-20261022/{gb10-results,sam-temporal-summary-v1}`.
The existing72-frame human packet can score every arm/refinement through the new
source-matched scorer. Human accuracy still pending; no deployment/push/main change.
All24 annotation/scoring tests pass. Candidate contract now documents optional
SAM refinement; Srijan remains integration owner.

**October 22 GB10 iteration complete:** [protocol, metrics and completion markers](setup/vision/SYNTHETIC-ITERATIONS.md).
Three synthetic-only direct-mask/presence fits and six-clip real inference are
complete. Temporal masks visually follow bottom-pool growth; raw QA availability
is 101/264 and 105/264 versus 0/264 for current-RGB U-Net. These are availability
counts, not human accuracy or a matched comparison with earlier 111-input probes.
The [72-frame offline mask review packet](annotation_review/VISIBLE-POOL-REVIEW.md)
is prepared and browser-verified; all supplied drafts remain unreviewed. Real
reviews stay evaluation-only. Thirty-six GB10 optical controls also completed,
without measured physical calibration or source asset changes. Both longer
detector fits and duration controls completed; longer fitting worsened provisional
real agreement. Frozen temporal stress checks and the actual vial→mask→QA benchmark
also completed: 15.47 ms median / 24.53 ms p95 over 528 cached-input samples, with
identical masks/colors/QA in both passes. Digital darkening ×0.70 gives 0.993
prediction IoU and 2/264 proposal flips; a later reference causes 126/264 flips.
Reference capture/reset/alignment is the next implementation dependency, alongside
human review. No camera-to-dashboard latency or reviewed accuracy claim.
Verified reports: `results/synthetic-iterations-20261022/gb10-results/candidate-checks-v1`;
plots/tables: `candidate-summary-v1` under the same experiment root.
[Candidate integration contract](setup/vision/TEMPORAL-POOL-CANDIDATE.md).
All GPU work is finished; inspect markers rather than rerunning jobs.
No local GPU work, default promotion, main change or push occurred.

**October 22 targeted transfer experiments complete:**
[protocol, results and next calibration work](setup/vision/TARGETED-SYNTHETIC-TRANSFER.md).
An isolated session-layer renderer produced 192 paired empty/stream-only/early/
established scenes, including 32 body boxes ≤32px high. New synthetic-only paired
and lower-vial crop fits both used 614 images/epoch, three fixed epochs and COCO
initialization. The paired fit improved new-render held-out geometric AP 42.47%→
74.40% and small-body matches 0/7→5/7, but real provisional AP worsened 26.50%→
23.18%; crop-fit AP was 8.44%. A synthetic-calibrated basal filter removed 177
unmatched boxes while losing 9 matches. Actual unchanged-QA real raw passes were
baseline 61/111, paired 20/111, crop 23/111, basal 59/111. No candidate is promoted.
Verified combined evidence:
`C:/Users/Austin/ai-club-tools/results/targeted-synthetic-20261022/summary-v2/`.
Side-by-side PNGs are in `review-v1`, `roi-review-v1`, `basal-v1`; the seven-case
`Human-Target-Scope-Review.zip` is a portable human reference, not approved labels
or an editor import. Original USDs/labels and app/QA defaults remain unchanged;
Run 5 excluded. Next: human target-scope/onset review and controlled supervised
empty/known-fill captures for optical calibration, rather than promoting these
failed transfer candidates. Verification: `setup/verification/targeted-synthetic-20261022.json`.

**October 22 real/render visual diagnosis:** five source-exact PNG boards plus
eight new predictions from the synthetic-only Faster R-CNN are in
`C:/Users/Austin/ai-club-tools/results/real-vs-synthetic-20261022/review-v1/`.
The selected real shallow-blue case has a bottom-pool proposal plus extra vial/
label proposals; the established blue case localizes visually in both domains.
The pipeline transfers partially. Real reference scope is also inconsistent:
the 20px box in source `vial_fill_20260922_171633_379025` is a band near the holder,
not evidence of a tiny first pool. Do not interpret small box height as reviewed
onset truth or the proxy AP gap as pure rendering failure. Pairing is illustrative,
not physically matched; no source labels were changed and no Run 5 input used.
See [the updated audit](setup/vision/SYNTHETIC-TRANSFER-AUDIT.md) and verification
`setup/verification/real-synthetic-visual-20261022.json`.

**October 22 synthetic-only Dataset 2 experiments complete:**
[results and interpretation](setup/vision/SYNTHETIC-TRANSFER-AUDIT.md#october-22-follow-up--synthetic-only-competition-direction).
Two new Faster R-CNN liquid candidates use 614 synthetic training / 154 synthetic
development scenes and fixed final epoch 3, with generic COCO initialization.
Raw/RGB-augmented synthetic AP50–95 is 95.79%/96.70%; provisional real Run 4
agreement AP is 26.50%/24.96%. The augmentation did not improve transfer; neither
model is promoted. The original synthetic-only YOLO, plus each new Faster R-CNN
with synthetic YOLO vial proposals and generic SAM2.1, completed actual inference
and unchanged QA. There are 5,698 scored inference inputs across completed probes
and post-fit scoring, not independent physical trials. No Run 5 input or real
CSI training was used for new candidates. Legacy real-trained models are comparators.
Verified combined artifacts:
`C:/Users/Austin/ai-club-tools/results/sim-real-robustness-20261022/summary-v1/`.
The generated-only team data ZIP is under `results/synthetic-liquid-20261022/`
as `Dataset-2-Synthetic-Pilot-v1.1.zip`; it retains geometric-label/optical limits.
No fresh 3D scenes or reviewed visible-pool labels were produced. The next data
fixture needs early-pool/optical/pose coverage and consistent target scope, followed
by reviewed real evaluation. User-approved termination of existing Isaac PID 91932
freed memory for the GPU runs; no industrial connections or app promotion occurred.

**October 22 meeting ownership update:** Josiah/Hayden own the operator
dashboard/frontend and backend; Srijan combines model modules into a standard
pipeline; Zoe/Alayna/Reagan investigate NVIDIA additions (Cosmos, etc.). Cameron
has pushed the SAM2.1 pilot on `team-3`. These explicit assignments supersede
earlier proposed workstreams; see [current team ownership](team_handoff/README.md#current-assignments--october-22).
Austin reported Brett's suggestion to combine Dataset 1 and demonstrate the
sim-to-real gap. Local experimentation may measure robustness and transfer
limitations; no independently reviewed real mask/defect accuracy is established.
Austin explicitly reaffirmed **synthetic-only new CSI-domain training**. Real
footage is for evaluation/inspection. Existing real-trained pipeline weights are
legacy comparators; do not describe that baseline as the competition's synthetic
training demonstration. Historical starter `train/` paths do not authorize real
fitting. Generic pretraining and real-informed camera/prompt development must be
disclosed separately from domain fine-tuning.

**October 22 GB10 benchmark complete:** [protocol and measured results](setup/vision/GB10-STATION-BENCHMARK.md).
Matched loopback HTTP medians were local 272 ms / GB10 130 ms; an isolated
persistent inference worker reduced them to 148 / 120 ms. Each arm used the same
111 cached inputs, full warm-up and 222 measured samples, with frozen models/QA.
The actual VPN/browser probe measured 556 ms median / 675 ms p95 to loaded sampled
overlay images. Live RGB-only rendering plus inference at the tested quality
managed 1.81 frames/s on GB10; prefer cached test input or an approved camera/file.
Source/QA/mask joins, 74 render RGB hashes and original asset hashes were checked;
the worker preserved masks and sample QA on all 222 same-machine pairs.
The candidate is not deployed: local app defaults remain unchanged. Benchmark
GPU processes stopped, shared lock released, SSH tunnel closed and local source
pause restored. Isolated GB10 environment/artifacts remain in
`~/ai-club-tools/experiments/gb10-station-benchmark-20261022`.
See [verification](setup/verification/gb10-station-benchmark-20261022.json).

**October 22 team meeting package:** [current architecture source](team_handoff/part2/MEETING-20261022.md).
The updated seven-page PDF and nine-file code-free ZIP are under
`results/part2-team-release-20261022/release-v2` and `release-v2.zip`;
the original `release-v1` remains intact.
They cover the actual continuous station, proposed A/B/C integration streams,
qualified physical endpoints versus simulated progress, evidence/limits, CSI tasks
and candidate GB10 deployment. All pages were rendered/inspected; metric claims
were checked against the existing reports. No code, models or data are packaged.
The GB10 was benchmarked after VPN reconnection; page 7 now includes those results.
No publication, application deployment or branch push was performed.
See [updated release verification](setup/verification/part2-team-release-v2-20261022.json);
the original [v1 verification](setup/verification/part2-team-release-20261022.json) is retained.

**Current interface: continuous inspection station.** Run
`pwsh -File setup/run-operator-twin.ps1` and open `http://127.0.0.1:8767`.
Frames arrive and run through the frozen CV/QA pipeline automatically. Operators
enter recipe/vial context and work on captured exceptions while the source
continues. Review acknowledgement/action/closure records do not command the source.
Manual review pins the clicked frame before notes are entered. Isaac is a
replaceable test producer; `-Source frames -SourceId NAME` accepts local PNG/PTS
input without launching Isaac. No real camera connection was made. External
input starts with unknown recipe and no phase/meter signal or eligible reference.
See [station usage and frame contract](replay/operator/STATION.md).

**Optional real-time input is implemented.** Select **Input mode → Pre-rendered ·
real time**, or launch with `-Source prerender`. The prepared 21.7-second Isaac
clip contains 651 genuinely rendered frames at 30 fps. Camera input is cached;
CV runs live on samples from that same encoded video, skipping old inputs under
backpressure. Browser verification measured 90 presented frames in 3.01 seconds
and ten actual CV samples. The latest mask has its own timestamp/lag. Native
Isaac switching preserves reviews and the journal. Twenty-four operator tests
and nine browser checks passed. The current local instance on port 8767 is using
pre-rendered input. See [playback evidence](setup/verification/prerender-station-20261021.json).

**Mask/deviation visibility fix:** the timestamped sampled mask is now open beside
the real-time video. Above-5-dE valid readings display **Outside development band**
while awaiting the unchanged three-sample confirmation rule. Confirmed alerts
remain visible when a later frame fails QA; current-frame QA remains explicit.
Dashboard polling omits historical mask arrays, while PNGs/logs/full exports keep
their evidence. Twenty-six tests passed. The restarted local session is
`results/operator-prerender-20261021/station-v3`; `station-v1` and its full export
are preserved, and the interrupted `station-v2` launch is retained separately.
See [executed visibility checks](setup/verification/station-visibility-20261021.json).

`results/operator-station-20261021/` retains the 86-frame continuous Isaac source
check and independent RGB/ACK, sampling-color, captured-review and diagnostic
verification. A separate three-frame local-PNG producer passed actual inference
without Isaac. Seven station tests plus the previous 13 operator checks passed.
The guided reference/test/prime/recovery interface is now an explicit
`-Walkthrough` option, with its old evidence preserved separately.
See [continuous-station verification](setup/verification/inspection-station-20261021.json).

**Confirmed hackathon scope, from the reply Austin supplied:** no direct existing
line sensor/actuator communication and no Teledyne database updates. Build the
operator interface; PLC integration is a post-hackathon possibility. The updated
Part 2 package is `results/stage2-team-handoff-20261021/share-v5.zip` (five-page PDF,
seven code-free files, including the clarification). The physical MVP uses
approved video/files, attributed operator recipe/completion input, confirmed vial
identity and acceptable completed-fill references. Do not assume live phase/meter
tags; metered comparisons remain unavailable without qualified progress evidence.
The reply's good/scrap routing description differs from the earlier guide: confirm
the physical destination mapping, and keep it separate from recipe assessment.
See [scope and input implications](team_handoff/part2/OPERATOR-CLARIFICATION.md).

The first **operator response workspace is implemented**. Launch
`pwsh -File setup/run-operator.ps1`, then open `http://127.0.0.1:8766`.
[Walkthrough and boundaries](replay/operator/README.md): completed Run 4 frozen
inference, manual recipe/vial/completion context, exact source/mask evidence,
review request → acknowledgement → actual action → follow-up closure, persistent
append-only SQLite journal and JSON/CSV exports. A new explicitly fingerprinted
endpoint policy leaves existing replay defaults and automatic QA unchanged.
Current geometry/references remain unreviewed/development-only; recipe acceptance
is unavailable, and closure does not authorize release. No live station adapter
or hardware command exists. Nine focused tests and the real-recording headless
browser walkthrough passed, including delayed-response seeking, restart, downloads
and mobile layout. Screenshots/report: `results/operator-workspace-20261021/browser-v3/`.
See [verification](setup/verification/operator-workspace-20261021.json).
Next: operator rehearsal and confirmed CSI input/handling contracts, followed by
approved-reference/evidence adapter integration using reviewed measurements.

The earlier [guided Isaac operator experiment](replay/operator/LIVE-TWIN.md)
is preserved with `pwsh -File setup/run-operator-twin.ps1 -Walkthrough`.
The completed `operator-isaac-20261021/live-v1/`
run used 111 newly rendered RGB frames with actual frozen model inference.
The residue test warned (21 comparisons, peak 12.03 dE); the operator acknowledged
and requested an 8 mL **hypothetical simulated** prime-to-drain, then the recovery
used the retained path inventory (26 comparisons, peak 1.01 dE, no warning).
Reference/test/recovery QA passes were 26/37, 21/37 and 26/37. Final recovery sample
QA remains unavailable; all operational recipe decisions are unavailable.
The browser walkthrough, independent source/process/diagnostic verification and
13 focused tests passed. `results/operator-isaac-20261021/media-v1/` has the full
application video; `browser-v2/` has screenshots. The archived simulator was
stopped cleanly. Its later `interactive-v1/` session is preserved as historical
evidence; the continuous station above supersedes it for current local use.
No source assets, teammate renderer, QA thresholds, replay defaults or main branch
were changed. See [verification](setup/verification/operator-isaac-20261021.json).

The [public control-surface audit](setup/vision/CSI-CONTROL-SURFACES.md) reads all
eight event pages, related control references and both station HMI screenshots at
upstream `fc51de1`. Concrete operator surfaces include batch release/queue,
recipe volumes, inspection RGB/weight settings and existing good/scrap routing.
No team-accessible writable station interface is published or verified. Prioritize
transition review before batch release through the operator; confirm the approved
human handling workflow, destination labels and video/export arrangement. The previous 8 mL
simulated drain purge is not an approved physical recipe. Full snapshots and
93 inventoried outgoing URLs are in `results/csi-control-audit-20261021/`.

The [Part 2 team architecture brief](team_handoff/part2/README.md) is ready in the
same style as the Part 1 student handoff. Five pages cover the CPU/GPU architecture,
shared evidence/context/reference contracts, measured design choices, three proposed
workstreams, digital-twin integration and prospective CSI evaluation. The code-free
PDF/diagram/Markdown package is `results/stage2-team-handoff-20261021/share-v5.zip`;
all five pages were rendered/inspected and quoted metrics checked against the
verification reports. It is local, not published, and does not reassign Part 1 teams.

The [original-video camera correction check](setup/vision/REAL-CAMERA-PREPROCESSING.md)
is complete: 1,581 Run 4 samples, 1,533 camera-eligible pairs, **849 QA passes for
each arm**, six gains/six losses. No aggregate QA gain is demonstrated; keep raw
input as the physical-test baseline. Onset/mask accuracy still needs human review.
All 9,482 native frames and 1,581 sample identities/corrections are independently
checked; 12 paired diagnostic PNGs are retained.

The [CSI visit packet](setup/vision/CSI-SESSION-PACKET.md) proposes a 90-minute
supervised session October 26–28, pending actual availability/CSI confirmation.
`results/csi-preparation-20261021/visit-packet-v1.zip` contains the four-page PDF,
eight CSV forms, frozen baseline hashes/config, recording inspector and
[unsent scheduling request](docs/csi-session-request-2026-10-26.md). The inspector
has been rehearsed on original Run 4: 9,482 decoded frames, no missing/nonincreasing
PTS. Priorities are acceptable complete trajectories, actual recipe/meter logs,
clock/vial links and independent outcomes; flush comparison is optional.
No booking or industrial connection is documented. No returned onset/mask reviews
were found in the local results inventory on October 21.

The **October 21 twin ablation study** is also complete:
[protocol, actual metrics and local videos](setup/vision/TWIN-EXPERIMENTS.md).
`results/twin-experiments-20261021/study-v1/` contains **21 trajectories / 803
live RGB-to-inference frames**, verified by `verification-v1/`; all four diagnostics
were independently recomputed in `analysis-v1/`. Meter interpolation improves
evidence availability under flow changes; post-mask color correction leaves
camera-induced missing geometry unresolved.

The adaptive `preprocess-replay-v1/` reruns the actual frozen models on corrected
saved study RGB. `preprocess-analysis-v2/` verifies all 803 associations/corrections
and diagnostics: **0/10 nominal cases warn and 5/6 residue cases warn**, with
comparisons in all 16 cases. The half-mL-equivalent case is still missed.
787 frames are processed; 16 with unhealthy anchors are blocked. This is adaptive
simulation development, not a held-out live study or real defect accuracy.
Shorter-absorption optical probes recover image-only fill starts, but lower sample
QA and shift reference colors; they are not approved renderer changes. Complete
vial departures remain unsimulated. Operational recipe decisions remain unavailable.

The [Cosmos Transfer trial](setup/vision/COSMOS-TRANSFER-TRIAL.md) is now complete
for one neutral clip. `neutral-v4/` under
`results/cosmos-transfer-trial-20261021/` contains the verified retrieved bundle;
**93 H.264 frames / 5.8125 s** generated successfully on GB10. It invents a blue
pool in the source-empty first frame and changes liquid colors, so source labels
cannot automatically transfer. Frozen-model QA availability drops **24/29→17/29**;
these are simulated model-response counts, not accuracy. No training was started.
Cool/warm variants remain unrun. `verification-neutral-v1.json` checks all 37
retrieved files and 58 retained model samples. The run is finished and its GPU lock
is released. See [completed evidence](setup/verification/cosmos-transfer-20261021.json).

The [first 148-frame live pilot](setup/vision/LIVE-TWIN.md) remains canonical for
the initial transport/architecture check. TensorRT (`bbb6fca`) remains the
recording-speed comparison. Next work includes independent trajectories/appearance,
continuous station state, simulated operator response, and CSI parameter/reference
validation. Cameron/Hayden's renderer and source assets remain unchanged.

1. **YOLO team:** Aleyna, Zoe and Reagan work together on active-vial detection
   and review. Start with [the exact image/label folder guide](team_handoff/README.md#find-the-yolo-images-and-boxes).
   `starter/labels/` contains vial-only class-0 YOLO text labels, not annotated
   pictures or liquid labels. Preserve reviewer exports; no correction-to-training
   conversion has been completed for this release.
2. **Early-liquid evaluation:** the new local review ZIP has six raw Run 4 clips,
   72 PNGs and blank onset/mask forms. It is unreviewed/unassigned and is separate
   from both the published student data and the original 179-case review package.
   Collect human evidence using [the early-fill handoff](annotation_review/EARLY-FILL-HANDOFF.md)
   before reporting onset delay or mask accuracy.
3. **Acceleration:** keep TensorRT opt-in. Full sampled processing improves
   89.48 → 83.22 ms median, but one of 1,581 sample-QA decisions changes and the
   worst paired nonempty final-mask IoU is 0.70455. Inspect the retained difference
   and obtain human geometry evidence before considering promotion.
4. **Stage 2:** the advisory policy has offline/simulated evidence only. Obtain
   CSI-confirmed acceptable references, recipe/meter logs, clock mapping and
   independent physical outcomes. Use the [90-minute visit packet](setup/vision/CSI-SESSION-PACKET.md)
   and its unsent October 26–28 request draft; no booked or executed session is documented.

The Transfer generation and local model evaluation have completed. Inspect live
processes/jobs before future runs. The October 21 follow-up reached Rosie and its
user queue was empty; the completed GB10 generation released its GPU lock. These
are timestamped checks, not enduring availability claims.
Run 5 remains diagnostic-only. Cosmos captions are a separate failed suitability
pilot and are not in the current pipeline.

## Branch and integration history

Continue from **this checkout on `austin-work`**. Replay/vision (`487fa6f`) and annotation
review (`17490c5`) are integrated, with history preserved. The former annotation
worktree is retired. This page supersedes the separate branch/worktree directions.
Austin authorized only the team/data handoff on `main`, published as `1cb6a63`
directly above `5658d58`, without merging experimental history. All development
continues on `austin-work`. The original remote branches remain recovery copies.

## Where things live

| Work | Source and instructions | Current state |
| --- | --- | --- |
| Replay, tracking and measurement QA | `replay/`; [current vision guidance](replay/VISION.md) | Existing detector/defaults preserved |
| Annotation tooling | `annotation_review/`; [reviewer handoff](annotation_review/HANDOFF.md) and [contracts](annotation_review/CONTRACTS.md) | Published 179-case package; reviewed corrections are separate from training TXT |
| Completed comparison | `setup/vision/`; [results and protocol](setup/vision/LIQUID-SPECIALISTS.md) | Four fits/evaluations complete; 12 MP4s verified |
| Latest detector iteration | [First-visible-liquid pilots](setup/vision/EARLY-LIQUID.md) | All three fits and Run 4 evaluations complete; eight-detector scorecard assembled |
| Mask pipeline | [SAM liquid probe](setup/vision/SAM-LIQUID.md) | SAM2.1 tested on 30 Run 4 images; raw masks only |
| Initial pipeline video | [Offline application](setup/vision/SAM-LIQUID.md#offline-pipeline-video--october-20) | Original 60 s Run 4 detector/SAM/sampling demonstration |
| Current full-pipeline experiment/video | [Acceleration and early-fill review](setup/vision/PIPELINE-ACCELERATION.md) | 89.48 → 83.22 ms median sampled processing; clean single-view H.264 demos; one reproducible sample-QA difference |
| Live digital twin | [Live Isaac pipeline](setup/vision/LIVE-TWIN.md) | 148 live frames / four complete sequences; provisional carryover warning; matched scripted-flush case; uncalibrated software-in-loop pilot |
| Latest twin experiments | [21-case study and adaptive replay](setup/vision/TWIN-EXPERIMENTS.md) | 803 verified live frames; corrected saved-RGB replay gives 0/10 nominal and 5/6 residue warnings; one condition per case, not real accuracy |
| Transfer data candidate | [Completed appearance trial](setup/vision/COSMOS-TRANSFER-TRIAL.md) | One verified 93-frame clip; empty-frame liquid/hue hallucinations block automatic label inheritance. QA availability 24/29→17/29; no training. |
| Early pool prompts | [Paired trial](setup/vision/SAM-LIQUID.md#pool-prompt-comparison-and-causal-bridge--october-20) | Localized box/points plus causal bridge; before/after videos complete; QA unchanged |
| Innovation lab handoff | `results/innovation-lab-brief-20261020/` under the local tools root | Six-page architecture PDF, editable Word brief, single-panel final MP4s and code-free sharing ZIP |
| Team data + student brief | `results/team-data-20261020-v1/` under the local tools root | Verified data archives, source manifests, YOLO/COCO starters and three-page architecture assignment PDF |
| Stage 2 action experiments | [Design and metrics](setup/vision/STAGE2.md) | Full Run 4 audit, causal phase checkpoints, policy/reference stress tests and advisory action simulation |
| NVIDIA integration experiments | [Pilot and actual results](setup/vision/NVIDIA-PILOT.md) | TensorRT vial branch 3.13× host speedup; Cosmos3-Edge structured explanations fail 9/12 contracts; optional draft-caption video |

Local assets are under `C:/Users/Austin/ai-club-tools/` (`/mnt/c/Users/Austin/ai-club-tools/`
in WSL). Results, environments and source labels stay outside Git. The ChatGPT
project mirror is reference data, not a checkout.

The innovation lab package is
`C:/Users/Austin/ai-club-tools/results/innovation-lab-brief-20261020/Innovation-Lab-Liquid-Pipeline-Handoff.zip`.
Its `share/` source directory contains only PDF/DOCX, two final-solution H.264
videos (60 s and a 12 s early-fill highlight), two PNGs and a readme. No source
code, weights or labels are included. The videos reuse the verified causal-box
experiment, with source-pixel/PTS checks and full decoding of all 360 output
frames. Builder/provenance files stay separately in `work/`, outside the ZIP.

For the **student-facing team release**, use the newer
`results/team-data-20261020-v1/Team-Architecture-Implementation-Brief.pdf`.
It outlines roles, interfaces and evaluation questions without the final prompt,
threshold or temporal-handling recipe. The earlier six-page innovation brief is
the more detailed reference and is not included in this student data release.
`Team-Data-Guide.pdf` separately explains inventory, formats, splits and gaps.

`archives/Team-Core-Data.zip` (281,264,726 bytes) combines components 00–03:
metadata/handouts, detector starter, raw vial Runs 1–4 and 300 source-aligned
Run 4 PNGs with unreviewed detector inputs. The 721-original/157-development
profile and optional 1,442-sample translated training profile both loaded through
Ultralytics; nine COCO detection exports also passed format/identity checks.
Other-camera context (04) and Run 5 diagnostics (05) remain separate archives.
Full catalog: 4,647 indexed JPEGs, 15 videos, 16,814 exported provisional boxes;
21 vial stills require manual review. No approved masks/onsets, final SAM masks,
prompt/state recipe, executable code, weights or reviewer workspaces are included.
Six focused packaging tests passed; archive CRC/SHA read-back checks and all 300
SAM PNG pixel/PTS checks passed. See `setup/verification/team-data-20261020.json`.
The repository entry point is now [team_handoff/README.md](team_handoff/README.md),
with PDFs, data and checksum links in the versioned
[team-data-v1.0.0 release](https://github.com/ALL-Applied-AI-Network/msoe-ai-club-rockwell-automation-nvidia-hackath-austin-koske-s-team-8e90f213c6/releases/tag/team-data-v1.0.0).
All 11 release assets are published; GitHub sizes and SHA256 digests match the
local files. Both standalone PDFs were downloaded again and their hashes and
three-page counts checked. See the
[publication verification](setup/verification/team-release-publication-20261020.json).
Release attachments keep the large data out of Git. No source labels/data were
modified. The student handoff, PDFs, diagram and download links are also available
on `main`; no application or experiment code was merged there.

## Latest full-pipeline results and Stage 2

Latest complete pipeline: [full TensorRT comparison](setup/vision/PIPELINE-ACCELERATION.md),
local root `results/pipeline-acceleration-20261020/`. A new 640×640 FP16 engine
preserves the existing pipeline input shape; the earlier component-only engine
was 480×640. Four full passes rerun the live YOLO/Faster R-CNN/conditional-SAM
graph on 1,581 source-verified Run 4 samples each: 6,324 measured samples total.
The original PyTorch control reproduces all prior measurements and masks exactly.

Decode + sampled processing p50/p95: PyTorch 89.48/101.20 ms versus TensorRT
83.22/94.39 ms, a 6.99% median latency reduction. Visual tracking and fill state
match on every sample, with 4/6 completed QA windows and nine causal phase
checkpoints in both arms. Sample QA matches 1,580/1,581 frames; at 143.6 s TensorRT
rejects a sample PyTorch accepts. Final masks are identical on 1,531 frames, but
the worst paired nonempty-mask IoU is 0.70455 on an 88→62-pixel small pool. No
backend is promoted, no QA threshold is changed, and no human mask accuracy is
claimed. Both repeats reproduce their respective masks/QA exactly.

Use `published-media-v1/accelerated-pipeline-run4-60s.mp4` for the current clean
single-view application video, or `accelerated-pipeline-six-early-fills.mp4` for
all six early neighborhoods. These are actual measured-pipeline outputs, without
Cosmos captions. Both H.264 videos fully decode, 540 output frames total. The
comparison and the changed-QA PNG remain available for review.

`early-fill-review-v1.zip` (about 26.3 MB) contains six raw clips, 72 exact PNGs,
native frame/PTS identities, blank onset/mask return forms and the portable box
editor. All cases are unreviewed/unassigned; the existing Run 3 milestone and all
reviewer workspaces are intact. The optional box editor does not draw pixel masks
or adjudicate onset. `CSI-VALIDATION-PLAN.md` is an unsent supervised-session draft;
dates, recipes, process-log/clock mapping, approved references and workflow need
CSI confirmation. Twenty focused tests pass; 177 artifacts and all native source
samples are independently verified. See the
[verification](setup/verification/pipeline-acceleration-20261020.json).

NVIDIA follow-up: [pilot results](setup/vision/NVIDIA-PILOT.md), local outputs under
`results/nvidia-pilot-20261020/`. TensorRT FP16 YOLO11n vial-only host processing
p50 is 3.08 ms versus PyTorch 9.64 ms (96 source-verified Run 4 frames, three passes).
All 285 predicted boxes match with zero unmatched boxes; median IoU 0.99955.
This is backend agreement and component timing, not full-pipeline accuracy/speed.

Reason2 access returned HTTP 403; Rosie DNS failed. The existing accessible pinned
Cosmos3-Edge reasoner ran locally in a separate environment. Structured event
explanations pass only 3/12 display contracts and invent unsupported details.
A narrower 12-request caption trial takes p50 0.994 s full-frame / 0.606 s cropped,
but still guesses details and is not selected for operational decisions. Recorded
status messages remain deterministic, quality unavailable and hardware commands none.
The 48-second `cosmos-demo-v1/cosmos-event-review.mp4` labels all six full-frame
captions as unverified drafts, separately from recorded QA counts. All 240 demo
frames fully decode. Transfer/Predict/DeepStream were not added. See the
[verification](setup/verification/nvidia-pilot-20261020.json) for source/metric checks.

Latest: [Stage 2 refinement](setup/vision/STAGE2-REFINEMENT.md), outputs under
`results/stage2-refinement-20261020/`. The next-pilot choice is background-referenced
sRGB color, recipe/ingredient/metered-volume reference lookup, source/camera guards,
and the existing three-sample advisory policy. Deployed replay is unchanged.

Runs 1–3 select background anchors from 1,478 frames. On 849 frozen Run 4 sampling
masks, 28,017 paired pixel tests give green-gain nuisance p95 dE 5.22 raw versus
0.68 sRGB-normalized. These are invariance tests, not absolute color accuracy.
1,800 additional analytic episodes / 7,920 policy evaluations test components,
progression and the combined candidate. Fresh joint seeds give 0/24 null warnings,
72/72 two-mL injection warnings and 45/72 half-mL warnings; all episodes retain
assessable samples. These are simulator results, not real defect accuracy.

Truthfully stale process inputs and inconsistent anchors block decisions; wrong
progression falsely marked current remains a failure. The scoped source audit
finds no recorded acceptable standard preparation or completed physical trial.
Twenty Stage 2 contract tests pass. The 23-second camera-normalization MP4 is an
explicit controlled-pixel demonstration. See the
[refinement verification](setup/verification/stage2-refinement-20261020.json).

Austin authorized offline experiments toward an action pipeline. See
[STAGE2.md](setup/vision/STAGE2.md) and
[verification](setup/verification/stage2-20261020.json). Local outputs:
`results/stage2-20261020/` under the tools root.

Full Run 4: 9,482 frames decoded, 1,581 samples, six image fill candidates;
849/931 within-fill samples pass unchanged per-frame QA. Existing departure-window
QA passes 4/6. Nine causal phase checkpoints include earlier evidence for all six
final episodes, 3.6–4.0 s before departure. Final-episode matching is retrospective;
an ingredient pause does not establish final dispensing completion. Real recipe
assessments remain unavailable.

The analytic policy fixture contains 624 simulated episodes / 3,744 evaluations.
With transplanted Run 4 QA availability, the phase-aware three-sample arm warns on
254/288 injected cases and 0/24 null cases. These are toy injection rates, not real
defect accuracy. One-second progression error and unreported 10% green gain each
cause 24/24 null warnings in a separate stress test: trusted process progression
and commissioned photometric health checks are necessary before operational use.

The proposed advisory loop is review/hold, repeated-mismatch flush recommendation,
recorded completed-flush receipt, then two qualified completed normal vials before
resume recommendation. Local real/simulation MP4s are fully decoded. Ten contract
tests pass. No recipe bank, geometry approval, plant connection or real flush
effectiveness is claimed; deployed replay defaults and sampling QA are unchanged.

Existing untracked `analysis/`, `datasets/`, `docs/`, `experiments/`, `handoffs/`
and `team_plan.md` are preserved in place and excluded **locally** using
`.git/info/exclude`. They are not part of this integration; use `git status
--short --ignored` to see them. Recovery snapshots, including the retired
worktree's ignored build files, are in
`C:/Users/Austin/ai-club-tools/recovery/repo-consolidation-20261020T203829Z/`.

## Completed detector comparison

Root: `C:/Users/Austin/ai-club-tools/results/liquid-specialists-20261020`.
Live scheduler verification during consolidation confirmed array **286264**, all
four jobs **COMPLETED, exit 0:0**. Collector log says the bundle is ready.

`completed-artifacts.tar.gz` and its checksum are now local and verified:
`521608a3d63b6de0458914bcc91c421899acc9addfbacb3312f892d22271cc2f`.
The bundle's existing model files matched local bytes; missing DINO model files
were restored. Local smoke artifacts were preserved separately from remote smoke
artifacts in the bundle. No replacement fits were launched for that experiment.

| Arm | Completed epochs | Selected Run 4 epoch | Local evaluation |
| --- | ---: | ---: | --- |
| `yolo-all` | 29 | 19 | `evaluation-yolo/yolo-all` complete |
| `yolo-focused` | 28 | 18 | `evaluation-yolo/yolo-focused` complete |
| `rtdetr-all` | 28 | 18 | `evaluation-rtdetr/rtdetr-all` complete |
| `dinov3-all` | 25 | 15 | `evaluation-dino/dinov3-all` complete |

The final assembly is `evaluation/`: `comparison.json`, PR/problem-frame PNGs,
12 MP4s and `verification.json`. All videos passed full decode/hash verification
(39,292 output frames). DINO achieves 51.34% liquid AP50–95 / 93.53% recall /
80.19% precision at .35 on provisional Run 5 targets, versus baseline 49.18% /
72.64% / 96.80%. More unmatched boxes accompany the recall gain. See the protocol
for all arms and limits. No detector is promoted; measurement QA is unchanged.

## Current model work: first-visible liquid

Austin wants detection as soon as a visible pool appears at the vial bottom and
continued architecture iteration. New root:
`C:/Users/Austin/ai-club-tools/results/early-liquid-20261020`.
Rosie array **286291**: `yolo11n-960`, `yolo11s-640`, `fasterrcnn-resnet50`.
Local ResNet GPU smoke passed empty targets, two steps and exact checkpoint reload.
All three fits completed, exit 0:0. Selected epochs are 29, 27 and 7 respectively.
VPN access was restored. A stale/NUL-filled management-node read of a training
CSV blocked packaging. CPU-only Slurm job **286305** read the originals on the
training node, verified complete curves/checkpoints and produced
`completed-artifacts-node.tar.gz`, SHA256
`31b86fe156e70d839daa90389030c0f836046682a2ae8d56dfed5dd7ca489a94`.
The incomplete local retrieval is preserved under `retrieval-first-attempt/`;
no remote source file was corrected or sanitized. The polling collector was stopped.

Local `development-run4/` contains the combined eight-detector comparison,
`model-scorecard.png` and short-target PNGs. Common-adapter Run 4 liquid AP50–95:
53.81% (nano 960), 53.66% (small 640), 53.50% (ResNet). The 18 short supplied
targets matched at .35 are 15, 15 and 17 respectively, versus baseline 15/18.
Run 4 has **no <=32px target**; these results cannot measure first-visible pools.
See the [completed verification](setup/verification/early-liquid-results-20261020.json).

SAM2.1 Tiny was tested with liquid-box, vial-box and lower-vial prompts on 30
Run 4 stills. Liquid prompts follow lower colored regions in inspected examples;
vial-only prompts outline glass even on label-absence cases. All 75 raw masks
and 30 overlays are under `prompted-liquid-20261020/run4-probe`. The
[SAM pipeline plan](setup/vision/SAM-LIQUID.md) separates presence/localization,
mask refinement and measurement QA. SAM3 remains an untested concept-prompt arm
requiring authorized checkpoint access. No human masks or onset labels were made.

An offline application now uses ResNet liquid prompts with SAM2.1 and unchanged
per-frame geometry/measurement gates on the first 60 seconds of Run 4. Artifacts:
`results/liquid-pipeline-demo-20261020/render-v1/` (local MP4, posters, decoded
storyboard, source-aligned raw masks and provenance). All 300 video frames and
300 source RGB hashes/native PTS were checked. There are 49 per-frame sampling
passes; this is not full-window QA or an accuracy score. Oversized prompts can
include glass/upper structures. See the SAM protocol for exact counts and limits.

The paired prompt experiment now fixes the inspected early full-vial masks using
localized pool seeds/points plus a causal prompt-box bridge. Final local media:
`results/liquid-pipeline-demo-20261020/causal-box-comparison-v1/`.
`early-fill-before-after.mp4` covers source PTS 10–22 s; `before-after.mp4` covers
60 s. All three new experiment directories and their six MP4s are preserved.
Association-only did not fix oversized prompts; point-only still included lower
glass despite better coarse geometry counts. The final bridge used boxes only
0.2 s old at 13.4/13.8 s, with current seeds and no SAM mask propagation. This is
an inspected Run 4 pilot with a colored-pool prior, not onset/mask validation.

Next: additional development fills and pale/clear cases, other detector prompts,
then forward-only SAM mask propagation with resets per vial. Human body/stream review and
reviewed first-visible-liquid timing remain separate required evaluation inputs.
Use `envs/vision-alternatives-windows/Scripts/python.exe` locally. Do not rerun
completed fits/evaluations or use Run 5 to select settings.

## Annotation review: existing package and correction boundary

Portable package:
`results/annotation-review-20261019/review-handoff-v1.zip`, also published in the
[private v0.2.0 release](https://github.com/ALL-Applied-AI-Network/msoe-ai-club-rockwell-automation-nvidia-hackath-austin-koske-s-team-8e90f213c6/releases/tag/annotation-review-v0.2.0).
From an extracted package, reviewers run `python launch_review.py --reviewer NAME`.
Each gets a separate workspace. The package retains the original 32/32/31 Run 3
assignments; confirm progress with the reviewers rather than treating the October
20 target date as proof of completion. Forty recipe-navigation and 44 QA frames
are follow-up diagnostic cases. Reagan has now joined Zoe and Aleyna on YOLO.

Collect each correction JSON and uncertain cases without merging divergent
histories. Approved boxes and timing remain the data team's decisions. Keep timing
CSV/mappings separate from spatial corrections; never infer still/video alignment
from assumed 30 fps. QA rejection reasons are diagnostic context.
The original labels at `results/typesafe-box-all-20261018` remain read-only.
The editor exports correction JSON; it does not directly rewrite the student
starter's YOLO TXT files. Reviewed source corrections need an explicit, versioned
conversion into the vial-only profile and any translated descendants, preserving
the original train/development split and review provenance. Publish a new dataset
revision after validation rather than overwrite v1.0.0 release assets.

## Verification history

Latest pipeline evidence: 16 annotation-review tests and four pipeline comparison
tests passed; 177 artifact hashes, all 9,482 native source frames, 1,581 sampled
RGB/PTS identities, six raw review clips and both published videos were checked.
See [the recorded verification](setup/verification/pipeline-acceleration-20261020.json).
These are October 20 results; documentation changes do not imply a rerun.

Passed: 12 annotation-review tests; 56 replay tests using `data-windows`; two
specialist-data tests using `vision-alternatives-windows`; annotation JavaScript
syntax check. Follow-up: seven liquid-focused tests, compilation/shell syntax,
full four-arm media verification and ResNet GPU smoke passed. WSL's bare Python
lacks video dependencies; use the existing Windows environments for replay/vision.
No approved labels, physical-onset accuracy or detector promotion is claimed.
