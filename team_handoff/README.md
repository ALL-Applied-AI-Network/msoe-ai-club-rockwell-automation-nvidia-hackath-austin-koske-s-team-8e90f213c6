# Liquid vision team handoff

**October 26 temporal CNN release:** [current architecture, exact model downloads and runnable examples](temporal_cnn/README.md). The selected liquid path is direct paired U-Net; Faster R-CNN and SAM are optional comparators. Austin authorized this scoped handoff on `main`; unrelated development remains on `austin-work`.

**Tuesday October27 preparation:** [visit guide and test plan](integration/CSI-VISIT-20261027.md).
The local six-page PDF/ten-form packet is under`results/csi-tuesday-20261027/guide-v3`.
Time, host assignment and approved video arrangement remain unconfirmed. Published
CSI workstation information lists Windows11 /RTX PRO5000 Blackwell; confirm the
actual unit. The integrated worker/API/UI must pass the saved-clip rehearsal first.


**Current architecture decision — October 24:** [direct temporal U-Net, SAM-free](../setup/vision/CURRENT-ARCHITECTURE.md).
This supersedes the original PDF's required R-CNN → SAM liquid path for the new
integration candidate. YOLO still supplies vial geometry; U-Net supplies the
liquid mask/presence directly. SAM2.1 is optional refinement. The original release
PDFs and data remain unchanged; live adapters and physical validation are pending.

**End-presentation materials (October 22):**
[synthetic training and digital twin brief](presentation/README.md) — eight-page
PDF, editable text, speaker notes and single-panel demo in a local sharing ZIP.
It separates completed Dataset 2 experiments from the proposed Dataset 1 extension.
**Subsequent Dataset 1 work:** [first synthetic fixture pilot](../setup/vision/DATASET1-PILOT.md)
now covers capped/uncapped/empty at explicit G positions. Both small classifiers
score360/360 targets on12 held-out synthetic scenes; real transfer, two remaining
defect classes and physical cross-station identity are unimplemented. The dated
presentation PDF predates this pilot; its original edition is preserved.

Dataset **`liquid-team-data-v1-20261020`** · [Download release v1.0.0](https://github.com/ALL-Applied-AI-Network/msoe-ai-club-rockwell-automation-nvidia-hackath-austin-koske-s-team-8e90f213c6/releases/tag/team-data-v1.0.0)

Build a pipeline that detects accumulated liquid as soon as it is visibly identifiable
at the vial bottom, produces a useful mask, and reports whether the image supports
a reliable sample.

**Current training boundary, reaffirmed October 22:** use generated synthetic
images for new CSI-domain training. Real recordings and their labels support
inspection/evaluation, not weight fitting. The starter's historical `train/`
directory names and older PDFs do not override this boundary. Existing real-trained
models are legacy development comparators, not evidence of a synthetic-only
competition solution. Keep synthetic scene/seed splits intact, select on synthetic
development evidence and disclose generic pretrained weights. Run 5 remains
diagnostic-only and excluded from fitting/selection.

**Local synthetic training pilot:**
`C:/Users/Austin/ai-club-tools/results/synthetic-liquid-20261022/Dataset-2-Synthetic-Pilot-v1.1.zip`
contains 768 generated images (614 train / 154 development), two-class YOLO labels
and single-class vial/liquid COCO exports. It includes no real images or weights.
Its labels are provisional geometric boxes; optics and first-pool coverage remain
limited. This is the existing synthetic fixture with a reproducible scene split,
not a newly improved renderer. It has not been published to GitHub. See the
[October 22 transfer measurements](../setup/vision/SYNTHETIC-TRANSFER-AUDIT.md).

## Start here

1. Read the [architecture implementation brief (PDF)](https://github.com/ALL-Applied-AI-Network/msoe-ai-club-rockwell-automation-nvidia-hackath-austin-koske-s-team-8e90f213c6/releases/download/team-data-v1.0.0/Team-Architecture-Implementation-Brief.pdf).
   It defines the three modules and their interfaces; your team designs and explains the implementation.
2. Read the [data guide (PDF)](https://github.com/ALL-Applied-AI-Network/msoe-ai-club-rockwell-automation-nvidia-hackath-austin-koske-s-team-8e90f213c6/releases/download/team-data-v1.0.0/Team-Data-Guide.pdf) for formats, provenance, splits and missing labels.
3. Download [Team-Core-Data.zip](https://github.com/ALL-Applied-AI-Network/msoe-ai-club-rockwell-automation-nvidia-hackath-austin-koske-s-team-8e90f213c6/releases/download/team-data-v1.0.0/Team-Core-Data.zip) (**281.3 MB**).
   It contains components 00–03 below, including both PDFs. You can start all three teams with this download.
4. Verify against [SHA256SUMS.txt](SHA256SUMS.txt) and extract outside the Git checkout.
   The shared top-level directory is `liquid-team-data-v1/`.

This is a private repository; sign in with a GitHub account that has repository access.
The release attachments contain data and documents, with no implementation source,
weights, final-solution masks or detailed solution recipe. GitHub's automatically
listed source-code archives are repository snapshots, not the team data download.

## Three teams

### Current assignments — October 22

Austin confirmed these assignments after the meeting; they supersede earlier
proposed ownership in the dated briefs, while the model interfaces below remain
the integration contract.

| Owner | Current responsibility |
| --- | --- |
| Josiah and Hayden | Operator dashboard/frontend and backend pipeline; consume standard model observations and preserve evidence and human response history. |
| Srijan | Combine the models into a standard pipeline; own frame/coordinate/model interfaces and explicit unavailable outputs. Coordinate the backend handoff with Josiah/Hayden. |
| Zoe, Alayna and Reagan | Investigate NVIDIA additions such as Cosmos; compare candidates with the baseline and document content/label integrity and performance limits. |
| Cameron | Current pushed module: SAM2.1 pilot and mask evaluation on `team-3`. |
| Austin | Local experiments, integration support and supervised-test preparation. |

Austin reported that Brett suggested exploring Dataset 1 alongside our filling
pipeline and emphasized demonstrating the sim-to-real gap. Treat this as mentor
direction, not a measured result or a guarantee about judging. Dataset 1 adds
final-fixture visual inspection; fill-to-fixture identity needs independent
evidence. Dataset2's cap_seal folder already supplies real overhead fixture
imagery (1181 stills/5 videos;335 stills in Runs1–4). Its reviewed per-slot defect
labels and real coordinate mapping remain to be established or obtained from the
separate Dataset1 package, alongside physical validation. Review/approval ownership is not inferred from a model
or an NVIDIA-tool result.

The [Dataset 1 extension proposal](integration/dataset1-extension-20261022.json)
describes a second station observation alongside fill inspection. The separate
[three-class synthetic pilot](../setup/vision/DATASET1-PILOT.md) implements a first
experimental classifier/evidence output. The full proposal, real validation and
confirmed cross-camera vial link remain future work.
The [frozen RGB real-camera probe](../setup/vision/DATASET1-REAL-TRANSFER.md) has now
completed480 slot-frame predictions on16 cap_seal stills. Its
[offline review packet](../annotation_review/SLOT-REVIEW.md) is ready; counts remain
provisional and real accuracy awaits human mapping/visibility/class review.

The next stage has a separate [Part 2 architecture brief](part2/README.md):
recipe/phase/metered references, qualified warnings, operator recovery and digital-twin
testing. It includes completed experimental evidence and proposed implementation
workstreams. It extends this Part 1 handoff; the original data release and team
assignments remain the starting point. The new PDF/sharing ZIP are local artifacts,
not published release attachments.

For the October 22 team meeting, use the [updated seven-page Part 2 architecture
release](part2/MEETING-20261022.md): operator workflow, current experimental evidence,
proposed integration workstreams, digital-twin/CSI tasks and remote compute layout.
The PDF and code-free ZIP are local artifacts; the published Part 1 data is unchanged.

**Integration and Monday readiness supplement (October 22):** a new five-page brief
follows the original architecture handout and adds module deliveries, shared
evaluation, operator integration and October 22–26 checkpoints. Its
[editable document source](integration/brief-20261022.json) is in this checkout.
The PDF and code-free ZIP are prepared locally at
`C:/Users/Austin/ai-club-tools/results/team-integration-brief-20261022/release-v2`
and `release-v2.zip`; this edition clarifies synthetic-only training. The original
`release-v1` is retained. CSI booking/video access remain unconfirmed; this supplement
has not been published to GitHub and does not replace the original data release.

The confirmed hackathon scope is an **operator interface**, with no direct existing
line sensor/actuator access or Teledyne database changes. The revised Part 2 brief
uses operator context/completion and approved video; live metering is not assumed.
See the [project-specific clarification](part2/OPERATOR-CLARIFICATION.md).

| Team | Starting model | Responsibility | Start with |
| --- | --- | --- | --- |
| 1 | YOLO11n | Active-vial detection and geometry | Core: detector starter and vial source imagery |
| 2 | Faster R-CNN / ResNet50-FPN | Liquid localization, especially early shallow pools | Core: starter labels, COCO exports and vial source imagery |
| 3 | SAM2.1 / Hiera Tiny | Mask prompting/refinement and aligned integration | Core: source videos and 300 Run 4 development frames with machine boxes |

As of October 21, **Aleyna, Zoe and Reagan are working together on Team 1
(YOLO)**, including review of the vial boxes. The starting dataset remains v1.0.0;
this documentation update does not add reviewed training labels.

## Find the YOLO images and boxes

Use **Team-Core-Data.zip** above or
[01-detector-starter.zip](https://github.com/ALL-Applied-AI-Network/msoe-ai-club-rockwell-automation-nvidia-hackath-austin-koske-s-team-8e90f213c6/releases/download/team-data-v1.0.0/01-detector-starter.zip).
Both contain this layout:

```text
liquid-team-data-v1/
  starter/
    images/train/                 training pictures
    labels/train/                 matching YOLO box files
    images/val/                   development pictures
    labels/val/                   matching YOLO box files
    vial-originals.yaml           default vial-only profile
    train-originals.txt            selects 721 original training images
    development.txt               selects 157 Run 4 development images
    vial-with-translations.yaml   optional historical translated profile
    source_legacy_labels/labels/  preserved original two-class labels
```

**The boxes are stored separately; they are not drawn onto the JPEGs.** For example:

```text
starter/images/train/vial_fill_20260922_170432_316717.jpg
starter/labels/train/vial_fill_20260922_170432_316717.txt
```

Load an image with its matching annotation file in a compatible annotation tool
to display the rectangles. Each TXT row is
`class_id x_center y_center width height`, with coordinates normalized to the
image dimensions. In `starter/labels/`, **class 0 is vial**. Those files are
not liquid labels; the original two-class files are preserved separately.

Start with **`starter/vial-originals.yaml`**. The training directory also contains
`-shift` image/label pairs; the default file list selects the originals. The
optional translated profile uses all 1,442 training samples. Do not substitute
the entire folder for the originals list without intentionally choosing that
profile. Extracted `raw/` imagery is source material, not this training profile.
If `starter/` is missing, check that you downloaded Core or component 01 rather
than only the raw-data component or GitHub's automatic source-code ZIP.

## Review and correct labels

The existing [annotation review release](https://github.com/ALL-Applied-AI-Network/msoe-ai-club-rockwell-automation-nvidia-hackath-austin-koske-s-team-8e90f213c6/releases/tag/annotation-review-v0.2.0)
contains `review-handoff-v1.zip`, a separate portable review package. Extract it
and run `python launch_review.py --reviewer Reagan`, substituting your own name.
Open `http://127.0.0.1:8775`. It displays the package's saved machine boxes and
records human revisions in a separate reviewer workspace. It is not an arbitrary
YOLO-TXT import tool; its initial Run 3 queue is a bounded review batch, not every
image in the detector starter.

Agree on vial-box conventions, correct or reject errors, add missing targets and
mark uncertain cases with notes. Keep the original package and each reviewer's
export intact. **The editor exports correction JSON; it does not automatically
update the training TXT files.** Reviewed corrections need a validated conversion
into a new versioned training dataset, including any affected translated copies.
Until then, the released training boxes remain provisional machine suggestions.

## Optional component downloads

Use the combined core **or** components 00–03; they contain overlapping files.
Extract any chosen component ZIPs to the same parent directory. Components 04 and
05 are separate additions and are not in the core.

| Archive | Size (decimal MB) | Contents |
| --- | ---: | --- |
| [00-team-guide.zip](https://github.com/ALL-Applied-AI-Network/msoe-ai-club-rockwell-automation-nvidia-hackath-austin-koske-s-team-8e90f213c6/releases/download/team-data-v1.0.0/00-team-guide.zip) | 1.7 | PDFs, architecture diagram, shared inventory and contracts |
| [01-detector-starter.zip](https://github.com/ALL-Applied-AI-Network/msoe-ai-club-rockwell-automation-nvidia-hackath-austin-koske-s-team-8e90f213c6/releases/download/team-data-v1.0.0/01-detector-starter.zip) | 121.0 | YOLO and COCO detection formats, run splits and translation ancestry |
| [02-vial-source-runs1-4.zip](https://github.com/ALL-Applied-AI-Network/msoe-ai-club-rockwell-automation-nvidia-hackath-austin-koske-s-team-8e90f213c6/releases/download/team-data-v1.0.0/02-vial-source-runs1-4.zip) | 92.6 | Original vial images and four source videos, machine annotation exports |
| [03-sam-frames-run4-development.zip](https://github.com/ALL-Applied-AI-Network/msoe-ai-club-rockwell-automation-nvidia-hackath-austin-koske-s-team-8e90f213c6/releases/download/team-data-v1.0.0/03-sam-frames-run4-development.zip) | 66.0 | 300 source-aligned PNGs, native PTS/frame identity and machine box proposals |
| [04-other-cameras-runs1-4.zip](https://github.com/ALL-Applied-AI-Network/msoe-ai-club-rockwell-automation-nvidia-hackath-austin-koske-s-team-8e90f213c6/releases/download/team-data-v1.0.0/04-other-cameras-runs1-4.zip) | 245.5 | Optional tank/cap camera context with camera-specific class maps |
| [05-run5-diagnostic-only.zip](https://github.com/ALL-Applied-AI-Network/msoe-ai-club-rockwell-automation-nvidia-hackath-austin-koske-s-team-8e90f213c6/releases/download/team-data-v1.0.0/05-run5-diagnostic-only.zip) | 391.2 | Optional inspected diagnostics; keep separate from model development |

Exact archive sizes, file counts and hashes: [archives.json](archives.json).
Checksums cover all seven ZIPs and both standalone PDFs.

## Formats and evidence

- **Training: Runs 1–3. Development: Run 4.** Run 5 has already been inspected:
  use it only for diagnostics, never fitting, checkpoint selection, threshold tuning
  or sampling decisions. This package does not provide a blind final test.
- The default vial starter has **721 original training images and 157 development
  images**. A separate historical profile includes **1,442 training samples** with
  existing translated copies; these are not 1,442 independent originals.
- YOLO labels use normalized center/width/height coordinates. The vial starter
  remaps vial to class 0 and preserves original two-class files separately.
  COCO exports use pixel XYWH boxes. The PDFs specify each module's interface.
- Existing machine boxes are **unreviewed suggestions**. Missing liquid labels are
  provisional negatives, not confirmed-empty frames. Legacy liquid boxes may
  include the stream and are not approved accumulated-pool annotations.
- The full optional collection has **4,647 indexed stills, 15 source videos and
  16,814 exported provisional boxes**. Twenty-one vial stills require manual review.
  No approved liquid masks, first-visible onset labels or track identities are supplied.
- Do not infer video frame indices from still capture timestamps or assumed 30 fps.
  The SAM starter has explicit decoded frame indices/native PTS; still/video
  associations remain unmatched unless separately reviewed.

Original data and labels remain intact. Aleyna, Zoe and Reagan own approved
spatial labels and timing decisions; keep reviewer exports separate and preserve
provenance. Source imagery, model weights and reviewer workspaces stay out of Git.

## Working in the repository

Continue code work on `austin-work`; Austin has requested no push or merge to `main`.
Keep modules in their existing directories: `setup/vision/`, `annotation_review/`
and `replay/`. Publish future data revisions as new versioned release assets rather
than overwriting the evidence behind existing reviews.
