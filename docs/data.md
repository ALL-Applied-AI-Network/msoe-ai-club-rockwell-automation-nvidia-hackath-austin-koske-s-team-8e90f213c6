# Data and models

## Current model release

[Download temporal-cnn-v1.0.0](https://github.com/ALL-Applied-AI-Network/msoe-ai-club-rockwell-automation-nvidia-hackath-austin-koske-s-team-8e90f213c6/releases/tag/temporal-cnn-v1.0.0).
Repository access is required. Download release attachments, not GitHub's
automatically generated source archives.

| Attachment | Contents |
| --- | --- |
| `temporal-cnn-assets-v1.zip` | Liquid/vial checkpoints, metadata, frozen helper, inference scripts, example images/masks and single-panel video |
| `temporal-cnn-training-v1.zip` | 192 synthetic paired renders/masks, scene specifications and original manifest |
| `Temporal-CNN-Architecture.pdf` | Current model architecture diagram |
| `CSI-Tuesday-Visit-Guide.pdf`, `guide-v3.zip` | Original visit plan and blank field sheets |
| `SHA256SUMS.txt` | Download checksums |

Exact sizes and hashes: [model release manifest](assets/temporal-cnn-v1.json).
Both ZIPs contain their own file manifests. Extract outside Git and retain the
originals. [Run instructions](development.md#run-liquid-inference).

| Component | Training and selection |
| --- | --- |
| Liquid U-Net | Random initialization; 144 synthetic images from 36 scene families; 48 development images from 12 disjoint families; epoch 40 selected from 60 |
| YOLO11n vial geometry | Generic COCO initialization; synthetic CSI-domain fitting; final epoch 12 |

Liquid selection maximizes synthetic development presence accuracy plus positive
geometric mask IoU, evaluated every five epochs with earliest ties. These are
development scores, not an independent test. Supervision is geometric liquid
body, which can differ from visible liquid through glass.

| File | SHA256 |
| --- | --- |
| Liquid checkpoint | `6714fe32191d8eba230aa87246f38a25c2906c619d60d7ba30c9e60205352eef` |
| Vial checkpoint | `0bfac07a256eaf190416b632ad1f094a6b534309df1bb11a8c175e9d811b56bc` |
| Frozen model helper | `a1bd2e6c2dc80d35743049ace3e37f30fe4c0c8f1cd9aa8614867c90081fd31c` |

Do not modify the frozen helper to change an existing model. New architectures
or weights require a new identity, metadata and evaluation.

## Real recordings and provisional annotations

[Download team-data-v1.0.0](https://github.com/ALL-Applied-AI-Network/msoe-ai-club-rockwell-automation-nvidia-hackath-austin-koske-s-team-8e90f213c6/releases/tag/team-data-v1.0.0).
This older package contains real recordings, machine boxes and legacy detector
formats. Its `train/` names and original PDFs do not authorize real-data fitting.

| Archive | Contents |
| --- | --- |
| `Team-Core-Data.zip` | Combined components 00–03, about 281 MB |
| `00-team-guide.zip` | Original briefs, diagram, inventory and contracts |
| `01-detector-starter.zip` | YOLO/COCO exports and translation ancestry |
| `02-vial-source-runs1-4.zip` | Vial stills, four videos and machine annotations |
| `03-sam-frames-run4-development.zip` | 300 source-aligned frames with native timing and box proposals |
| `04-other-cameras-runs1-4.zip` | Tank and cap camera context |
| `05-run5-diagnostic-only.zip` | Previously inspected diagnostic data |

Use the combined core or components 00–03; their contents overlap. Optional
components 04 and 05 are separate. Extract under the common
`liquid-team-data-v1/` root. [Archive manifest](assets/team-data-v1.json) ·
[checksums](assets/team-data-v1-sha256.txt) · [original format guide](archive/Team-Data-Guide.pdf).

YOLO labels are text files separate from images, with one row per box:
`class_id center_x center_y width height`, normalized to the full image. COCO
boxes are pixel `[x, y, width, height]`. Do not train on images with boxes drawn
on them. Class maps differ between exports; the vial-only starter uses class 0.

In the extracted legacy starter, matching files are separate:

```text
liquid-team-data-v1/starter/
  images/train/    image files
  labels/train/    matching YOLO text files
  images/val/      Run4 development images
  labels/val/      matching YOLO text files
```

Each label shares its image's filename stem. `vial-originals.yaml` selects the
original-image profile; `source_legacy_labels/labels/` preserves the original
two-class labels. These are historical export names; use this real imagery for
evaluation under the current training policy.

The vial starter contains 721 original Runs 1–3 images and 157 Run4 images.
A historical profile repeats/translates these into 1,442 samples; they are not
1,442 independent originals. Machine boxes remain provisional. A missing liquid
box is not proof of emptiness, and some older boxes include the incoming stream.

## Training and evaluation policy

New CSI-domain fitting and checkpoint/threshold selection use synthetic data.
Keep scene families and descendants in the same split. Freeze weights and
thresholds before real transfer scoring; disclose generic pretrained weights.
Real imagery and human review are evaluation inputs. Legacy real-trained models
remain development comparators. Run5 is excluded from fitting, selection,
threshold tuning and sampling decisions; it is not a blind final test.

Retain decoded frame indices, native PTS/time base and image/source hashes.
Still capture timestamps must not be converted into video indices using assumed
30 fps. Cross-camera vial identity requires explicit association evidence.

## Human review

The visible-pool review tool distinguishes no visible pool, visible pool and
uncertain. Reviewers outline accumulated liquid while excluding the stream,
glass, holder and exterior reflections. First-visible onset is an interval
between a reviewed absent frame and a reviewed present frame.

`annotation_review.mask_review` builds an offline packet from frozen probe and
context directories. `annotation_review.score_mask_review` scores human exports;
`score_sam_temporal_review` compares the same source-matched labels across arms.
They require the original experiment packet, not the two-image inference example.
Consult each command's `--help` for its required paths.

Human exports use full-image 640 × 480 alternating RLE starting with a zero run.
Replay uses cropped foreground runs; these encodings are not interchangeable.
Unreviewed, uncertain and demonstration decisions must not be promoted to
approved labels. Preserve reviewer exports and histories separately.
