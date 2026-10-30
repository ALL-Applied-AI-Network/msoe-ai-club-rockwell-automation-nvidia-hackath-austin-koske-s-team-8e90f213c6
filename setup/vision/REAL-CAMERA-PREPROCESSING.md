# Original-video camera correction check — October 21

Completed development experiment on the original **Run 4** recording. No
synthetic perturbation was injected, no training ran, and Run 5 was not used.
**Keep raw input as the physical-test baseline.** Upstream correction gives no
aggregate sample-QA gain on this recording. Its simulation robustness result does
not establish improved real onset/mask accuracy.

## Paired results

Fully decoded all 9,482 native frames; evaluated the same 1,581 samples at 5 Hz.
The original raw PyTorch baseline is reused from its completed, hash-verified
`pipeline-acceleration-20261020/full-run4-v1/pytorch-pass0/` inference. The
candidate reruns vial/liquid/SAM inference and the unchanged QA on corrected RGB.

| Diagnostic | Raw baseline | Correction candidate |
| --- | ---: | ---: |
| All attempted source samples | 1,581 | 1,581 |
| Camera-eligible paired samples | 1,533 | 1,533 |
| QA passes on those paired samples | 849 | 849 |
| Nonempty final geometry masks on paired samples | 857 | 857 |
| Samples with a liquid proposal on paired samples | 928 | 927 |

The correction gate is unavailable for **25 startup samples and 23 unhealthy
background-anchor samples**. There is no raw fallback. All raw-baseline QA passes
also happen to lie in the paired subset; raw QA passes across all attempts are
849 as well. Six samples gain QA and six lose it, for 12 disagreements. These
are quality-gate changes, not reviewed improvements/regressions.

Among 863 pairs with a nonempty union of final geometry masks, median paired mask
IoU is **0.97124**, minimum **0**. This compares two model outputs and is not mask
accuracy. Both-empty pairs are excluded from that IoU summary. First-visible delay,
mask IoU/Dice against truth, and false early detection remain unavailable until
the data team returns independent reviewed labels.

## Frozen protocol

The background reference and patches are unchanged from `camera-reference-v2`:
`x192-y224`, `x112-y224`, `x112-y160`; maximum patch disagreement 0.05, channel-gain
range 0.65–1.55. The first-five-second reference becomes available only at 5 s.
Correct healthy frames by dividing encoded sRGB channels by the estimated gain,
rounding and clipping to uint8 **before** all inference. Raw and processed RGB
hashes are separate. Unavailable frames emit no candidate inference and reset all
candidate temporal state to prevent bridging across a blocked interval. Thus the
candidate includes its stated abstention/reset policy, not just a pixel transform.

YOLO vial, Faster R-CNN ResNet-50/FPN liquid, SAM2.1 Tiny, prompt/bridge settings,
confidence/NMS, measurement config and QA source hashes remain frozen. The only
baseline helper text difference is a previous CLI metadata wording correction;
imports, constants and inference definitions are verified equivalent by AST.
Executed candidate helpers and their hashes are retained. The existing downstream
camera-feature diagnostic is omitted from the candidate output because it would
apply a second normalization to processed RGB. Recipe decisions are not scored.

This is inspected Run 4 development footage. The fixed reference is specific to
this camera geometry; it is not a commissioned new-camera profile. No threshold
selection is made from these results. Neither pipeline is promoted by this check.

## Evidence and reproduction

Local root: `C:/Users/Austin/ai-club-tools/results/csi-preparation-20261021`.

- `real-camera-v1/`: frozen plan, helpers, 1,581 frame records, comparison and
  completed marker.
- `verification-v1/`: independent recount, all native raw/processed pixel and
  clock checks, and 12 paired diagnostic PNGs. Raw review clips should be viewed
  before these overlays to avoid influencing human onset/mask decisions.
- `capture-rehearsal-v1/`: the visit's recording-inventory helper exercised on the
  actual original Run 4 file; all 9,482 native PTS are present and increasing.
- `visit-packet-v1/` and `visit-packet-v1.zip`: four-page CSI PDF, eight blank CSV
  forms, frozen inference config/model hashes, read-only recording inspector,
  scheduling-request draft and early-review instructions. Booking remains unconfirmed.

Helpers: `evaluate_real_camera_preprocessing.py`,
`verify_real_camera_preprocessing.py`, `inspect_csi_capture.py`, and
`prepare_csi_session_packet.py`. Use fresh output directories. The focused
`test_csi_preparation.py` checks source-byte preservation, actual encoded native
clock inventory, output isolation, and unavailable/healthy camera input behavior.

See [CSI visit packet](CSI-SESSION-PACKET.md),
[scheduling request](../../docs/csi-session-request-2026-10-26.md), and
[human onset/mask review](../../annotation_review/EARLY-FILL-HANDOFF.md).
