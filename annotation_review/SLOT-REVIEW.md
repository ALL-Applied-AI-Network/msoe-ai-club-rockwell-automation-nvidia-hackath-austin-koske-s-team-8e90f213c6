# Real final-fixture slot review

October23 expanded packet:
`C:/Users/Austin/ai-club-tools/results/dataset1-fiveclass-20261023/review-v1/review.html`.
It has960 targets across G and S,192 positional priority cases and five-class
machine suggestions. Mapping and visibility remain unreviewed; the five-class
baseline has poor synthetic false-alarm performance and is not approved for use.
The scorer handles complete30-position fixtures,60-position scenes and frozen
warning escapes separately from wrong defect names. All eight focused tests pass.

The separate organizer-native manual packet is
`C:/Users/Austin/ai-club-tools/results/dataset1-fiveclass-20261023/native-review-v1/index.html`:
two1920×1080 examples,120 unresolved targets, no AI suggestions. Use that package's
own import/export workflow and retain reviewer identity separately. Native
coordinates must not be silently transferred to the640×480 packet.
The instructions below describe the earlier480-target G-only packet; counts and
model capability differ in the expanded version.

Open:
`C:/Users/Austin/ai-club-tools/results/dataset1-real-transfer-20261022/review-v1/review.html`

Sharing ZIP: the sibling `review-v1.zip`. No server, credentials, GPU or runtime
installation is needed. The source images are real cap_seal stills from Runs1–4;
all class suggestions remain unreviewed. No team-approved labels are supplied.

1. Enter your own name. Start with **Priority:96 targets**; all480 are available.
2. Confirm that the highlighted center aperture is the intended target. G IDs
   follow the synthetic authored grid and are not yet independently approved
   real position coordinates. Use incorrect/uncertain mapping when appropriate.
3. Check whether the target is visible enough. A robot/hand or ambiguity can make
   it unavailable; occluded/uncertain visibility requires an uncertain class.
4. Inspect the raw full scene and zoomed crop. Choose `pass`, `missing_cap`,
   `crooked_cap`, `damaged_vial`, `no_vial` or `uncertain`, with notes as needed.
5. Save explicitly. Only then can you display the machine draft. Review the first
   few items together to agree on conventions; keep uncertain cases explicit.
6. Download JSON regularly. Each save appends history. Keep each reviewer's export
   intact; imports reject a different packet, older history or divergent history.

`pass` means visible cap/body appear acceptable, not hidden quality or recipe
correctness. Distinguish an open transparent vial from an empty hole. Vial lean
alone does not prove a crooked cap. Neighboring vials are not the target. The CNN
supports only capped/uncapped/empty; a `pass` suggestion cannot clear crooked or
damaged cases. An empty slot is not a missed placement without attributed progress
or completion context. No fill-to-fixture physical identity is assumed.

The priority targets are six fixed positions per still, chosen independently of
predictions. Source hashes, crop coordinates and task identity travel with each
saved decision. Still/video mapping remains unmatched; do not infer a video frame
from capture wall time. These reviews are for evaluation only, never fitting the
synthetic-trained models or choosing operating thresholds.

From the repository root, score a real human export into a fresh directory:

```text
python -m annotation_review.slot_review score --packet PATH/review-v1 --review PATH/reviewer-export.json --out PATH/fresh-reviewed-results
```

The result reports reviewed coverage and a full five-class confusion table,
per-class misses/false positives/precision/recall and whole-scene coverage. Only
explicit visible labels with confirmed mapping are scored. Unknown/occluded or
unreviewed cases are not counted as successes. Missing-class recall is unavailable
when no reviewed positive examples exist. Source-matched partial development
review is not a production accuracy estimate.

`?demonstration=1` uses a separate test workspace. Its exports are ineligible for
real scoring. Executed headless verification saved/downloaded/reopened an uncertain
demonstration decision, checked matching histories, refused packet mismatch and
divergent import, and verified the scorer rejects that demonstration export.
No real human label was created by that test. Six focused review-integrity tests
and four transfer-input integrity tests pass.

Full experiment: [Dataset1-style real transfer](../setup/vision/DATASET1-REAL-TRANSFER.md).
