# Data-review handoff — October 19, target October 20

The first milestone is Run 3 vial-fill review. Tank and cap/seal prelabels remain
available, but neither blocks this milestone. All machine suggestions start
unreviewed; the demonstrator's revisions are never mentee-approved labels.

Download **review-handoff-v1.zip** from the private repository's
[annotation review release](https://github.com/ALL-Applied-AI-Network/msoe-ai-club-rockwell-automation-nvidia-hackath-austin-koske-s-team-8e90f213c6/releases/tag/annotation-review-v0.2.0).
Repository access is required. This is a downloadable package each teammate runs
locally; Austin's running localhost service is not needed.

## Initial batch

- **95 Run 3 stills:** a complete bounded sweep, including suppressed liquid
  candidates. Review for faint/partial fills and missing targets; presence scores
  do not decide absence. Stills have no approved video/vial association by default.
- **40 Run 5 exact video frames:** middle and provided-end navigation anchors for
  first and later vials of each of the 10 organizer recipe groups. These are
  extracted frames, not substituted stills or approved measurement windows.
- **44 Run 5 QA frames:** exact exported images from 11 rejected measurement
  windows in the replay owner's public `measurement_qa_review` version-1 export.
  Rejection reasons are diagnostic context, not approved product-quality labels.

The package has **179 navigation/review cases**. Start with Run 3; the other 84
frames are follow-up work. Source video/index/hash is retained for every frame.
Organizer anchors retain native PTS; QA frames retain the export's MP4 seconds
and explicitly mark native PTS unavailable.

## Named handoffs

| Reviewer | First milestone | Follow-up frames |
|---|---|---|
| Aleyna | First 32 Run 3 stills chronologically | 16 anchors (groups 5, 8, 11, 14) + 16 QA frames |
| Zoe | Next 32 Run 3 stills chronologically | 12 anchors (groups 6, 9, 12) + 16 QA frames |
| Reagan | Final 31 Run 3 stills chronologically | 12 anchors (groups 7, 10, 13) + 12 QA frames |

The queue's Assigned filter contains these assignments. Start with the **Run 3**
cohort. Use the first five images to agree on vial/liquid box conventions, then
finish your assigned Run 3 stills by October 20. Mark difficult cases uncertain
instead of guessing. Aleyna gathers QA questions; Zoe compares first/later recipe
examples; Reagan collects proposed clock landmarks for separate timing review.
Video anchors and QA windows follow after the Run 3 checkpoint. For each image, correct vial/liquid
geometry, reject spurious candidates, add missing scoped targets, or leave an
uncertain decision with notes. Save decisions and export one correction JSON per
reviewer. Keep individual workspaces; imports do not merge divergent edits to the
same target silently.

Aleyna, Zoe and Reagan own any approved label and timing decision. Record recipe
evidence and corrected fill timing in the separate replay CSV contract. A spatial
acceptance never approves a recipe ID, fill boundary, volume or measurement.
Reagan can consolidate proposed clock landmarks from all three reviewers, but
no wall-time-to-frame conversion becomes reviewed merely because it is plausible.

## Portable launch from the provided batch ZIP

Unzip into a fresh directory; Python 3.11+ is the only runtime prerequisite.

```text
python launch_review.py --reviewer Aleyna
```

Use your own name. The default workspace is a separate sibling `review-work-NAME`
directory. The ZIP includes the selected stills, extracted frames and **all saved
machine prelabel metadata**. To inspect non-batch original imagery, supply:

```text
python launch_review.py --reviewer Zoe --dataset /path/to/full/csi-vial-fill-dataset
```

The source-video link needs the full dataset, since videos are not duplicated into
the small review ZIP. Exact frame images/index/PTS/hash remain in the package.
For a shared remote machine use an SSH tunnel; the editor binds to loopback.

## Replay-agent dependency

The current package consumes the published `measurement_qa_review` version-1
manifest from `recipe-aware-demo-20261019/run5/qa-review/manifest.json`. It includes
30 detections, 11 rejected windows, config `c4feec54d72a7ede`, implementation
`ec079cfb8db14a19`. The exact source snapshot is `qa-source/manifest.json`;
its SHA256 is recorded in `cases.json`. Detection IDs are not organizer vial IDs.
No still-to-vial assignments were created.

The replay owner provides future diagnostic exports. A changed config/export
requires a fresh batch; do not overwrite evidence already used for review. The
adapter is documented in `annotation_review/CONTRACTS.md`. Run 3 ships without
waiting for any subsequent QA export.

No detector is trained, no synthetic examples are generated, and no annotation
accuracy is inferred from model confidence. Cameron/Hayden and the replay owner
retain their assigned scopes.
