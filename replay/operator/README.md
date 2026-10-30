# First Vial operator workspace

**October23 full-workflow demo:** [two-station app and complete pipeline](FULL-APP-DEMO.md)
replays exact real filling frames with temporal/SAM controls and actual synthetic
cap-state predictions, plus a working source-pinned review/response/export API.
It is opt-in; existing station defaults and the mentees' frontend/API are unchanged.

**October 23 candidate integration:** [two-station commissioning rehearsal](COMMISSIONING.md)
adds source-bound common evidence, reference/reset guards and a CPU cached-output
rehearsal for filling and final-fixture observations. It is opt-in and does not
change this application's model defaults. Real accuracy/references and physical
vial-to-slot linkage still require human/CSI inputs.

A working operator interface for the confirmed hackathon boundary: observe the
recording, supply intended recipe and vial context, inspect evidence, and record
the human response. It runs separately from the existing replay app.

For the **continuous live inspection station**, use
`pwsh -File setup/run-operator-twin.ps1` and open `http://127.0.0.1:8767`.
The [station and frame-input contract](STATION.md) cover automatic frame ingestion,
retained reviews and operator responses. Isaac is the default test-frame producer;
`-Source frames` accepts local producer input without starting Isaac.
For smooth playback, choose **Input mode → Pre-rendered · real time**, or launch
with `-Source prerender`. The 30 fps cached camera video feeds actual live CV;
the timestamped CV sample and lag remain visible separately. Switching sources
preserves reviews. See [cache preparation and playback details](STATION.md#optional-real-time-pre-rendered-input).
The older [scripted simulator experiment](LIVE-TWIN.md) is available with
`-Walkthrough`; it is no longer the default interface.

## Launch

From the repository in PowerShell:

```powershell
pwsh -File setup/run-operator.ps1
```

Open **http://127.0.0.1:8766** yourself; Ctrl+C stops the server. The launcher uses
the existing `data-windows` environment under `AI_CLUB_HOME`, or
`C:/Users/Austin/ai-club-tools`. It does not open a browser.

The default input is the completed Run 4 `pytorch-pass0` evidence in
`results/pipeline-acceleration-20261020/full-run4-v1`, with the matching original
`datasets/dataset2/csi-vial-fill-dataset/vial_fill/videos/vial_fill_video_4.mp4`.
These local inputs are not bundled with this app or the student handoff.
No GPU or model inference is needed. Paths can be supplied explicitly:

```powershell
pwsh -File setup/run-operator.ps1 -Evidence 'D:/results/full-run4-v1' -Video 'D:/data/vial_fill_video_4.mp4' -Out 'D:/results/operator-practice'
```

With a Python environment containing the existing replay dependencies:

```bash
python -m replay.operator --evidence /path/to/full-run4-v1 --video /path/to/vial_fill_video_4.mp4 --out /path/to/operator-practice
```

The output directory contains a persistent `operator-journal.sqlite`. Reuse it
to resume this recording; use a fresh directory for a different evidence identity.
Do not use browser-verification workspaces as actual operator records.

## Walkthrough

1. Enter your name. Inspect the source video or jump to a sample fill. Navigation
   comes from image-detected candidates, not confirmed physical vial identities.
2. At the beginning of the vial, select **New vial** and supply intended recipe,
   batch, physical vial ID and the basis of that association. Unknown recipe and
   unconfirmed identity remain explicit. The demonstration does not infer them.
3. After dispensing actually finishes, confirm completion. An ingredient pause
   is not completion. The endpoint uses three recent stable samples from the same
   visual track, after the case start, subject to the existing per-frame QA gates.
   A case spanning multiple image fill candidates is blocked.
4. Expand the source-aligned evidence to view the original frame, proposed mask
   overlay or binary mask. Its exact sampled frame and clock appear separately
   from the continuously playing video.
5. Request a review when evidence is uncertain or needs a human response.
   Acknowledge it, record the action actually performed, then close the follow-up
   once the vial record is complete or closed as uncertain. Acknowledgement alone
   does not clear a review. Reopened reviews require a new response.
6. Export JSON for complete provenance/history or CSV for the case ledger.
   Exports contain only events visible through the current playhead.

For a **UI practice run only**, the first neighborhood can be recorded from
11.2 to 26 seconds, and the second from 56.8 to 70 seconds. Use clearly invented
IDs such as `DEMO-A` and `DEMO-001`, not claims about organizer vial identities.
The first completed endpoint can be saved through **Record tools** as a development
reference for the later case. Reference comparison requires matching recipe and
measurement profile, with the reference saved strictly before the later case
started. Same-fill and future references cannot assess that case.

## Evidence and decision boundaries

This version is **recorded demonstration**, not a commissioned live station
interface. It reads frozen model outputs and checks their pinned plan/artifact
hashes and original video hash. Preview pixels are decoded at exact native PTS
and checked against frozen RGB hashes. No camera, line sensor, actuator, industrial
network, PLC or Teledyne database connection is made.

The endpoint policy is `operator_endpoint_last_three_samples/v1`, with its own
measurement-profile fingerprint. It adds operator-confirmed completion; it does
not alter or claim to reproduce the replay's automatic departure-window QA.
The existing replay defaults, QA thresholds and frozen inference remain unchanged.

Current masks are unreviewed model proposals and references are development-only.
**Recipe assessment stays unavailable**, even when a diagnostic color comparison
is within the development band. No independently verified product-quality outcome,
approved team label, metered progress or release authorization is generated.
Approved-reference ingestion and qualified live/file evidence adapters are future
work requiring actual CSI inputs and validation.

Operator actions are attributed manual entries, not authenticated proof of physical
intervention. Procedure-based entries require a procedure/receipt identifier; the
software does not execute the procedure. Closing a follow-up never authorizes
product release. Reported labeled-good/labeled-scrap destinations are separate
fields with unconfirmed semantics, as required by the conflicting station descriptions.

The SQLite history rejects updates/deletes through its normal schema, detects
stale revisions, and makes retried commands idempotent. This is a local audit trail,
not a tamper-proof or multi-user identity system. Backward seeking hides future
context and prevents backdated edits to a case with later history.

## Verification

```powershell
& "$env:USERPROFILE/ai-club-tools/envs/data-windows/Scripts/python.exe" -m unittest replay.tests.test_operator_workspace -v
```

Nine focused checks cover evidence identity, endpoint safety, reference causality,
review lifecycle, restart persistence, idempotency/concurrency, history projection,
loopback requests and exports. `replay/tests/verify_operator_browser.py` adds a
headless Chrome walkthrough against the actual Run 4 video, screenshots, downloads,
reload, mobile layout and delayed-request/backward-seek behavior. That optional
verification helper uses Playwright in a separate test environment; Playwright is
not an application dependency. It creates separate demonstration records and
stops only its own server. See the verification JSON linked from the root handoff.
