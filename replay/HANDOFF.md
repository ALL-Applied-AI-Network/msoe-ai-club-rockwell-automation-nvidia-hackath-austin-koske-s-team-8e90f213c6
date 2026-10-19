# Recipe-aware completed-vial replay handoff

October 19, 2026. Branch: `austin/recipe-aware-replay`.
The existing independent replay was published in `0782b26` and its documented
launcher was tested from a fresh `git archive` extraction, with no untracked
source dependencies. This handoff adds explicit recipe input, reference decisions,
QA diagnostics and the source-independent monitoring boundary. Raw data, labels,
synthetic experiments and rendering assets remain unchanged.

## Launch from the published branch

From PowerShell 7, clone the repository or switch an existing clean checkout to
`austin/recipe-aware-replay`. Use your normal data download; videos are not in Git.

```powershell
pwsh -File setup/bootstrap.ps1
pwsh -File setup/run-replay.ps1 -Data 'C:/path/to/csi-vial-fill-dataset' -RecipePlan replay/examples/run5-demo-plan.json -Out "$HOME/ai-club-tools/results/recipe-aware-demo"
```

Open `http://127.0.0.1:8765/`. Use `-Port 8766` for another app. Omit `-RecipePlan`
to start with unknown intended recipes and enter them through **Set intended
recipe**. Pause playback, enter ID, tank volumes, order, your name and source/reason,
then record the choice. It activates at the playhead. **Mark recipe unknown** also
requires a name and reason. One selection can activate per source frame; an
existing selection is retained as history, not silently overwritten.

In image mode the app never reads evaluation annotations. Fill IDs remain image
candidate IDs; they are not supplied vial IDs. Planned selections bind by source
hash and time, never by vial ID, count or observed color. No recipe color is guessed.

For Linux with the existing data environment:

```bash
export TEAM_MEMBER=austin
bash setup/run-replay.sh --data /path/to/csi-vial-fill-dataset --recipe-plan replay/examples/run5-demo-plan.json --out "$HOME/ai-club-tools/results/recipe-aware-demo"
```

Ctrl+C stops the app. Closing the browser pauses no hardware. Browser Play/Pause
controls an already analyzed recording, not an industrial process.

## Reproduce the demonstrated Run 5 decision

The example plan is **illustrative and approved by Austin for demonstration**.
Its recipe definitions were transcribed from the supplied README. Its activation
frames are developer-selected schedule points in image-detected idle gaps, not
reviewed operator logs or newly adjudicated event labels. It asserts an intended
plan for the demo and does not establish historical recipe correctness.

Use a fresh output directory for an isolated reproduction:

```powershell
$python = "$HOME/ai-club-tools/envs/data-windows/Scripts/python.exe"
$demoOut = "$HOME/ai-club-tools/results/recipe-aware-demo"
& $python -m replay make-reference --runs 5 --recipe-plan replay/examples/run5-demo-plan.json --reference-fill 1 --at-frame 900 --out $demoOut
& $python -m replay analyze --runs 5 --recipe-plan replay/examples/run5-demo-plan.json --out $demoOut
& $python -m replay export-review --runs 5 --recipe-plan replay/examples/run5-demo-plan.json --out $demoOut
pwsh -File setup/run-replay.ps1 -RecipePlan replay/examples/run5-demo-plan.json -Out $demoOut
```

These commands use the canonical dataset location in the repository. Add
`--data 'C:/path/to/csi-vial-fill-dataset'` to Python commands and `-Data` to the
launcher for another location. Repeated `make-reference` creates another explicit
reference choice; ordinary replay/analysis never chooses a reference automatically.

Then open `http://127.0.0.1:8765/?run=5&frame=2244`. Detected fill 2 completes at
frame 2244. The recipe 5 definition is T100=10, T200=10, T400=10 mL, order
T200>T400>T100. Its observed Lab is approximately `[5.50,-3.97,1.15]`; the selected
fill-1 reference is `[10.09,-13.22,6.01]`. Deviation is **9.56 ΔE2000**, so the
provisional **5.0** threshold requests review. The reference was measured at
frame 852 and explicitly selected at frame 900, before fill 2 starts at frame
1374. Fill 1 cannot assess itself using that reference.

The reference is a development example from a partial first fill, **not certified
clean product**. This comparison exercises eligibility and decision plumbing; it
does not establish which vial is defective, the cause, or a production threshold.
Every result is advisory and no result authorizes release.

Outputs under `$demoOut/run5/`:

| File | Purpose |
| --- | --- |
| `manifest.json` | Input recording hashes, runtime, measurement/config/implementation IDs, source session and limitations |
| `recipe-plan.json` | Exact explicit input and provenance used for replay |
| `references.json` | Exact reference bank used for decisions |
| `features.jsonl` | Sampled image evidence and causal detector state |
| `events.jsonl` | Source lifecycle, recipe activation, stream/fill events and available decisions |
| `observations.csv` | Color, QA, recipe context, deviation, reference ID/provenance and outcome |
| `decisions.json` | Full audit: input manifest, plan/hash, references, decisions and separately grouped counts |
| `qa-review/manifest.json` | All 30 measurement outcomes, including rejected-window evidence for Aleyna |

Event detection remains 30 accepted fills on Run 5; independent boundary scoring
is a separate `validate-detection` command. Measurement coverage remains 19/30.
With this one development reference, comparison results are one review and 29
unavailable decisions (missing eligible references or rejected image QA).
No defect precision/recall follows from these counts.

## Intended-recipe contract

See `examples/run5-demo-plan.json`. Top-level fields are `schema_version:1`,
`plan_id`, `source_description` and `selections`. Each selection contains:

| Field | Meaning |
| --- | --- |
| `selection_id` | Unique explicit selection identity |
| `run_id`, `source_video_sha256` | Source binding; mismatches are rejected |
| `available_frame` | When the choice was known in source time |
| `active_from_frame` | When it becomes active; cannot precede availability |
| `recipe_id` | String, or null to declare unknown |
| `volumes_ml` | Finite nonnegative T100/T200/T400 commanded mL |
| `dispense_order` | Each positive-volume tank exactly once, e.g. T200>T100 |
| `recorded_by`, `recorded_at_utc` | Authorship and actual recording timestamp |
| `provenance.kind`, `.description` | `manual_operator`, `operator_plan` or `illustrative_demo`, with a source explanation |

Recipe input contains no supplied vial IDs, quality labels or fill-end truth.
A fill uses the selection known at its estimated start. A later selection cannot
repair missing start context; a different definition during the fill, up to
departure, makes its comparison unavailable. Reusing a recipe ID with changed
volumes/order creates an incompatible recipe definition. Measurement QA stays
separate from context QA. Batch positions/first-vial holds need a future explicit
batch input and are not invented from these selections.

The HTTP API uses GET/POST `/api/recipe-plan`. POST accepts `run_id`,
`frame_index`, recipe fields, `recorded_by`, `description` and optional
`active_from_frame`; the server stamps availability at the replay playhead and
records UTC authorship time. This is an offline replay configuration interface.
A future physical adapter must use the monitor-owned cursor, not a client-supplied
past frame. `MonitorSession.select_recipe(...)` implements that boundary.

Manual history is persisted in `$demoOut/operator-state.json`, including the
original imported plan. Restarting the app with the same output restores it and
the reference bank. An incompatible imported plan requires a fresh output folder.
Seeking backward hides later activations/references again. Configuration downloads
contain the full declared plan/bank for auditing; decisions remain time-gated.

## Reference contract

Use the completed-vial button or `make-reference` for a deliberate choice. A valid
choice needs a completed QA-valid observation and stable known recipe context.
The bank persists in `$demoOut/development-references.json`. An imported bank uses
`--references bank.json` / launcher `-References` and must contain `references`.

Eligibility requires matching recipe ID **and definition**, measurement config
**and implementation profile**, source run/video/annotation provenance, a different
source detection, and `available_frame < evaluated start_frame`. The source
measurement must also have existed before reference selection. The outcome
exports rejected-reference reasons rather than substituting another recipe.

Every reference records its identity, Lab, scope, measurement profile, source
run/video/event, source measurement availability, choice availability and source
description. `same_run_development` references selected in the app also include
recipe-selection provenance, choice UTC and `certifies_clean_product:false`.
Legacy banks without required provenance/availability must be recreated explicitly.

External commissioning references need an explicit source description and target
run/video availability; `available_frame:-1` declares availability before the
recording. This is a documented supplier assertion, not automatic certification.
Their recipe-definition and measurement-profile IDs must still match. Missing,
incompatible, current-fill and future references produce `unavailable`.

## Aleyna's diagnostic handoff

Default generated location: `$demoOut/run5/qa-review/manifest.json`.
Schema: [`review-manifest.schema.json`](review-manifest.schema.json), version 1.
Use `--review-base-url http://127.0.0.1:8766` if replay uses another port.

The manifest includes source hash/fps, unchanged config, counts and one `cases[]`
entry per accepted image detection. Each entry has its detection/candidate ID,
`measurement_available`, rejection reasons, estimated start/end, departure and
confirmation/availability landmarks in frames and MP4 seconds, every sampled
window frame and QA flag, accepted samples, measured color and replay links.
For each rejected window it additionally saves representative JPEGs: **all actual
window samples plus departure**, with relative paths, roles and SHA256 hashes.
Every case remains `review_status:"unreviewed"`, `reviewed_label:null`.

For the annotation agent: resolve image paths relative to the manifest directory;
identify a case using `(source.video_sha256, detection_id)`, not supplied vial ID.
Read `measurement_available` as QA usability, never as product quality or event
recall. Treat estimated boundaries and rejection reasons as diagnostic proposals.
Write reviewed timing/window decisions in a separate companion artifact using the
team's chosen format. Do not overwrite generated manifests; export can regenerate
them. This handoff does not adjudicate labels or prescribe replacement windows.

## Physical-demo input boundary

`inputs.FramePacket` contains source frame index, relative source timestamp, RGB,
optional PTS/time base and optional independently reviewed wall-time estimate.
`RecordingSource` owns MP4 decoding. `MonitorSession` owns feature extraction,
image-only detection, unchanged stationary-tail measurement and assessment. Both
the batch app and a future approved source use `MonitorSession.ingest(packet)`.
No industrial camera adapter, address or network connection is implemented.

The current packet contract is sampled constant cadence: every `sample_stride`
frames, matching the declared source fps. A future adapter must supply reviewed
geometry/identity and timing rather than fabricating an MP4 hash or dropping
frames silently. Source timing is not automatically physical timing.

1. Construct a **new** session, call `start()`, and explicitly supply/confirm recipe
   context. The default context and reference bank are empty.
2. Feed sampled packets in order; after every completed fill, persist the new
   observation/decision and emitted events. A bounded tail supports color
   measurement; full feature history is retained only when requested by replay.
3. `stop(reason)` records session termination and abandons an open fill. EOF never
   creates a completed fill. Missing per-frame timestamps make measurements
   unavailable; frame gaps, cadence jumps and out-of-order input stop the session
   with `InputInterrupted`. Invalid geometry also stops it.
4. Recovery creates a new session identity and fresh detector/measurement state;
   `start()` cannot resume a stopped object. Reconfirm manual recipe context.
   The first fill without an observed idle state is marked possibly partial.
   Never join a fill or automatically reuse a reference across interrupted input.

The browser's **Restart** replays the same saved recording/history. A new physical
session is a separate object with no inherited operator context. Restarting the
application restores its explicitly saved replay history, not physical continuity.
No software start/stop action here controls, stops or authorizes industrial motion.

## Remaining mentee inputs and scope

| Group | Input still needed |
| --- | --- |
| Aleyna / Zoe / Reagan | Independently reviewed frame timing, event boundaries, usable measurement windows, uncertainties and source hashes; Run 3 weak/partial fills especially |
| Cameron / Hayden | Their small versioned synthetic experiment and checked manifests; rendering/model quality remains their work |
| Srithej / Josiah | Confirmed operator recipe-selection workflow, CSI booking/supervision, permitted source/feed method and network arrangement |

Austin still needs to choose/commission meaningful recipe references and a real
operator plan for physical use. No decision is needed to run this illustrative
demo. Physical integration remains blocked on the approved feed/workflow and
supervised session. Stage alerts, Cosmos, training, rendering changes, threshold
tuning and label adjudication are outside this handoff.

## Verification

```powershell
& "$HOME/ai-club-tools/envs/data-windows/Scripts/python.exe" -m unittest discover -s replay/tests -v
node --check replay/static/app.js
```

The suite adds recipe changes/unknown context, reference eligibility and provenance,
recipe/measurement incompatibility, source interruptions, fresh-session restart,
app history restoration, diagnostic export and the shared processing path.
Run 5 evidence and the QA manifest are captured in `verification/recipe-aware-demo.json`.
