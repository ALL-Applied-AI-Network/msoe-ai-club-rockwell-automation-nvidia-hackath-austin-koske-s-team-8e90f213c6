# Investigation panel

The run-review screen shows possible causes, endpoint evidence and requested checks. The panel calls the existing CPU sequence-diagnosis pilot; it
never changes model predictions, measurement QA, recipe decisions or hardware.

## Use

Open a run from **RUN HISTORY**; the panel is below the frame review. **THIS RUN** starts unavailable:
processed frames do not create completed-vial records. The card shows the actual
frame count separately from the imported vial count and lists the missing inputs.

**SYNTHETIC EXAMPLE** opens four pinned feature-level examples from the completed
pilot: transient recovery, persistent deviation, a delivery-record discrepancy
inside the research color band, and unhealthy camera evidence. Each carries a
persistent synthetic banner and no real session ID. Exports include the synthetic
mode in the JSON and filename. Selecting an example does not save it as run evidence.
The camera frame and operator decision continue to use their original evidence.

A transient first-vial shift followed by convergence can support a carryover
hypothesis, but dosing errors or local optical changes can produce the same
trace. A persistent shift stays cause-undetermined. An attributed delivery
record can remain relevant when color falls inside the experimental band.
Confirmed physical cause, product decision and hardware command remain null.

## Completed-vial imports

Choose **IMPORT REVIEWED VIAL RECORDS** after processing has stopped. Supply a JSON
wrapper with `reviewer`, `evidence_id`, `document` and optional
`temporal_reference_reviews`. The document uses `vial-diagnosis-input/v1`, defined
in [the diagnosis module](../replay/diagnostics.py). Human declarations are
attributed inputs; the application does not independently authenticate them.

The input must identify this run/camera and profile
`vial-fill-640x480-temporal-v1`. It contains intended/previous recipe context,
independent accepted recipe endpoint references, an analysis boundary `as_of_s`,
and one to 32 externally reviewed completed-vial rows. Do not derive physical vial
IDs or fill completion from color or treat successive frames as different vials.

Every row must use an existing model event ID as `observation_id` and its saved
original image URL as `source_evidence_id`. Copy the exact saved Lab and actual
`measurement_valid` as `qa_pass`; an import cannot replace them. Recipe context,
camera/profile and source-clock ordering must match. Distinct source events,
physical-vial IDs and completion evidence are required.

Clock IDs are `recording_seek:<session_id>` or `browser_elapsed:<session_id>`,
matching the actual event. `source_time_s` is its stored timestamp divided by 1000.
Neither clock is native camera PTS. Stills with unknown source time need an explicit
mapping before they can participate; this panel does not invent that mapping.

Recipe references must be independent, accepted endpoint references from the same
camera/profile/domain/clock, available before evaluation. Real references require
an attributed human review (`status=human_reviewed`, `origin=human_review`, reviewer,
evidence ID and condition `accepted_recipe_endpoint`). Demonstration or simulated
references cannot qualify real records. The current/previous references are
separate from the earlier empty image used by the temporal model.

An unreviewed temporal reference stays unqualified unless the import supplies a
matching review record. That record needs the exact reference SHA256, real domain,
camera/profile/clock IDs, earlier source and availability times, attributed reviewer
and evidence ID, `status=human_reviewed`, `origin=human_review`, `condition=empty`
and `alignment=confirmed`. Demonstration records are excluded. All remaining
completion, alignment, camera-health and identity declarations must also be explicit.

Optional command/delivery checks need attributed source records linked to the same
run, recipe, clock and evaluated vial IDs. The diagnosis only uses records available
by `as_of_s`. It cannot verify a physical meter or command from an uploaded JSON.

Imports are appended to SQLite with the original submitted document, source-bound
normalized document, reviewer/evidence ID and computed diagnosis. Existing frame
artifacts/decisions stay unchanged. **DOWNLOAD EVIDENCE** exports the diagnosis,
source references, alternatives and missing checks for investigation. An imported
result is historical evidence at its stated analysis boundary, not a current-feed
health indicator. New runs and source/view changes discard obsolete UI responses.

## Interfaces and verification

- `GET /api/v1/investigation/examples` lists the four examples.
- `GET /api/v1/investigation/examples/{id}` computes a labeled synthetic diagnosis.
- `GET /api/v1/sessions/{id}/investigation` returns the last import or explicit
  unavailable state. It never converts frame history into vial endpoints.
- `POST /api/v1/sessions/{id}/investigation` validates and retains real completed-vial
  imports. Mock events, duplicate sources, edited Lab/QA and wrong runs/clocks fail.

The fixed pilot settings remain ΔE2000>5, at least three vials for a recovery pattern,
late/first ratio≤0.5 and previous-recipe Lab direction cosine≥0.8. These are research
settings rather than commissioned product tolerances. No detector, causal model
or new training is added. The 1,620-sequence analytic pilot is described in
[evaluation](evaluation.md); its counts do not establish real cause accuracy.

Run API checks with `cd api && uv run --no-sync pytest`. The 12 diagnosis-contract
checks are `python -m unittest discover -s replay/tests -p test_diagnostics.py -v`.
Browser verification and the portable Docker kit include the synthetic/run boundary
and an explicit unavailable real-run example.
