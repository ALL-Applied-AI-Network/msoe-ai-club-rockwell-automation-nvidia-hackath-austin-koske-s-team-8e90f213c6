# CSI test request draft: recipe-aware liquid inspection

Prepared October 20, 2026 for Austin's team. **Draft; not sent, booked or executed.**
Operational details below require CSI confirmation. The organizer feedback supplied
by Austin is the source for the existing flow meters, shared-path residue,
Teledyne color/weight inspection and supervision/network-approval requirements.

## Objective and requested session

Request one **90-minute supervised commissioning session**. Preferred dates:
to be supplied by Austin and confirmed by CSI. Attendees: Austin Koske, the CSI
operator/supervisor designated by Shamar Webster, and a data-team representative
if available. CSI operates recipes and interventions; Austin records permitted
evidence and runs offline analysis. Request channel: the organizer's
`#data-and-testbed-requests` with Shamar tagged, or the supplied CSI contact.
No message or industrial connection has been made by preparing this document.

The first goal is one independently documented intended-recipe mismatch, if the
approved test naturally produces one, plus an acceptable comparison. The initial
session establishes data quality and feasibility. It does not estimate production
defect rates or prove that flushing works.

Proposed time budget, subject to CSI: 15 minutes for workflow/clock confirmation,
20 minutes for acceptable references, 40 minutes for an approved comparison and
15 minutes for export/evidence checks. CSI confirms whether the requested fills
and conditioning fit that duration before booking.

## Questions to settle before running

1. What station signal identifies the selected recipe, active ingredient, actual
   cumulative per-fluid delivered quantity and dispensing completion? Are exported
   historical Run 4 records already available for an offline-first test?
2. How is the physical vial identified between filling and inspection/routing?
   When is the latest time an operator can request hold/review in the existing
   workflow? Record that deadline and its clock source.
3. What makes a vial unacceptable for its intended recipe? CSI supplies a written
   acceptance criterion and measurement uncertainty before outcomes are scored.
4. Which existing recipes and comparable initial shared-path conditions are
   approved for an A→B comparison? CSI defines conditioning and any flush procedure.
5. Which recording/export arrangement is permitted? Prefer supplied file exports.
   Any live feed or device/network arrangement requires CSI's explicit approval.

## Initial experiment

**Reference preparation:** request independently prepared/verified acceptable
examples of two CSI-selected existing recipes, A and B, under the same camera
conditions. Target three independently prepared examples per recipe if feasible.
Record actual ingredient amounts, measurement method/uncertainty, vial identity,
camera settings and the staff acceptance decision. A label that says “Recipe B”
or a final vial that looks stable is insufficient on its own as a reference.

For early warnings, also request **complete known-acceptable dispensing
trajectories**, with the same camera, ingredient-phase and cumulative-meter logs.
CSI verifies their conditioning and outcomes. Independently prepared static
mixtures anchor acceptable appearance, but final-mixture examples alone do not
define what an unfinished fill should look like. Expected intermediate appearance
must be commissioned from verified trajectories or a separately validated mixing
model, with held-out acceptable fills used to check that reference progression.

**Observed changeover:** record one approved A→B sequence under the existing
workflow, including up to three identified B vials. Record intended recipe,
metered progress, actual independent outcome evidence, camera video and any
available Teledyne color/weight verdict. Residual liquid is a hypothesis to test;
we do not assume any vial is unacceptable or that Teledyne passes it.

**Intervention comparison:** if CSI approves, record a second sequence following
its approved flush workflow, with a newly established comparable starting
condition. Again observe up to three B vials. CSI supplies intervention settings
and execution receipts. A flush receipt alone is not recovery evidence.

The initial target is six reference examples and up to six observed B vials,
plus CSI-defined conditioning. If outcomes are inconclusive, retain them as
uncertain and request the specific missing evidence; do not relabel based on our
camera score. A subsequent study can predeclare at least five independently
conditioned sequences per arm, balanced in order where practical. Multiple frames
or successive vials from one shared-path condition are not independent repeats.

## Evidence to export

Keep original files and a SHA256 manifest. Unknown fields remain empty/null.

| File | Required content |
| --- | --- |
| Camera video | Original source file, camera identity/settings and native decoded frame/PTS/time-base inventory. Other synchronized camera recordings if approved and available. |
| Recipe/process log | Station clock, intended recipe, physical vial ID, ingredient/phase, cumulative delivered quantity per fluid with units, and authoritative dispensing-complete state. |
| Clock mapping | Explicit process-clock/native-video anchors, video hash, source frame/PTS/time base, mapping method, reviewer and estimated uncertainty. No wall-time conversion from assumed frame rate. |
| Independent outcome record | Physical vial ID, intended recipe, actual ingredient amounts/composition evidence, method/uncertainty, CSI acceptance criterion/version, accepted/unacceptable/uncertain decision, reviewer and evidence file IDs. |
| Existing inspection/routing record | Teledyne color/weight result if available, physical disposition, latest actionable deadline and clock, linked by confirmed vial identity. Missing means unavailable. |
| Intervention record | Approved workflow ID, request/acknowledgement/completion times, operator and receipt, plus conditioning evidence for the next sequence. |

Suggested process-log columns, adapted to the actual CSI export:

```csv
process_time,clock_id,physical_vial_id,intended_recipe_id,phase_id,tank_1_cumulative_ml,tank_2_cumulative_ml,tank_3_cumulative_ml,dispense_complete
```

Use actual tank identifiers/units. Preserve the raw log; a derived adapter must
record conversions, timestamp semantics and hashes. Do not infer metering or recipe
identity from liquid color. Process freshness checks cannot detect a wrong clock
mapping falsely marked current; confirm alignment using documented landmarks.

## Analysis and success criteria

Freeze the measurement profile, expected trajectory, thresholds and evidence gates
before scoring the prospective sequences. Acceptable references follow intended
recipe, current ingredient and cumulative metered volume; unfinished mixtures are
not compared with final-recipe color throughout filling. Reference preparation and
validation sequences must remain distinct.

Report these separately:

- Eligible-measurement coverage and unavailable reasons, including missing inputs.
- Warning lead relative to the confirmed routing deadline, with synchronization
  uncertainty. A warning after that deadline is not an actionable detection.
- Warnings on independently unacceptable vials, and unnecessary warnings on
  independently acceptable vials. Show exact denominators and unavailable cases.
- Early-liquid detection delay and mask quality only where the data team has
  reviewed first-visible timing and spatial masks under a written convention.
- Recovery evidence after an executed approved intervention, with independent
  outcomes for subsequent completed vials. Report observed changes without
  claiming causal flush efficacy from one comparison.

A compelling initial demonstration is: a confirmed physical vial differs from
its intended recipe, qualified evidence identifies the mismatch before disposition,
and the operator can review the evidence in the existing workflow. Include the
existing inspection verdict only when recorded; the stronger “Teledyne passed it”
story requires that exact linked evidence. Repeated independent sequences are
needed before making performance or waste-reduction claims.

If no independently unacceptable vial is observed, report the demonstrated
measurement/coverage behavior and the absent defect truth. Do not manufacture a
successful result from synthetic color shifts or a recognized color name.
