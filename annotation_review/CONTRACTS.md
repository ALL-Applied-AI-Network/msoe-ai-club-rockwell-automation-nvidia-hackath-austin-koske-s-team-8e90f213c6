# Spatial, clock and replay diagnostic interfaces

The tool imports no `replay` modules and writes no replay files. Source MP4s,
organizer annotations, stills and machine suggestions are read-only. The external
replay owner controls measurement validity, recipe context and diagnostics.

## Correction export: `first-vial-spatial-review/v1`

`samples` contains immutable source identities, hashes, original machine records
and fingerprints. Each append-only `revision` records the target, local class ID,
original/proposed box, status, reviewer, notes, revision number and UTC save time.
Coordinates are normalized full-image XYXY. Accepted boxes equal originals;
changed boxes need positive area; rejected boxes are null; uncertain decisions
need a note. Confidence never implies review. Demonstration exports cannot import
into real workspaces. Imports are transactional and reject changed source bytes,
different machine records, out-of-range boxes and divergent histories.

Reviewer names are self-reported, not authenticated signatures. Corrections remain
separate from machine records. There is no automatic YOLO training export or
promotion of uncertainty into a negative label.

## Prioritized cases: `first-vial-review-cases/v1`

`cases.json` lists `sample_id`, priority, assignee, reason and optional separate
`context_references`. Existing still IDs resolve from the prelabel manifest. A new
video-frame case carries `source.kind=video_frame`, camera/run, source MP4 relative
path and SHA256, zero-based decoded frame index, native PTS/time base, PTS seconds,
the extracted JPEG path/hash/dimensions and decoder version. Those timestamps are
MP4 presentation time, never physical elapsed or still-capture wall time.

Batch extraction enumerates actual decoded frames and reads their native PTS. It
does not calculate a video frame from a still timestamp. Provided Run 5 start/end
indices only select navigation images; they are not approved measurement windows
or automatic assignments of any existing still to a vial.

## Published replay QA adapter: `measurement_qa_review`, version 1

The current adapter reads the replay owner's public JSON contract:
`artifact_kind="measurement_qa_review"`, `schema_version=1`. It reads no replay
internals. `source` identifies camera/run/video SHA256 and decoded frame count;
`cases` identifies detections, rejection reasons, unreviewed markers and exact
`representative_frames` with path/hash/frame index/role/MP4 seconds/replay URL.
Only `measurement_available=false` windows enter the initial QA queue.

```text
python -m annotation_review.qa_adapter --manifest /path/to/qa-review/manifest.json --dataset /path/to/csi-vial-fill-dataset --batch /path/to/existing-initial-batch
python -m annotation_review.package_batch --batch /path/to/existing-initial-batch
```

The adapter verifies the source-video hash and exported JPEG hashes, copies the
exact JPEG bytes, and saves the complete source manifest/config/implementation
snapshot. A detection ID is not a supplied vial ID. Reported MP4 seconds remain
replay-export seconds; native PTS is explicitly unknown. No FPS-based clock join
is made. Re-running the same export is idempotent; changed evidence for an
existing case requires a fresh batch so earlier corrections retain provenance.

The current source export has SHA256
`d34ef32e040248ed31f3587f3a3308c91b6bffbbc0cc7e6b9284f6dbae4732c9`,
config `c4feec54d72a7ede`, implementation `ec079cfb8db14a19`.

## Optional alternate diagnostic adapter input: `first-vial-replay-diagnostics/v1`

The replay agent can write this versioned JSON, or an explicit conversion into it.
The adapter does not guess an undocumented manifest schema or read internal caches.

```json
{
  "schema": "first-vial-replay-diagnostics/v1",
  "cases": [
    {
      "case_id": "run5-rejected-window-example",
      "camera": "vial_fill",
      "run": 5,
      "source_video_sha256": "<64 hex characters matching source MP4>",
      "frame_indices": [2226, 2232, 2238],
      "reason": "<diagnostic reason supplied by replay owner>",
      "measurement_config_id": "<replay export configuration>",
      "source_observation_id": "<stable replay observation ID>"
    }
  ]
}
```

The sample frame numbers illustrate shape only, not a current measurement claim.
Run `python -m annotation_review.batch ... --diagnostics FILE` with a **fresh**
output directory. Run-5 MP4 hashes and explicit indices are checked, frames are
extracted exactly, and the input snapshot/hash is preserved as navigation context.
When the manifest is absent, Run 3 and organizer-based transition anchors still
ship. Unknown schemas and mismatched videos fail loudly. The review tool does
not decide which measurements should have passed QA.

## Optional reviewed still/video associations

Load an explicit CSV with `--mapping FILE`:

```csv
sample_id,still_sha256,status,video_sha256,frame_index,reviewer,evidence,notes
```

Statuses: `reviewed`, `proposed`, `unmatched`. Reviewed associations require an
explicit decoded frame index, matching still/video hashes, reviewer and evidence.
Proposed associations remain uncertain. No automatic association is created when
the file is missing. This is a mapping input; box edits cannot approve it.

## Separate reviewed timing input for replay

The existing replay README documents one reviewed `runN.csv` per run:

```csv
vial,recipe_id,t100_ml,t200_ml,t400_ml,dispense_order,start_frame,end_frame
```

Decoded frame indices are zero-based; end is inclusive. Vial IDs, commanded mL,
recipe evidence, ordering and corrected timing belong to Aleyna, Zoe and Reagan.
This editor neither invents them from appearance nor inserts them into box labels.
The replay owner retains validation and ingestion of that CSV contract.
