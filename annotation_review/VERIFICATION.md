# October 19 handoff verification

The 179-case ZIP was extracted into a fresh directory and all 213 then-packaged
files matched their package SHA256 inventory. A clean Python virtual environment
with no review runtime dependencies launched the extracted source. The documented
`python launch_review.py --reviewer Aleyna` path also launched successfully (a
separate port/workspace was selected for the check). The launcher made no model
calls. Final packages include this evidence and an updated hash inventory.

## Browser correction round trip

1. Opened a demonstration workspace on the extracted package.
2. Selected Run 3 still `vial_fill_20260923_184630_557477`, machine liquid target.
3. Changed its right coordinate to `0.530938`; saved a demonstration note.
4. Clicked **Export corrections**; downloaded the JSON.
5. Opened a second empty demonstration workspace, clicked **Reopen correction
   JSON**, selected the downloaded file, then selected the liquid candidate.
6. Confirmed restored coordinates, note and demonstration reviewer in the UI.
   Exported source snapshots and revision history compared equal across both
   workspaces. `verification/roundtrip-proof.json` records the hashes and geometry;
   `verification/demo-correction.json` is the actual downloaded correction.

The demo is ineligible as a reviewed label. Its still association remains
unmatched. No timing, recipe, vial identity or quality judgment was approved.

The browser check also exercised accepting an original vial suggestion after
fixing UI rounding: acceptance sends exact immutable machine coordinates rather
than rounded displayed coordinates. That second check is demonstration-only too.

## Input integrity and focused tests

After UI edits, SHA256 checks passed for **4,647 indexed source stills, 84 case
frames, three organizer reference-label files and the Run 5 source MP4**.
`verification/source-integrity.json` records the covered inputs. All original
machine metadata remains unreviewed. This verifies covered file integrity and
software behavior; it makes no annotation-accuracy claim.

Twelve focused tests cover HTTP review/export/import, immutable originals, source
tampering, optimistic conflicts, forged imports, exact acceptance, uncertainty,
demo separation, explicit clock mappings and the public QA adapter. JavaScript
syntax validation passed. Run the tests from the repository or extracted ZIP:

```text
python -m unittest discover -s annotation_review/tests -v
```

To inspect the saved demonstration without approving a team label, use a fresh
demo workspace and import with the same `--dataset`, `--prelabels`, `--cases` and
`--demo` arguments. A real human workspace refuses this demonstration export.
