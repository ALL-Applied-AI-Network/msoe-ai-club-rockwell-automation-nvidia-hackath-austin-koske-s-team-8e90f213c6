# Independent annotation review

Python **3.11+**, a browser, the existing dataset and saved prelabels are enough.
The editor uses the Python standard library. No Austin-specific virtual environment,
GPU, API credential, Node build or new labeling request is needed.

## Portable launch

Fetch and check out the `annotation-review` branch. Extract `labels-only.zip` from
the existing October 18 handoff into a directory of your choice. It must contain
`manifest.json` and camera folders with `classes.txt` and `annotations.json`.
The original bulk output directory is accepted too.

From the repository root, Windows PowerShell or Linux/macOS:

```text
python -m annotation_review serve --dataset /path/to/csi-vial-fill-dataset --prelabels /path/to/extracted-labels --workspace /path/to/my-separate-review --port 8775
```

Open **http://127.0.0.1:8775**. Replace all paths; paths with spaces need quotes.
Windows users can use `py -3.12` instead of `python`. Stop with Ctrl+C.
Use one workspace per reviewer. The dataset and prelabel input folders remain
read-only. Workspaces inside either source folder are refused.

For a named installed command in an isolated environment:

```text
python -m venv .venv-review
# Windows: .venv-review/Scripts/python -m pip install ./annotation_review
# Linux/macOS: .venv-review/bin/python -m pip install ./annotation_review
```

The installed `annotation-review` command takes the same arguments. Core runtime
dependencies are empty; `pyproject.toml` defines optional `video` and `legacy`
extras. The pre-existing generation scripts are archived in `legacy_prelabel/`;
their imports require the optional legacy packages. They are **not executed by
the review app**. Do not run them to reproduce the review interface.

## Review, correct, export, reopen

1. Choose Run 3 or a supplied initial cases queue. Tank and cap/seal work is a
   later milestone; all existing prelabels remain available under All cameras.
2. Select your own reviewer name: Aleyna, Zoe or Reagan. Identity is self-reported;
   this loopback app is not an authenticated multi-user service.
3. Select a machine target. Gold dashed boxes always retain the original machine
   geometry. Presence/confidence are displayed for context and never approve it.
4. Accept the original, drag/numerically enter a changed box, reject it, or mark it
   uncertain with a note. Add a missing target when no candidate exists. Use
   **Save explicit decision**; unsaved changes are not exported.
5. **Export corrections** downloads a JSON with original sources/hashes, machine
   suggestions, proposed boxes, status, reviewer, notes and append-only history.
6. In a fresh workspace, use **Reopen correction JSON**, or import from the CLI.
   Source/candidate mismatches and divergent histories are rejected atomically.

```text
python -m annotation_review export --dataset DATA --prelabels LABELS --workspace WORK --file corrections.json
python -m annotation_review import --dataset DATA --prelabels LABELS --workspace FRESH_WORK --file corrections.json
```

Coordinates are `[left, top, right, bottom]`, normalized to the **original full
image**. IDs are local to each camera's class list. A rejected box has a null
proposed box. Originals are never overwritten. Review history is append-only;
optimistic revision checks prevent silently overwriting another edit.

Use `--demo` with a separate workspace for demonstrations. Demonstration revisions
are explicitly ineligible as reviewed labels and cannot import into real review
workspaces. Machine suggestions remain visibly unreviewed until a person acts.

## Time and ownership boundaries

Still capture timestamps and MP4 clocks are distinct. By default still/video
associations are **unmatched**; no `wall_seconds * 30` join exists in this tool.
Spatial corrections never assign vial IDs, recipes, measurement validity or fill
timing. Reviewed recipe/timing CSVs remain separate data-team inputs, using the
replay's documented contract:

```csv
vial,recipe_id,t100_ml,t200_ml,t400_ml,dispense_order,start_frame,end_frame
```

Frame indices there refer to zero-based decoded source MP4 frames, inclusive end.
This app does not infer those fields. Replay diagnostics enter through a versioned
external cases adapter, never by importing or editing replay internals. No color
thresholds, training, synthetic generation or shared compute setup are changed.

## Checks

```text
python -m unittest discover -s annotation_review/tests -v
```

Tests cover immutable suggestions, source mismatch rejection, correction round
trips, conflicting revisions, explicit acceptance, uncertain notes and demo
separation. Passing these establishes software/provenance behavior, not label
accuracy.
