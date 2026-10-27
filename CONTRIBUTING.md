# Contributing

Start with [the architecture](docs/architecture.md) and
[development commands](docs/development.md). Work on a branch and review the
diff against `main`. Keep unrelated experiments and local data out of the change.

The model worker must preserve source identity, native timing, full-image
coordinates and reference state. Reuse the existing measurement QA. Model
confidence must not bypass sample or recipe qualification. Changes to those
boundaries need corresponding examples and evaluation.

New CSI-domain training and model selection use synthetic data. Real data is
for evaluation; Run5 stays outside fitting, tuning and sampling decisions.
Preserve labels, reviewer histories, scene assets and frozen experiment helpers.
Publish new weights/data as versioned release assets with checksums.

Srijan owns model integration; Josiah/Hayden own the backend and operator UI.
Coordinate changes across their observation interface. Human reviewers own
label, reference and timing decisions. GPU experiments run on the approved
remote host; inspect existing jobs before launching another.

Run the checks relevant to the changed component. For documentation changes,
check paths, links and commands against the code. Keep measurement results in
verification records and summarize their scope in [evaluation](docs/evaluation.md).
