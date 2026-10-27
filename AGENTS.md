# Repository instructions

Read `docs/architecture.md`, `docs/development.md` and `docs/data.md` before
changing the pipeline. Component status is documented there; do not assume the
app skeleton runs inference.

- Work on the requested branch. Do not merge or push to `main` unless explicitly
  requested. Preserve unrelated working-tree edits, datasets and results.
- Do not spawn subagents unless explicitly authorized.
- Keep model experiments in `setup/vision/`, measurement/reference logic in
  `replay/`, annotation tools in `annotation_review/`, and the app in `api/` / `client/`.
- Train on synthetic data only. Runs 1–4 are real development data (transfer checks,
  no training). Run5 is the ground truth: never fit, tune or select on it (see
  `docs/plan.md#data-splits`). Keep scene families,
  sessions and batches together and disclose generic pretraining.
- Preserve source labels, reviewer histories, authored scenes and frozen helper
  bytes. Human review establishes label/reference/timing decisions.
- Reuse measurement QA. Keep proposals, sample eligibility, qualified recipe
  evidence and operator responses separate. Preserve original RGB and clocks.
- Inspect processes, job states and completion markers before restarting an
  experiment. GPU work uses the approved remote host; Rosie uses Slurm and
  Singularity 3.10. Do not connect industrial networks, cameras or equipment.
- Keep maintained documentation in `docs/`. Update architecture and evaluation
  pages when behavior or measured results change. Historical JSON records retain
  their original paths/hashes; do not rewrite them to imitate a new execution.
- For experiment review, provide local MP4/PNG paths. Do not launch a preview
  server or browser unless requested.
