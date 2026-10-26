# Working context

Read [HANDOFF.md](HANDOFF.md) for current status, artifact locations and next work.
Use this repository's primary checkout on `austin-work`; both former agent branches are
integrated here. Use it for follow-up work unless Austin requests another branch
or worktree. Do not spawn subagents unless explicitly authorized.
Austin authorized only the team/data handoff on `main` (published as `1cb6a63`).
October23 explicit exception: Austin also authorized the frontend/API starter
merge under root`api/` and`client/`, published as`1ec7cbc` (PR1). This does not
authorize merging experiments or other application development into`main`.
October26 explicit exception: Austin authorized the temporal CNN model/code,
synthetic training assets and updated architecture/integration documents on
`main`. Use the scoped `team_handoff/temporal_cnn/` release; do not merge the
whole development branch or unrelated application/experiment work into `main`.
Keep experiments, application changes and follow-up development on `austin-work`;
other than these explicit handoffs, do not push or merge those into `main`.

- Inspect existing processes, scheduler jobs and completion markers before
  continuing an experiment. Do not duplicate training or completed evaluations.
- Keep annotation tooling in `annotation_review/`, replay in `replay/`, and
  detector experiments in `setup/vision/`. Use the public QA export contract
  between review and replay; preserve app defaults and measurement QA.
- October 22 assignments from Austin supersede earlier proposed workstreams:
  Josiah/Hayden own the operator dashboard/frontend and backend pipeline;
  Srijan owns combining the models into a standard pipeline; Zoe/Alayna/Reagan
  investigate NVIDIA tools (Cosmos, etc.); Cameron's current delivery is SAM2.1.
  Existing label/timing approval remains human work; preserve authored synthetic
  assets and coordinate scene changes with their authors.
  Machine prelabels and vial-only negatives are provisional;
  demonstration corrections cannot approve team labels.
- Keep Run 5 out of fitting, checkpoint selection, threshold tuning and sampling
  decisions. Its inspected teacher-agreement scores are diagnostic evidence.
- Austin reaffirmed October 22: new CSI-domain training must use synthetic data.
  Real imagery is inspection/evaluation evidence, not a fitting set. Label legacy
  real-trained models as development comparators; do not present them as the
  synthetic-trained competition solution. Freeze checkpoints using synthetic
  criteria before real transfer scoring, and disclose generic pretraining.
- Rosie uses Singularity 3.10 and GPU jobs through Slurm. Preserve frozen helper
  snapshots in experiment directories. Do not connect industrial equipment,
  networks or cameras.
- Preserve local datasets, drafts, reviewer workspaces and results. Stage named
  files; never sweep those into commits or delete them for a clean Git status.
- For experiment review, provide local MP4/PNG paths. Do not launch a browser or
  preview server unless requested. The annotation editor remains available for
  human review when requested.
