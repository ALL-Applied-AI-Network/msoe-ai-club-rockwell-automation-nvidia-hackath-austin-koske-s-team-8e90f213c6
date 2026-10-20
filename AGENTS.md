# Working context

Read [HANDOFF.md](HANDOFF.md) for current status, artifact locations and next work.
Use this repository's primary checkout on `austin-work`; both former agent branches are
integrated here. Use it for follow-up work unless Austin requests another branch
or worktree. Do not spawn subagents unless explicitly authorized.
Austin explicitly requested that `main` not be pushed. Keep integration and
follow-up commits on `austin-work`; do not push or merge them into `main`.

- Inspect existing processes, scheduler jobs and completion markers before
  continuing an experiment. Do not duplicate training or completed evaluations.
- Keep annotation tooling in `annotation_review/`, replay in `replay/`, and
  detector experiments in `setup/vision/`. Use the public QA export contract
  between review and replay; preserve app defaults and measurement QA.
- Aleyna, Zoe and Reagan own approved labels and timing. Cameron and Hayden own
  synthetic examples. Machine prelabels and vial-only negatives are provisional;
  demonstration corrections cannot approve team labels.
- Keep Run 5 out of fitting, checkpoint selection, threshold tuning and sampling
  decisions. Its inspected teacher-agreement scores are diagnostic evidence.
- Rosie uses Singularity 3.10 and GPU jobs through Slurm. Preserve frozen helper
  snapshots in experiment directories. Do not connect industrial equipment,
  networks or cameras.
- Preserve local datasets, drafts, reviewer workspaces and results. Stage named
  files; never sweep those into commits or delete them for a clean Git status.
- For experiment review, provide local MP4/PNG paths. Do not launch a browser or
  preview server unless requested. The annotation editor remains available for
  human review when requested.
