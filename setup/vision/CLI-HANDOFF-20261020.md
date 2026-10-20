# Model continuation — October 20, 2026

Use the primary checkout on **austin-work**. Do not push or merge into main.
The [consolidated handoff](../../HANDOFF.md) is the current entry point.

## Completed work

[Four-arm liquid-specialist comparison](LIQUID-SPECIALISTS.md) is complete.
Slurm array 286264 finished successfully. DINO completed 25 epochs and selected
15. Evaluation, assembly, local PNG review and full decoding of all 12 delivered
MP4s are done. Do not resubmit or rerun completed evaluations.

Local root: C:/Users/Austin/ai-club-tools/results/liquid-specialists-20261020.
See evaluation/comparison.json and evaluation/verification.json. The exact
executed evaluation/collection/verification scripts are retained there.
Repository evidence: setup/verification/liquid-specialists-20261020.json.

DINO liquid AP50–95 is 51.34%, precision 80.19%, recall 93.53% at unchanged .35,
against unreviewed Run 5 teacher boxes. The deployed baseline remains in place;
measurement QA is unchanged. The inspected Run 5 is not a blind final test.

## Active iteration

Austin's target is liquid visible at the bottom as soon as it exists. The
[early-liquid protocol](EARLY-LIQUID.md) defines the next three fits and exact
local evaluation command. Slurm array **286291** has arms:

- 0: yolo11n-960
- 1: yolo11s-640
- 2: fasterrcnn-resnet50

Remote root: $HOME/ai-club-tools/results/early-liquid-20261020.
Local root: C:/Users/Austin/ai-club-tools/results/early-liquid-20261020.
Inspect jobs/logs first using the existing Windows SSH configuration:

```powershell
ssh -o BatchMode=yes -o GSSAPIAuthentication=no -o ConnectTimeout=12 msoe-bastion 'sacct -X -j 286291 -n -P -o JobID,State,ExitCode,Elapsed,NodeList'
```

The detached collector writes collection.log, scheduler-status.txt and, after
three successful exits, completed-artifacts.tar.gz plus its checksum. Wait for
the checksum, complete each transfer, and verify the archive before extraction.
Never overwrite different local artifacts. Do not replace training jobs after
a VPN disconnect; they survive it. Partial outputs require inspection and fresh
output directories for any necessary rerun.

After retrieval, use vision-alternatives-windows for evaluate_early_liquid.py.
It compares the new fits with the completed Run 4 development audit, preserving
per-family input resolution and reporting short-target recall separately.
There are no development targets <=32 pixels tall; first-visible-pool accuracy
is currently unmeasured. Human label/timing work remains with the data team.

## Boundaries

Runs 1–3 train; Run 4 develops; Run 5 never chooses training/threshold/sampling
settings. Source labels, frozen helper snapshots, app defaults and QA remain
intact. Rosie GPU work uses Slurm/Singularity 3.10. No industrial connection,
automatic adjudication, synthetic rendering or automatic model promotion.
Provide local MP4/PNG paths; do not launch browser previews. No subagents unless
Austin explicitly authorizes them. See AGENTS.md for repository-wide context.
