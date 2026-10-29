# Brev fallback for rendering

Use the team's Brev organization and its approved credits. Provisioning requires
the team's signed-in Brev account; this setup does not start a paid instance.

1. Open the [Isaac Sim and Isaac Lab launchable](https://brev.nvidia.com/launchable/deploy/now?launchableID=env-35JP2ywERLgqtD0b0MIeK1HnF46).
2. Prefer an available RTX PRO 6000; L40S is the guide's fallback. Check the live
   price and choose a provider/environment with Start and Stop support.
3. Use the instance name and SSH connection shown by Brev. Wait for provisioning
   in the Compute view if the launch screen does not update.
4. Copy the repository/setup and permitted twin assets onto the instance. Check
   the preinstalled Isaac version before installing a second copy.
5. If the launchable supplies `python.sh`, use it directly for
   `setup/isaac/render_smoke.py`, then the existing project renderer. Set
   `OMNI_KIT_ACCEPT_EULA=YES` after accepting the Omniverse license.
6. Record GPU, version, run duration and console cost with the result. Copy code
   and results back, then stop the instance. Stopped storage can still incur cost.

The Cosmos generation launchable is a different, larger workload. Choose it only
for an actual generation experiment, not the small reasoner test. This folder's
Rosie launchers always use Singularity, even though Brev has other runtime options.

Source: [official hackathon compute guide](https://github.com/MSOE-AI-Club/maic-content/blob/main/unlisted/Rockwell-NVIDIA-Fall-2026-Compute/Rockwell-NVIDIA-Fall-2026-Compute.md).

## Carryover bulk render

`render_carryover.sh` renders the synthetic carryover set from
`setup/isaac/make_carryover_fixture.py`. Each scene family draws three random dyes, so no CSI dye,
recipe reference or Run 5 value is used. Each previous-to-next changeover is rendered twice under
independent lighting and fill: the clean intended mix (`-refe`), the clean previous mix (`-prev`,
the direction the FLUSH rule compares against) and the vial with residue fraction 0–30% of the
previous recipe (`-vial`, residue 0 for a third of them). The GB10 test below predates `-prev`. A quarter of the families are
held out as development families. Labels are the mixing parameters, not optical truth.

**GB10 test, 2026-10-29:** 24 cases from 3 families at 32 spp and 4 subframes rendered in 106 s
(about 3.3 s per render after Isaac startup) on GB10 Isaac 6.0.0.1, and passed `verification.json`.
Review sheets: `contact-rgb.png`, `contact-boxes.png`, `contact-masks.png` in the test output
(`~/carryover-20261029/render-test` on spark8).

**Brev, 2026-10-29:** the Isaac launchable (`env-35JP2ywERLgqtD0b0MIeK1HnF46`, g6e.4xlarge, one
L40S, $3.61/h, stoppable) ships Isaac Sim 6.0.1 as the `nvcr.io/nvidia/isaac-sim:6.0.1` container,
not on the host. `run-in-container.sh` runs the script below in it. Two things failed first: with
`docker run --gpus all` the graphics capability is dropped and the path tracer returns a blank
frame or hangs in `orchestrator.step` at 0% GPU; the nvidia runtime fixes it. The 24-case smoke
then passed verification in 94 s including startup (about 2.2 s per render). `brev exec` kills
background processes when it returns, so long runs go in `tmux`.

```bash
brev copy setup-and-assets.tgz linewatch-carryover:/home/ubuntu/   # setup/ and datasets/mixing_twin/...
brev exec linewatch-carryover 'tmux new-session -d -s bulk "bash ~/linewatch/setup/brev/run-in-container.sh carryover-v1 400 8 400"'
```

Directly on a machine with Isaac installed, after copying the repository and the supplied twin assets:

```bash
export OMNI_KIT_ACCEPT_EULA=YES                # after accepting the Omniverse license
export ISAAC_PY=/isaac-sim/python.sh           # the launchable's Isaac Python, if elsewhere
bash setup/brev/render_carryover.sh ~/carryover-v1 /path/to/csi_digital_twin_v1 400 8 400
```

The defaults (400 families, 8 changeovers) give 9,600 renders in 24 shards of 400; at the GB10
rate that is about 9 GPU-hours. Shards are spread over every visible GPU. Rerunning the same
command skips shards whose `verification.json` passed, so a stopped instance resumes. When it
finishes it prints renders and GPU-hours from `timing.jsonl`; copy the output folder (including
`timing.jsonl` and `logs/`) back, record the console cost, and stop the instance.

Then, on the copied renders (`renders/shard-*` folders):

```bash
python -m replay.carryover_synthetic ~/carryover-v1/renders/shard-* --out carryover-eval.json
cd REPO && sbatch setup/vision/train-carryover-rosie.sbatch OUT_ROOT ~/carryover-v1/renders/shard-*
```

The first tests the FLUSH rule and the Lab-projection residue estimate (thresholds swept on train
families, reported on dev families); the second trains three seeds of a learned estimator on
Rosie for comparison with the projection.

### Pre-bulk tests (2026-10-29)

- **v1 fixture, first 400 renders:** clean vials (no residue) measured a median 8.9 dE2000 from
  their own reference; 96% of that tracked the independently drawn +-40% lighting. The bulk run
  was stopped at 449 renders and the fixture changed so a changeover's three renders share
  lighting and fill, with a labelled 5-15% drift arm.
- **v2 fixture, 480 renders (20 families, 160 changeovers, 0.27 L40S GPU-hours):** clean vials
  0.68 dE2000 median (95th percentile 1.20) from their reference. Median dE2000 by residue:
  1.2 at 5-10%, 1.8-2.2 at 10-20%, 2.8-4.5 at 20-30%. The current rule (band 5, cos 0.8)
  flagged 17% of dev-family carryover vials with no false alarms; thresholds swept on train
  families (band 2, cos 0.5) flagged 52% of dev carryover vials, still with no false alarms on 17
  clean dev vials. The Lab-projection residue estimate: 0.074 mean absolute error on
  carryover vials. A 30-epoch CPU run of the learned estimator on 120 train changeovers reached
  0.084 dev MAE against 0.024 for the projection (dev MAE over all vials) — it learns, but needs
  the bulk set. Synthetic renders only; renderer colour shifts are not optical truth, and the
  small dev counts make these rates indicative.
- Output folders: `~/ai-club-tools/results/carryover-v1-brev/test` (v1) and `.../test2` (v2).
