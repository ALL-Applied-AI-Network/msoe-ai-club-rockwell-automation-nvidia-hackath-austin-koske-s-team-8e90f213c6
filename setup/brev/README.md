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
independent lighting and fill: the clean intended mix (`-ref`) and the vial with residue fraction
0–30% of the previous recipe (`-via`, residue 0 for a third of them). A quarter of the families are
held out as development families. Labels are the mixing parameters, not optical truth.

**GB10 test, 2026-10-29:** 24 cases from 3 families at 32 spp and 4 subframes rendered in 106 s
(about 3.3 s per render after Isaac startup) on GB10 Isaac 6.0.0.1, and passed `verification.json`.
Review sheets: `contact-rgb.png`, `contact-boxes.png`, `contact-masks.png` in the test output
(`~/carryover-20261029/render-test` on spark8).

On the Brev instance, after copying the repository and the supplied twin assets:

```bash
export OMNI_KIT_ACCEPT_EULA=YES                # after accepting the Omniverse license
export ISAAC_PY=/isaac-sim/python.sh           # the launchable's Isaac Python, if elsewhere
bash setup/brev/render_carryover.sh ~/carryover-v1 /path/to/csi_digital_twin_v1 400 8 400
```

The defaults (400 families, 8 changeovers) give 6,400 renders in 16 shards of 400; at the GB10
rate that is about 6 GPU-hours. Shards are spread over every visible GPU. Rerunning the same
command skips shards whose `verification.json` passed, so a stopped instance resumes. When it
finishes it prints renders and GPU-hours from `timing.jsonl`; copy the output folder (including
`timing.jsonl` and `logs/`) back, record the console cost, and stop the instance.
