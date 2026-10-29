# Compute setup

Model-only CPU inference needs no remote account. Use the
[development guide](development.md) for that path. The launchers below support
data preparation, rendering and GPU experiments; they do not install the full
operator pipeline.

## Windows data environment

With Python 3.12 and PowerShell 7, run from the repository:

```powershell
pwsh -File setup/bootstrap.ps1
& "$HOME/ai-club-tools/envs/data-windows/Scripts/python.exe" setup/check_dataset.py --data "C:/path/to/csi-vial-fill-dataset"
```

The bootstrap creates a separate environment under `~/ai-club-tools`.
The dataset check reads the supplied collection and decodes one frame per
video; it does not validate labels or timing. Keep Isaac's runtime separate.

## GB10

Use your own SSH key and the configured campus connection. The examples in
[SSH config](../setup/ssh/config.example) name the `msoe-spark` alias.
`setup/ssh/add-spark-key.ps1` installs a public key when account access is available.
Never copy private keys or shared credentials into the repository.

For a new host, `setup/bootstrap.sh` provides `data`, `isaac` and `cosmos`
profiles. On an existing shared host, inspect the installed environments before
changing them. The checked runtime versions are recorded in
[models.json](../setup/models.json); these are historical pins, not a live host inventory.

From the deployed setup directory, typically `~/ai-club-onboarding`:

```bash
export TEAM_MEMBER=your-name
bash setup/spark/run.sh data setup/doctor.py --profile data
```

The Spark runner coordinates GPU work with a lock. Use it for shared-host jobs
and check existing processes/completion records before starting another run.
If the hostname fails to resolve, check campus/VPN connectivity first.

For an Isaac installation after accepting its license:

```bash
export OMNI_KIT_ACCEPT_EULA=YES
source setup/env.sh
bash setup/spark/run.sh isaac setup/isaac/render_smoke.py --out "$AI_CLUB_HOME/results/$TEAM_MEMBER/isaac-smoke-01"
```

The pinned ARM launcher preloads `libgomp` and uses `setarch aarch64 -R` for the
Isaac process, a workaround for the tested build's startup hang. It does not
change the host setting. `ISAAC_DISABLE_ASLR=0` disables that workaround.
The project renderer additionally needs the external twin/assets configured
by `setup/isaac/render_project.sh`. A smoke render is not a training dataset
or optical calibration.

## Rosie

Rosie uses your university account, Slurm and **Singularity 3.10**. Do not run
GPU work on login nodes or substitute Docker for the provided SIF workflow.

From Windows:

```powershell
pwsh -File setup/deploy.ps1 -Target Rosie
ssh msoe-bastion
```

On Rosie:

```bash
cd ~/ai-club-onboarding
bash setup/bootstrap.sh rosie
bash setup/rosie/submit.sh smoke
squeue -u "$USER"
```

`submit.sh job path/to/script.py` submits a project GPU job;
`submit.sh data path/to/script.py` selects the data environment. Logs go under
`~/ai-club-tools/logs`. The container path is configured in
`setup/config.env.example`; per-user environments layer over the SIF without
modifying it. Inspect the job exit and output before retrying.

## Optional tools and deployment helper

Cosmos is research tooling, not a required liquid inference stage.
`setup/cosmos/reason.py` runs a pinned local model; `setup/cosmos/api.py` uses
an externally supplied NVIDIA API key. Its `--dry-run` validates a request
without sending it. Keys and gated-model access belong to the user account.
See [the original resource catalog](archive/resources-20261016.md) and
[Brev setup](../setup/brev/README.md) for additional tooling.

`setup/deploy.ps1 -Target Spark` copies setup scripts, excluding private config,
logs and caches. It does not clone all application/model code or supply a camera
endpoint. `-IncludeTwin` requires the separately available project assets.
Keep environments, downloads and generated results outside Git.

## Compute and synthetic data used (as of 2026-10-29)

**GPU hours: about 6.6 in total**, measured on 2026-10-29 over the campus VPN.

| Where | What ran | GPU-hours |
| --- | --- | --- |
| Rosie (Slurm, H100) | 22 GPU jobs on account `koskea`, 2026-10-16 to 10-20: GPU checks, Cosmos trial, detector bootstrap, liquid segmentation, real+synthetic and sim-only training, epoch sweep, liquid-specialist and early-liquid arrays. Plus 4 CPU-only jobs (CSI history, artifact read) | **2.65** (from `sacct`; 2.81 h wall time including CPU jobs) |
| GB10 (spark8) | Isaac renders, Cosmos transfer, synthetic iterations, dataset1 render/train runs, station benchmark, VLM RCA, forger-prep training | **about 4** (3.95 h of merged job-log time, 2026-10-16 to 10-28) |
| Laptop (RTX 4080 SUPER) | Inference, line-runner replays, live-camera tests (24 ms per model sample) | Not logged; minutes to tens of minutes per replay |

How the numbers were taken:

- **Rosie** is exact: Slurm elapsed time times allocated GPUs, for the hackathon jobs only. The
  same account's unrelated jobs (`urcv-*`, `foxwatch-*`, `home-cleanup-*`) are excluded. Job IDs
  found in the result logs that belong to other users are not counted; teammates' own Rosie
  jobs, if any, are not included.
- **GB10** is an estimate. GPU accounting is disabled there, so each job log's creation and last
  write times are used as its start and end, overlapping logs are merged, and install, download
  and long-lived daemon/Omniverse logs are left out. It is an upper bound on GPU-busy time
  (a job is not on the GPU for all of its run) and misses jobs that wrote no log.
- Of the 9 YOLO11n runs with an Ultralytics timing line, training took 1.06 GPU-hours combined;
  those are inside the totals above.

The models are small and quick to train; the time is split roughly evenly between training
(mostly Rosie) and rendering, Cosmos and VLM experiments (the GB10). To recompute Rosie:

```bash
sacct -X -u "$USER" -S 2026-10-01 -E now --format=JobID,JobName%30,Elapsed,AllocTRES%50,State -P
sacct -X -u "$USER" -S 2026-10-01 -E now --format=ElapsedRaw,AllocTRES -P -n \
  | awk -F'|' '{ g = 0; if (match($2, /gres\/gpu=[0-9]+/)) g = substr($2, RSTART + 9, RLENGTH - 9); s += $1 * g } END { printf "%.1f GPU-hours\n", s / 3600 }'
```

The second command counts every job on the account since the start date; the 2.65 above
excludes the four `foxwatch-*` jobs (0.05 GPU-hours).

**Synthetic data: not a large amount.** Several hundred unique renders, with experiments
reusing them through seeded repeats and augmentation:

| Use | Synthetic data |
| --- | --- |
| Shipped liquid U-Net (`temporal-cnn-training-v1.zip`) | 192 paired renders and masks: 144 training images from 36 scene families, 48 development images from 12 disjoint families |
| Revised-render detector experiment | 18 unique renders, repeated to 768 (mixed with 1,442 real) or 2,210 (sim-only) training exposures |
| Faster R-CNN synthetic iterations (GB10) | 614 synthetic samples per epoch; 154 and 202 synthetic development images |
| Diagnosis pilot | 1,620 synthetic four-vial sequences (feature scenarios, not rendered images) |

The synthetic experiment folders hold about 10 GB and 30,000 image files, but most of those are
real frames, masks, overlays and review panels, not distinct renders.
