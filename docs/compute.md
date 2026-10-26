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
