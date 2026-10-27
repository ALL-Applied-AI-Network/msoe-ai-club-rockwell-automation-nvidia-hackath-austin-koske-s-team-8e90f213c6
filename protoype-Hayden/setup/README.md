# Team setup

Start here. Choose the environment for your assignment; you do not need every tool.
Data review runs on a laptop or the GB10. Rendering runs on the GB10 or an RTX Brev
machine. Rosie runs GPU learning and Cosmos inference through **Singularity 3.10
and Slurm**. No Rosie command in this folder uses Docker.

The setup has been deployed to `~/ai-club-onboarding` on Spark 8 and Austin's
Rosie account. Software, logs and outputs live under `~/ai-club-tools`, outside
Git. Hugging Face models use `HF_HOME` (default `~/.cache/huggingface`).
See [verification](verification/README.md) for executed checks and their scope.

**Current handoff:** Windows, GB10 and Rosie data checks passed. GB10 Isaac
rendered all 12 fixture examples with visible color/height changes. Cosmos3-Edge
image inference passed on both GB10 and Rosie H100; Rosie's Slurm/Singularity
GPU check also passed. The final pinned environments were rechecked successfully.

## Pick your starting point

| People | Start with | First successful result |
| --- | --- | --- |
| Aleyna, Zoe, Reagan | Laptop data environment or GB10 data profile | All data checks pass; open a recording using existing project code |
| Cameron, Hayden | GB10 Isaac profile | Render the setup scene, then one vial from the existing project |
| Srithej, Josiah | This README, resource catalog and CSI request in `docs/` | Know which machine and camera access the visit needs |
| Austin | Rosie GPU and Cosmos profiles | Slurm GPU check and image-to-text output |

## Data work on Windows

Install Python 3.12 if it is not available. From the team repository, using
PowerShell 7:

```powershell
pwsh -File setup/bootstrap.ps1
```

This creates `~/ai-club-tools/envs/data-windows`, leaving system Python alone.
Use that environment's `Scripts/python.exe` for data work. Isaac has its own
runtime; do not install these packages into an existing Isaac installation.

Once you have the supplied dataset, run `setup/check_dataset.py` with that
interpreter. It decodes one frame from each of the 15 videos and checks access
to the Run 5 label table, without modifying labels. Use `--data PATH` if your
download is stored elsewhere. This input check does not certify timing or labels.

On Linux, from the repository:

```bash
bash setup/bootstrap.sh data
bash setup/spark/run.sh data setup/doctor.py --profile data
```

## Connect to the GB10

Each teammate uses their own SSH key. Austin has configured his own key; do not
copy his private key. Get the shared account credential from Austin privately if
you need to install your public key. This helper adds a key without changing the
account password:

```powershell
pwsh -File setup/ssh/add-spark-key.ps1
```

Add the Spark block from [ssh/config.example](ssh/config.example) to your own
SSH configuration. Then:

```powershell
ssh msoe-spark
```

On the GB10, the shared starter folder is already deployed:

```bash
cd ~/ai-club-onboarding
export TEAM_MEMBER=cameron  # substitute your own name
bash setup/spark/run.sh data setup/doctor.py --profile data
```

Set your name in `setup/config.env` if you want it to persist for this copy. Start
from `config.env.example`. On the shared account, coordinate environment updates
with Austin; ordinary work uses the installed environments. The Spark runner
locks GPU work so two team commands do not compete on the same GB10. It returns
a clear busy message rather than starting another job.

## Render with Isaac Sim

Isaac Sim is pinned to **6.0.0.1** with a separate Python 3.12 environment. New
machines can install it using `bash setup/bootstrap.sh isaac`. The GB10 has
already been provisioned. The ARM install uses managed Python because its
development headers are needed by a dependency.

The ARM launcher preloads `libgomp` and uses `setarch aarch64 -R` for the Isaac
process. Repeated native launches otherwise hung before Kit logging on this
GB10; disabling address randomization for the process resolved that startup
failure. It does not change the host's system setting. `ISAAC_DISABLE_ASLR=0`
allows a future runtime to be tested without this workaround.

Read the [Omniverse license](https://docs.omniverse.nvidia.com/platform/latest/common/NVIDIA_Omniverse_License_Agreement.html)
and set the acceptance variable when using Isaac:

```bash
cd ~/ai-club-onboarding
export TEAM_MEMBER=cameron
export OMNI_KIT_ACCEPT_EULA=YES
source setup/env.sh
bash setup/spark/run.sh isaac setup/isaac/render_smoke.py \
  --out "$AI_CLUB_HOME/results/$TEAM_MEMBER/isaac-smoke"
```

The first launch compiles shaders and may take several minutes. Headless display
warnings do not by themselves mean failure. Check the saved image and `result.json`.

To render **one existing vial example**:

```bash
bash setup/isaac/render_project.sh
```

The project launcher reuses the existing renderer and the original 12-record
smoke fixture through an Isaac 6.0 compatibility adapter. It creates a temporary
liquid proxy at the frozen world pose, avoiding the source liquid parent's
animation that left all GB10 captures empty. Source code, scene files and existing
training artifacts stay untouched. An optional second argument controls count;
the first is a fresh output directory:

```bash
bash setup/isaac/render_project.sh "$AI_CLUB_HOME/results/$TEAM_MEMBER/vials-001" 12
```

An optional third argument supplies your own sample JSONL in the existing twin
schema. Keep the small fixture unchanged as a repeatable setup check.

From Windows, copy output back with `scp -r`. The remote output path is the one
shown by the launcher. These images test software and geometry; liquid color and
process physics remain uncalibrated.

For multiple examples, `capture_check.json` verifies that strongly different
recipe colors produce visibly different captures. A one-image run cannot check
variation; use the 12-example command for that check. This guards against the
stale-image bug found during onboarding, without certifying optical accuracy.

## Run Cosmos on the GB10

The default is **Cosmos3-Edge** for a small image-to-text check. The tested
Transformers version exposes its Edge implementation. Model revisions are pinned
in the runner and recorded in [models.json](models.json). The first model run
downloads weights; subsequent runs use the local cache.

```bash
cd ~/ai-club-onboarding
source setup/env.sh
bash setup/spark/run.sh data setup/make_example.py \
  --out "$AI_CLUB_HOME/results/$TEAM_MEMBER/example.png"
bash setup/spark/run.sh cosmos setup/cosmos/reason.py \
  --image "$AI_CLUB_HOME/results/$TEAM_MEMBER/example.png" \
  --out "$AI_CLUB_HOME/results/$TEAM_MEMBER/cosmos.json"
```

For a new compatible GB10, install using `bash setup/bootstrap.sh cosmos`. The
native runtime uses CUDA 13 PyTorch 2.11.0 and torchvision 0.26.0. A different
GPU/driver may require a different wheel backend; run the doctor before inference.

Nano is also exposed using `--model nvidia/Cosmos3-Nano`. Its reasoner weights
are about 29 GiB, so choose it deliberately. Nano model loading and task quality
are separate checks from the executed Edge smoke. This starter accepts one image;
multi-image/video inference can be added following the official cookbook.

## Run on Rosie with Singularity

Use **your own** university account and Rosie SSH key. The login node's Python
3.8 is not the environment for this project. The site PyTorch SIF supplies Python
3.12 and CUDA PyTorch; the bootstrap creates a writable per-user virtual
environment on top of it. A second isolated environment holds the modern data
packages, because the site CUDA image constrains its own NumPy version. It does
not alter the SIF or system packages.

First copy this setup into your own account from Windows:

```powershell
pwsh -File setup/deploy.ps1 -Target Rosie
ssh msoe-bastion
```

Then, on Rosie:

```bash
cd ~/ai-club-onboarding
bash setup/bootstrap.sh rosie
bash setup/rosie/submit.sh smoke
```

The last command prints a job ID. Check `squeue -u "$USER"`; job logs are in
`~/ai-club-tools/logs`. The small job requests one H100, two CPUs, 16 GiB RAM and
five minutes. GPU computations are submitted to Slurm, not run on the login node.

For Cosmos:

```bash
source setup/env.sh
bash setup/rosie/run.sh --cpu setup/make_example.py \
  --out "$AI_CLUB_HOME/results/$TEAM_MEMBER/example.png"
bash setup/rosie/submit.sh cosmos \
  --image "$AI_CLUB_HOME/results/$TEAM_MEMBER/example.png" \
  --out "$AI_CLUB_HOME/results/$TEAM_MEMBER/cosmos-rosie.json"
```

For a project Python GPU job:

```bash
bash setup/rosie/submit.sh job path/to/your_script.py --your-argument value
```

For a CPU data-processing job, use `bash setup/rosie/submit.sh data script.py`.
The `--data` runner selects the isolated data environment; `--cpu` selects the
Cosmos environment without GPU passthrough for light setup checks.

The site image is `/data/containers/msoe-pytorch-25.05-py3.sif`. Override
`ROSIE_PYTORCH_SIF` only if the site changes it. Some image startup scripts emit
`not a valid test operator` warnings; the executed import/GPU checks determine
success. We do not depend on the broken login-node Snap `uv` installation.

H100/V100/A100 memory capacity does not make those GPUs suitable for RTX
rendering. Use them for CUDA learning/inference; use the GB10 or Brev RTX for
Isaac camera images. The inspected Rosie partitions currently expose T4, V100
and H100 resources; an A100 queue was not listed in that inventory.

## Hosted Cosmos without a local model

Create your own [NVIDIA API key](https://build.nvidia.com/settings/api-keys), export
`NVIDIA_API_KEY` in your shell, then run the standard-library client:

```bash
python setup/cosmos/api.py --image path/to/image.png --out response.json
```

`--dry-run` checks the payload without credentials or a network request. Hosted
model access and API limits belong to each user's NVIDIA account. The local Edge
runner needs no hosted API key. If a Hugging Face repository requires access,
authenticate in your own account; never copy a teammate's token into the repo.

## Keep a fresh copy of setup on a remote machine

```powershell
pwsh -File setup/deploy.ps1 -Target Spark
# Include the existing fill-station assets and renderer when needed:
pwsh -File setup/deploy.ps1 -Target Spark -IncludeTwin
```

Deployment excludes private config, logs, caches and generated results. It copies
the setup into `~/ai-club-onboarding`; it does not clone the entire Git repository.
The optional twin payload is about 26 MiB plus code. It is present in the tested
Spark copy. For full project development, clone/share the source separately and
run these same scripts from that checkout.

## Shared conventions

Keep raw data unchanged. Point data code at the canonical dataset location;
installation does not imply the full dataset has been copied to a machine.
Put new results under `AI_CLUB_HOME/results/TEAM_MEMBER`, with a fresh run name.
Record inputs, versions, model revision, GPU and elapsed time with every result.
Keep generated data, environments, model weights and credentials outside Git.

The [resource catalog](resources.md) covers every tool category from the hackathon
guide and explains the remaining account, hardware and setup requirements.
The [Brev guide](brev/README.md) gives the RTX launchable path without starting a
billable instance automatically. Provisioning is complete only where the
verification notes show an executed test.

If SSH reports that it cannot resolve an MSOE hostname, check the campus network
or VPN before reinstalling software. That is a connection issue; the remote
environments and submitted Slurm jobs remain on their machines.
