# Start here

**October24 station onboarding:** use the
[AI-guided data-to-deployment workflow](../team_handoff/integration/AI-NATIVE-ONBOARDING.md)
and project skill at`.agents/skills/commission-vision-station/SKILL.md` for source
inventory, current model/profile requirements, qualification and deployment
handoff. The initial helper checks files/hashes only; later stages reuse existing
tools and require their actual evidence. Current ownership/model direction is in
[the architecture decision](vision/CURRENT-ARCHITECTURE.md). The original
environment bootstrap instructions below remain available.

Pick your assignment below. You do not need to install the entire NVIDIA stack.

## Aleyna Zoe and Reagan

With Python 3.12 installed, open PowerShell 7 in the repository and run:

```powershell
pwsh -File setup/bootstrap.ps1
& "$HOME/ai-club-tools/envs/data-windows/Scripts/python.exe" setup/check_dataset.py --data "C:/path/to/csi-vial-fill-dataset"
```

Replace the example data path with your extracted dataset folder. The bootstrap
prints the Python interpreter to use and a check report. A complete
download should decode all 15 videos and find 30 Run 5 label rows. The check
reads files without changing them. Use `--data PATH` for another download folder.

## Cameron and Hayden

Get your GB10 SSH access from Austin. [Key setup instructions](README.md#connect-to-the-gb10).
The common software and existing vial assets are already on Spark 8. After SSH:

```bash
cd ~/ai-club-onboarding
export TEAM_MEMBER=cameron  # change to your name
export OMNI_KIT_ACCEPT_EULA=YES  # after accepting the Omniverse license
bash setup/isaac/render_project.sh
```

The command prints the output directory and renders one existing smoke example.
Coordinate use of the shared machine; the launcher prevents competing team GPU
commands. Keep using this launcher for the Isaac 6.0 compatibility adapter. Its
optional arguments are output directory, sample count, and your own sample JSONL.
Start with the supplied fixture, then develop your assigned cases. No installation
is needed on the already provisioned GB10.

## Rosie users

Use your own university account. From the repository on Windows:

```powershell
pwsh -File setup/deploy.ps1 -Target Rosie
ssh msoe-bastion
```

On Rosie:

```bash
cd ~/ai-club-onboarding
bash setup/bootstrap.sh rosie
bash setup/rosie/submit.sh smoke
```

This uses the site Singularity image and submits a short H100 check through
Slurm. [Cosmos and project-job commands](README.md#run-on-rosie-with-singularity).

## Srithej and Josiah

Your workflow and booking tasks do not require a GPU installation. Use the
organizer email Austin shared and the [collection guide](https://msoe-maic.com/library?article=Rockwell-NVIDIA-Fall-2026-Data-Collection).
Prepare the session request with proposed dates, duration, attendees and a test
plan; confirm the operator workflow, flush/carryover behavior and approved camera
feed/network arrangement. Send the draft to Austin to coordinate the booking.
The compute plan is GB10 for rendering and Rosie for training/reasoning.

## If something fails

Send Austin the command, machine name and first error. Don't spend a day
reinstalling. The [full instructions](README.md), [resource catalog](resources.md)
and [executed checks](verification/README.md) explain which paths are tested.
