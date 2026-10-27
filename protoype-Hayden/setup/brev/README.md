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
