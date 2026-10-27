# Resource guide and setup choices

Checked October 16, 2026 against the [hackathon resource page source](https://github.com/MSOE-AI-Club/maic-content/blob/main/unlisted/Rockwell-NVIDIA-Fall-2026-Resources/Rockwell-NVIDIA-Fall-2026-Resources.md),
[compute guide source](https://github.com/MSOE-AI-Club/maic-content/blob/main/unlisted/Rockwell-NVIDIA-Fall-2026-Compute/Rockwell-NVIDIA-Fall-2026-Compute.md),
[coding guide source](https://github.com/MSOE-AI-Club/maic-content/blob/main/unlisted/Rockwell-NVIDIA-Fall-2026-Coding-Agents/Rockwell-NVIDIA-Fall-2026-Coding-Agents.md)
and [collection guide source](https://github.com/MSOE-AI-Club/maic-content/blob/main/unlisted/Rockwell-NVIDIA-Fall-2026-Data-Collection/Rockwell-NVIDIA-Fall-2026-Data-Collection.md).
The public website was unavailable to the text reader, so its maintained source
repository was used. The links below are setup references, not a required stack.

## Ready paths for this project

| Work | Setup provided | Machine |
| --- | --- | --- |
| Video, labels and color measurements | Data environment and doctor | Laptop or GB10 |
| USD and Replicator images | Pinned Isaac, primitive render, existing vial renderer launcher | GB10; RTX Brev fallback |
| Image reasoning and operator explanation experiments | Local Edge runner; Nano option; hosted endpoint client | GB10 or Rosie H100 |
| Small learned detector or other PyTorch work | Existing SIF, writable environment, Slurm templates | Rosie H100 |

The [Isaac requirements](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/requirements.html)
exclude GPUs without RT cores such as A100 and H100. Our actual GB10 render is the
compatibility evidence for this setup. The [Cosmos cookbook](https://github.com/NVIDIA/cosmos/tree/main/cookbooks/cosmos3)
separates reasoning from generation; an image-to-text smoke does not test video
generation or validate a manufacturing explanation.

## Other tools mentioned by the guides

These are optional separate stacks, not silently installed into the team's data
or Isaac environments. Their status is explicit so a mentor can select one
without assuming it already works on Rosie.

| Tool | Setup route and remaining requirement |
| --- | --- |
| Cosmos Generator, Diffusers, vLLM, TensorRT-LLM and NIM | Use the selected [Cosmos integration cookbook](https://github.com/NVIDIA/cosmos/tree/main/cookbooks/cosmos3). Generator needs separate model/runtime dependencies; NIM needs NGC access. Local inference here uses Transformers. No generator/serving deployment was started. |
| FLUX and other image augmentation | Isolate from measurement/reference data. Follow the selected model's license and runtime. This is an optional augmentation experiment, not a way to create physical defect ground truth. |
| TAO, DEFT AOI and Visual ChangeNet | Current [TAO 7.0.1 setup](https://docs.nvidia.com/tao/tao-toolkit/7.0.1/text/getting_started.html) uses the TAO Skill Bank and Execution SDK, with separate NGC/HF credentials and platform setup. Slurm is a listed backend; its compatibility with this exact Rosie Singularity installation has not been tested. Do not substitute Docker on Rosie. |
| Cosmos fine-tuning and VLM fine-tuning | Start a separate training environment after choosing the model and a fixed evaluation. The local reasoning smoke is inference only. |
| Isaac Lab and robot policies | [Spark Isaac Lab playbook](https://build.nvidia.com/spark/isaac/isaac-lab). Separate RL stack; not required for vial image rendering. |
| NuRec and real-to-sim | [NuRec robotics setup](https://docs.nvidia.com/nurec/robotics/index.html). Requires a reconstruction dataset and additional runtime; not provisioned for this fill-monitor task. |
| Physical AI Data Factory augmentation | [Upstream workflow](https://github.com/NVIDIA/paidf-augmentation). Separate model/service setup and licenses; not installed as a dependency of the detector. |
| SOP monitoring blueprints | [Upstream setup](https://github.com/NVIDIA/sop-monitoring-blueprints). Treat training and inference microservices as separate deployments, not a command to run Docker Compose on Rosie. |
| Kumo-TS | [Upstream setup](https://github.com/NVIDIA/Kumo-TS). Kumo-Forecast and Kumo-Anomaly use separate environments and both expose `sdk`. Choose the relevant package before `uv sync --frozen --python 3.12`; no forecasting job was started. |
| NeMoClaw and NeMo Agent Toolkit | [NeMoClaw setup](https://docs.nvidia.com/nemoclaw/latest/user-guide/openclaw/get-started/quickstart), [Agent Toolkit](https://docs.nvidia.com/nemo/agent-toolkit/latest/get-started/tutorials/create-a-new-workflow.html). Optional orchestration services; no persistent agent or daemon installed. |
| Emulate3D and Omniverse Connector | Use CSI's configured workstation after scheduling with Shamar; this SSH setup cannot provision that in-person workstation. |
| Factory Operations FOX and FactoryTalk Optix OPC UA | Architecture/integration references. Requires approved plant interfaces; no plant connection or controller change is part of onboarding. |
| FANUC, ROS 2 and grasping examples | Separate Dataset 3 workflow and approved robot access; not dependencies of our Dataset 2 monitor. |
| Coding agents | The hackathon [coding guide](https://msoe-maic.com/library?article=Rockwell-NVIDIA-Fall-2026-Coding-Agents) recommends Antigravity CLI and describes Claude/Codex access. Each person completes their own installation/login. No teammate's login or agent subscription is copied onto shared accounts. |

## Version decisions

The guides point at moving `latest` documentation. This starter instead pins
Isaac Sim 6.0.0.1 and Transformers 5.20.1, and records exact model revisions.
Some Cosmos cookbook text says Edge requires Transformers main; the inspected
5.20.1 package exposes the Edge class and completed our GB10 image smoke. Prefer
that executed version over chasing an unpinned development checkout.

The underlying Rosie SIF provides CUDA 12.9 and its own PyTorch build. The GB10
uses a separate CUDA 13 wheel. Do not copy virtual environments between x86_64
Rosie, aarch64 GB10 and Windows. Recreate them with the matching setup profile.
