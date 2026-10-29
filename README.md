# Linewatch: recipe-aware vial fill inspection

Linewatch watches CSI's Fluid Mixing Station 3 filling camera and, given the schedule of
recipes being made, decides for every vial whether it matches its recipe, what to do about it
(pass, scrap, flush the line, stop and check dosing, or hold) and when, and explains why. It is
designed to replace the line's recipe-unaware Teledyne colour check. CSI does not allow a
connection to the PLC or sorter, so actions go to the operator and are logged as the commands
an integration would send.

**Start here: [the plan](docs/plan.md)**: pipeline, inputs/outputs, actions, RCA, metrics and
schedule to the November 1 submission.

```text
video + recipe schedule → YOLO11n + temporal U-Net + measurement QA → per-vial segmentation
→ endpoint colour vs recipe → action + deadline → RCA (rules / LLM / Cosmos Reason)
→ vial ledger, action log, RCA reports, annotated video, metrics
```

## Run it

- Laptop (Docker): [demo](docs/demo.md) · GB10: [deployment](setup/gb10/README.md)
- Fastest on a GB10: download `linewatch-gb10-v1.2.0.tar.gz` from the
  [latest release](https://github.com/ALL-Applied-AI-Network/msoe-ai-club-rockwell-automation-nvidia-hackath-austin-koske-s-team-8e90f213c6/releases/latest),
  then `bash START.sh --selftest` and `bash START.sh`.
- [CSI session reference](docs/csi-session.md) for on-site recording.

## Repository

| Path | In the submission pipeline | Contents |
| --- | --- | --- |
| `integration/` | Yes | Model runtime (YOLO, U-Net, measurement QA) used by the API |
| `replay/` | Yes | Geometry, measurement QA, endpoints, references, diagnostics (rule RCA) |
| `api/`, `client/` | Yes | FastAPI service and React operator app |
| `setup/gb10/` | Yes | GB10 kit, onboarding, self-test, session upload |
| `setup/vision/` | Models only | Model definitions and training; most files are dated experiments |
| `annotation_review/` | Evaluation | Human review and scoring tools |
| `docs/` | Yes | Maintained documentation ([index](docs/README.md)) |
| `setup/isaac/`, `setup/cosmos/`, `setup/rosie/`, `setup/brev/` | No | Synthetic data and remote compute experiments |
| `protoype-Hayden/`, `predict_batch.py`, `train_v5.py`, `team_handoff/` | No | Earlier prototypes and team handoff material, kept for reference |

Branch status: [docs/branches.md](docs/branches.md). Rules for contributors and agents:
[CONTRIBUTING.md](CONTRIBUTING.md), [AGENTS.md](AGENTS.md).

## Project

Built for the Rockwell Automation / NVIDIA Physical AI hackathon with MSOE and UWM CSI
(Dataset 2). [Organizer documentation](https://msoe-maic.com/library/?article=Rockwell-NVIDIA-Fall-2026).
Measured results and their limits are in [evaluation](docs/evaluation.md).
