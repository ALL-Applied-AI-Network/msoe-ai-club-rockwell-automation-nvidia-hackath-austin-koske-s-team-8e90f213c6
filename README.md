# Linewatch: recipe-aware vial fill inspection

Linewatch watches CSI's Fluid Mixing Station 3 with the station's fill and tray cameras and,
given the order of recipes being made, decides for every vial whether it matches its recipe, what
to do about it (pass, scrap before the sorter, flush the line, stop and check dosing, or hold)
and when, and explains why. It runs on an NVIDIA GB10 at the line; its vision models were trained
on Isaac Sim renders of the station only. CSI does not allow a connection to the PLC or sorter, so
actions go to the operator and are logged as the commands an integration would send.

**Start here: [PRESENTATION.md](PRESENTATION.md)**: executive summary, results against baselines,
the video script, data and compute.

```text
cameras + recipe order → YOLO11n vial + temporal U-Net liquid + measurement QA → per-vial endpoint
colour vs the recipe's reference → PASS / SCRAP / FLUSH / STOP / HOLD with a deadline
→ cause (rules; Cosmos / VLMs compared) → tray camera confirms where the vial went
```

## Run it

- Laptop (Docker): [demo](docs/demo.md) · GB10: [deployment](setup/gb10/README.md)
- Fastest on a GB10: download the `linewatch-gb10` kit from the
  [latest release](https://github.com/ALL-Applied-AI-Network/msoe-ai-club-rockwell-automation-nvidia-hackath-austin-koske-s-team-8e90f213c6/releases/latest),
  then `bash START.sh --selftest` and `bash START.sh`. Without cameras, add
  `--mock-fill VIDEO --mock-tray VIDEO` and pick **Mock** on the LINE page.
- [CSI session reference](docs/csi-session.md) for on-site recording.

## Repository

| Path | In the submission pipeline | Contents |
| --- | --- | --- |
| `integration/` | Yes | Model runtime (YOLO, U-Net, measurement QA) used by the API |
| `replay/` | Yes | Line runner and live engine, cameras, decisions, tray, RCA, carryover forecast and scheduler |
| `api/`, `client/` | Yes | FastAPI service and React operator app |
| `setup/gb10/` | Yes | GB10 kit, onboarding, self-test, session upload |
| `setup/vision/` | Models only | Model definitions and training; most files are dated experiments |
| `annotation_review/` | Evaluation | Human review and scoring tools |
| `docs/` | Yes | Maintained documentation ([index](docs/README.md)) |
| `setup/isaac/`, `setup/cosmos/`, `setup/rosie/`, `setup/brev/` | No | Synthetic data and remote compute experiments |
| `protoype-Hayden/`, `predict_batch.py`, `train_v5.py`, `team_handoff/` | No | Earlier prototypes and team handoff material, kept for reference |

Rules for contributors and agents: [CONTRIBUTING.md](CONTRIBUTING.md), [AGENTS.md](AGENTS.md).

## Project

Built for the Rockwell Automation / NVIDIA Physical AI hackathon with MSOE and UWM CSI
(Dataset 2). [Organizer documentation](https://msoe-maic.com/library/?article=Rockwell-NVIDIA-Fall-2026).
Measured results and their limits are in [evaluation](docs/evaluation.md).
