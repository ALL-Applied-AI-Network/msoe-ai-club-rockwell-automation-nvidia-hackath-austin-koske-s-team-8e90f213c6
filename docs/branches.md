# Branches

Snapshot October 27, 2026. `main` is the submission branch; every branch below either is
already in `main`, has work to bring in deliberately, or is reference-only. Nothing here has been
deleted. Ask the owner before deleting a branch.

| Branch | Owner | Commits not in `main` | Recommendation |
| --- | --- | --- | --- |
| `austin-work` | Austin | 0 | Working branch; identical to `main` at this snapshot |
| `austin/integration-dashboard-main` | Austin | 0 (2026-10-29) | LINE page, live cameras, line runner, RCA backends, carryover tooling; fast-forwarded into `main` and `austin-work` on 2026-10-29 |
| `austin/demo-ui`, `austin/docs-cleanup`, `austin/recipe-aware-replay` | Austin | 0 | Merged; delete |
| `austin/run5-recipe-check` | Austin | 1 (file content already identical in `main`) | Merged; delete |
| `annotation-review` | Austin | 0 | Merged; delete |
| `IntegrationDashboard` | Josiah | 0 | Merged; continue app work from `main` |
| `three-model-integration`, `liquid-pool-model-v5` | Srijan | 0 | Merged; delete or continue from `main` |
| `zoe_jev_cosmos` | Zoe, Jev | 2: Cosmos Reason2-2B LoRA notebook, adapter and tokenizer (~70 MB weights, ~900k lines of tokenizer text) and Jev's boxed test images | **Bring in the notebook and README** under `setup/cosmos/` for the RCA stream; publish adapter weights as a release asset rather than committing them |
| `alayna-yolo` | Aleyna | 3: YOLO training run outputs, weights (`best.pt`, `yolo26n.pt`) and `.idea/` files | Reference only; released YOLO11n stays. Record results in `docs/evaluation.md` if compared; weights to a release asset |
| `team-3` | Cameron | 1: SAM2.1 pilot scripts and PDF | Reference only (SAM is out of scope for submission) |
| `cameron/data_pipeline` | Cameron | 1: Isaac Replicator generator, Brev/Rosie scripts | Reference only; synthetic-fault work for submission starts from `main` |
| `hayden-work` | Hayden | 1: contact sheet and manifest | Reference only |
| `team/app-starter` | Austin | 2: early API/client skeleton | Superseded by `api/` and `client/` on `main`; delete |

Rules for the remaining days: branch from `main`, keep model weights and large outputs in GitHub
releases (not Git), and open a pull request into `main` per workstream in [the plan](plan.md).
