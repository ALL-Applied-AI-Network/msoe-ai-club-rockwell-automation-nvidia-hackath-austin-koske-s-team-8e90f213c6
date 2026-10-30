# Rockwell Automation + NVIDIA Innovation Lab · Fall 2026

**October 26 temporal CNN release:** [current architecture, exact model downloads and runnable examples](team_handoff/temporal_cnn/README.md). The selected liquid path is direct paired U-Net; Faster R-CNN and SAM are optional comparators. Austin authorized this scoped handoff on `main`; unrelated development remains on `austin-work`.

## Current project — October 24, 2026

We are working on **Dataset 2: Fluid Mixing Station 3**, with a separate Dataset 1
final-inspection pilot. The leading filling candidate uses YOLO for vial geometry
and a paired temporal U-Net for liquid masks and presence. Its liquid branch is
SAM-free; SAM2.1 refinement remains optional experimental work. See the
[current architecture decision](setup/vision/CURRENT-ARCHITECTURE.md) for model
identities, evidence, ownership and pending validation. This records the team
integration direction; existing application defaults are unchanged.

The latest matched GB10 trial measures **16.37 ms median CV processing** and
**105/264 QA-eligible samples** for direct U-Net on inspected Run 4 frames.
Timing excludes decode, transport and browser; QA availability is not mask
accuracy. Independent pool/onset review, reference qualification and fresh
physical testing remain necessary. Recipe-aware advice requires separate
qualified recipe, completion and identity inputs.

- **Students:** [team brief, data downloads and exact YOLO label folders](team_handoff/README.md).
  Aleyna, Zoe and Reagan are working together on YOLO vial detection and box review.
- **Development:** [current handoff and next work](HANDOFF.md), on `austin-work`.
  `main` contains the student data/document handoff; experimental development stays
  on `austin-work`.
- **Recording performance:** [TensorRT comparison and local demo paths](setup/vision/PIPELINE-ACCELERATION.md).
  Run 4 sampled decode + processing median **89.48 → 83.22 ms** on an RTX 4080
  Super, a **6.99% latency reduction**, with one changed sample-QA decision in
  1,581 compared frames. This is a 5 Hz offline experiment; TensorRT remains opt-in.
- **Live digital twin:** [21-case study and results](setup/vision/TWIN-EXPERIMENTS.md).
  **803 live inference frames** verified. An adaptive replay with camera correction
  before the frozen models gives **0/10 nominal warnings and 5/6 residue warnings**;
  the half-mL-equivalent case is missed. These are simulated development cases,
  not real defect accuracy. Operational quality remains unavailable.
- **Synthetic appearance:** [Cosmos Transfer trial](setup/vision/COSMOS-TRANSFER-TRIAL.md).
  One 93-frame clip generated and verified. It invents liquid in empty frames and
  changes ingredient colors, so automatic label reuse is blocked. Source/generated
  QA availability is 24/29 versus 17/29; these are diagnostics, not accuracy. No training.
- **Next evidence:** [early-liquid onset/mask review](annotation_review/EARLY-FILL-HANDOFF.md)
  and the [CSI visit packet for early next week](setup/vision/CSI-SESSION-PACKET.md)
  (proposed October 26–28; booking unconfirmed). The
  [original-video correction check](setup/vision/REAL-CAMERA-PREPROCESSING.md)
  shows no aggregate QA gain, so raw input remains the test baseline.
  Reviewed onset/mask accuracy and real defect precision/recall are not established.

The entries below include dated experiment history. October 24 DINOv3 fits and
the matched U-Net comparison are complete; independent real accuracy and
physical deployment remain unestablished. See [current status](HANDOFF.md).

## Event information

This is your team's repository for the four-week physical AI hackathon run by Rockwell Automation, NVIDIA, the MSOE AI Club and UWM's Connected Systems Institute. Build everything here. Sponsors can follow your work as it progresses, and your final submission links straight to this repository.

**Full event guide** (datasets, compute, rules, judging, prizes): <https://msoe-maic.com/library/?article=Rockwell-NVIDIA-Fall-2026>
**Your team workspace** (team, repository access, submissions): <https://dashboard.all-ai-network.org/j/L36HEH>
**Questions:** [join the hackathon Slack](https://join.slack.com/t/rockwellnvidi-wq82211/shared_invite/zt-4bp05nsn9-UW3afFtM9wjzWtZiQoJBHg) and ask in the channel that matches your question. The event guide's "Getting help" section says who to ping for what.

## Key dates

| When | What |
|---|---|
| Thu Oct 8, 4:00–6:00 PM | Kickoff at Rockwell Automation HQ, 1201 S 2nd St, Milwaukee |
| **Wed Oct 14, 11:59 PM CT** | **Team plan due:** [submit it here](https://dashboard.all-ai-network.org/s/V28NHK) |
| **Sun Nov 1, 11:59 PM CT** | **Final submission due:** [submit it here](https://dashboard.all-ai-network.org/s/V28NHK) |
| Thu Nov 5, 5:00–8:00 PM | Finals at MSOE Diercks Hall: live presentations and prizes |

## Organizer onboarding checklist

- [ ] Everyone on the team has an ALL AI Network account and has joined this team in the [workspace](https://dashboard.all-ai-network.org/j/L36HEH).
- [ ] Everyone has connected GitHub and accepted the repository invitation (check your GitHub email, or open the workspace and press the check button).
- [ ] Pick your dataset. We recommend one strong solution around a single dataset.
- [ ] Download the data and make sure everyone who needs it can open it.
- [ ] Read the compute guide and decide what you'll need.
- [ ] Join Slack, open `#hackathon-announcements`, and find your dataset's channel.
- [ ] Submit the [team plan](https://dashboard.all-ai-network.org/s/V28NHK) by Wednesday, October 14. It's a short resource request, not a graded paper, and any teammate can update it before the deadline.

## The three datasets

1. **[Final Vial Inspection](https://msoe-maic.com/library?article=Rockwell-NVIDIA-Fall-2026-Vial-Inspection):** find missing caps, crooked caps, damaged vials and unexpected empty positions after the robot fills a fixture.
2. **[Fluid Mixing Station 3](https://msoe-maic.com/library?article=Rockwell-NVIDIA-Fall-2026-Fluid-Mixing):** use synchronized camera views to detect fill and color problems caused by recipes and fluid carryover.
3. **[Grasping and Sorting](https://msoe-maic.com/library?article=Rockwell-NVIDIA-Fall-2026-Grasping-Sorting):** use a digital twin and multiple cameras to make a standalone FANUC arm grasp and sort objects more reliably.

## Read before your team plan

- **[Compute: Brev, DGX Spark and build.nvidia.com](https://msoe-maic.com/library?article=Rockwell-NVIDIA-Fall-2026-Compute).** Brev credits are spent whenever an instance is running, so stop instances you aren't using.
- **[Collect and test your own data](https://msoe-maic.com/library?article=Rockwell-NVIDIA-Fall-2026-Data-Collection):** visits, extra data, digital twins, and booking a supervised test on the real equipment.
- **[Tools and resources](https://msoe-maic.com/library?article=Rockwell-NVIDIA-Fall-2026-Resources):** AI coding tools, vision-language models, NVIDIA Cosmos and more. Start with a simple baseline.

## Final submission (Sun Nov 1)

- A video of about five minutes
- A one-page project summary
- A link to this repository
- References and credits for data, models, code and tools

## Working in this repository

**Liquid vision teams: [start with the architecture brief and data downloads](team_handoff/README.md).**
The [versioned team release](https://github.com/ALL-Applied-AI-Network/msoe-ai-club-rockwell-automation-nvidia-hackath-austin-koske-s-team-8e90f213c6/releases/tag/team-data-v1.0.0)
contains the student PDFs, a combined core dataset and optional component archives.
Large data stays in release attachments; development continues on `austin-work`.

**Current development context: [consolidated handoff](HANDOFF.md).** Replay,
annotation review and vision experiments are integrated on `austin-work`; start here
for artifact locations, ownership and unfinished work.

Team environment onboarding: **[start here](setup/START_HERE.md)** for laptop
data work, GB10 Isaac/Cosmos, and Rosie Singularity/Slurm. The
[setup guide](setup/README.md) includes commands and executed verification.

Recording replay app: run `pwsh -File setup/run-replay.ps1`, then open
<http://127.0.0.1:8765>. See [replay instructions](replay/README.md) for source
paths, annotation handoffs, reference timing, batch exports and current limits.
Use [current replay vision guidance](replay/VISION.md) for tracking and sampling.
The [historical recipe-aware handoff](replay/HANDOFF.md) documents input/export
contracts; its old fixed-patch color example is superseded.

```bash
git clone <this repository's URL from the green "Code" button>
```

- Commit early and push often. Mentors and judges see what's here.
- **Don't commit large datasets or model weights.** Keep them out with `.gitignore` and note in this README where to download them.
- Before the finals, replace this README with your project: the problem, your approach, results, how to run it, and credits.

**Can't push or see this repository?** Open the [team workspace](https://dashboard.all-ai-network.org/j/L36HEH) and press the GitHub check button. If it still doesn't work, ping Brett Storoe in `#hackathon-general` on Slack.
