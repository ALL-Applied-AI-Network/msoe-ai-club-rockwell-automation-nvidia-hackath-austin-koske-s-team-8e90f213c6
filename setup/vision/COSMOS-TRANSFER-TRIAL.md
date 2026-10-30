# Cosmos Transfer appearance trial

October 21, 2026. **One neutral Transfer clip completed and verified; held from training because it changes liquid content.**
This is a candidate extension to Cameron/Hayden's synthetic-data work, isolated
from their renderer, approved labels and the deployed inspection pipeline.

The proposed training-data path is:

```text
Isaac scene + process trajectory
  -> RGB and geometry controls
  -> Cosmos Transfer appearance variants
  -> temporal / geometry / liquid-color review
  -> eligible synthetic training set
  -> controlled training comparison on independently reviewed real development data
```

Transfer belongs in offline data preparation. Recipe decisions keep using
observed camera pixels and aligned process evidence. Generated continuations
cannot establish correct mixing physics, recipe acceptance or flush effectiveness.

## Why try it

The [803-frame twin study](TWIN-EXPERIMENTS.md) found that camera color variation
can break liquid geometry before measurement, and that stream detectability is
sensitive to uncalibrated material appearance. Transfer could vary lighting,
glass reflections and metal surfaces around known source trajectories. Whether
that improves first-liquid detection or liquid masks is still an experiment.
Photorealism alone is not a useful success metric.

## Prepared pilot

Local root:
`C:/Users/Austin/ai-club-tools/results/cosmos-transfer-trial-20261021/`.

`inputs-v1/` contains one source/control pair and three generation specifications:

- `source-nominal.mp4`: a **5.8125 s / 93-frame / 16 fps** input, 960×720.
  It covers empty/early/accumulating liquid from the prior nominal Isaac reference.
  **29 distinct native 640×480 / 5 Hz frames** are held/repeated, then resized;
  this does not create native detail or new simulated motion. `frame-map.json`
  records each source timestamp/hash. No real footage or Run 5 is used.
- `edge-control.mp4`: Canny edges (100/200) from those same resized frames.
  This first trial uses RGB-derived edges, not a newly rendered native depth pass.
- `transfer-neutral.json`, `transfer-cool-light.json`, `transfer-warm-light.json`:
  base edge control weight 1.0, guidance 3, 35 steps, seeds 19/23/29. Prompts ask
  to preserve camera, vial count/geometry, liquid heights/hue and stream timing.
  **Prompts and strong control do not guarantee preservation.**
- `source-control-contact-sheet.png`: input/control previews, not generated outputs.
- `preparation.json`: checked source evidence, configuration and media hashes.
  `training_eligible=false`; no source annotations are approved for transfer.

Pinned candidate: **NVIDIA Cosmos-Transfer2.5-2B**, base single-view edge model.
Source repository revision `2ff49d0717af02057ae79bc75c00fbff9da1b4e7`;
checkpoint repository revision `ce8440327c632d8313c3bde69db13b627ba5cae1`.
The [official inference guide](https://github.com/nvidia-cosmos/cosmos-transfer2.5/blob/2ff49d0717af02057ae79bc75c00fbff9da1b4e7/docs/inference.md)
lists 65.4 GB for base single-GPU inference. The 16 GB local GPU does not meet
that configuration. The [setup guide](https://github.com/nvidia-cosmos/cosmos-transfer2.5/blob/2ff49d0717af02057ae79bc75c00fbff9da1b4e7/docs/setup.md)
specifies CUDA 13.0 for DGX Spark. This is a chosen initial candidate, not a claim
that 2.5 is NVIDIA's newest generation or that it has run on our GB10.

The GB10 was reachable, with approximately 118 GiB available host/unified memory,
1.7 TB free storage and no matching inference process at inspection. Existing
Cosmos uses Torch 2.11.0+cu130. No environment was modified and no generation
was started. Future GPU execution must use the shared `spark-gpu.lock`; an
isolated Transfer environment must preserve the existing Reason/Isaac installs.
Rosie remains an H100 fallback using Slurm and Singularity 3.10 only.

## Access and execution status

`access-v1.json` retains the original 403/GatedRepo checks. After Austin accepted
access, `access-v2.json` and `access-v3.json` returned 200 for Transfer and
Guardrail. `tokenizer-access-v2.json` also returned 200 for the required
**Cosmos-Predict2.5-2B video tokenizer**; the pinned Cosmos-Reason1 text encoder
was already accessible. The access helper now checks all four of these model
repositories. No Hugging Face token is written to the shared GB10:
the downloader receives it on encrypted SSH stdin, uses it in memory, and records
only model revisions/file hashes.

The current GB10 experiment is
`~/ai-club-tools/experiments/cosmos-transfer-20261021-v2/`.
The pinned upstream source uses Python **3.13** for available CUDA 13 ARM wheels.
The first Python 3.12 attempt is preserved in `cosmos-transfer-20261021-v1/`.
Missing X11 headers and the ARM decoder's FFmpeg 6 runtime libraries are
extracted from public Ubuntu packages into the experiment-local `sysroot/`
(including private BLAS/LAPACK search paths); system packages and teammate
environments are not changed. `bootstrap-headers.log`, `header-package-sha256.txt`, and the eventual
`environment-freeze.txt` retain the environment evidence.

The upstream runtime pins an older Transfer repository revision
`b67b64abda3801a9aceddbff2bdb86126c06db74`; its edge checkpoint has the **same
LFS SHA256** as the prepared newer revision, verified in
`model-revision-equivalence-v1.json`. Do not silently substitute a different
model. The downloader fetches only the required edge checkpoint and video tokenizer,
plus text encoder, safety models and tokenizer metadata. Its completed manifest
verifies all downloaded sizes and available LFS hashes.

All three specifications passed the actual upstream parser in
`schema-validation-v1.json`. Download completion and LFS checks passed.
The installed runtime is Python 3.13.16, Torch 2.9.1+cu130. Startup attempts
`outputs-neutral-v1/` through `v3/` are retained: missing offline safety encoder,
missing FFmpeg runtime, then missing image-context encoder. The latter is
`google/siglip2-so400m-patch16-naflex`, pinned in the separately verified
`context-downloads-completed.json`. The safety model's existing SigLIP weights
are reused from its verified bundle rather than redownloaded or substituted.

**Neutral v4 completed successfully**: 35 diffusion steps, prompt and video checks
passed, exit 0. Launcher elapsed time was **1,544.997 s (25 min 45 s)** on GB10,
including model loading, checks and export; sampling itself took about 21 min
32 s. This is one run, not a hardware benchmark. The detached run finished
through the VPN interruption and was retrieved after reconnection.

The completed bundle's SHA256 and **all 37 included file hashes/sizes** were
checked locally. The generated H.264 clip fully decodes to **93 frames / 16 fps /
960×720 / 5.8125 s**. Source inputs match the frozen preparation byte-for-byte.
`verification-neutral-v1.json` independently checks 58 retained model samples
against decoded/resized pixels, timestamps, sampling lineage, mask sizes and
paired diagnostics. It verifies actual retained inference rather than claiming
another neural-inference reproduction. The GB10 GPU lock is released.

## Result: usable as a failure probe, not inherited-label training data

Visual inspection of the aligned contact sheet flags:

- **Frame zero already has a blue pool** in the generated active vial while the
  source vial is empty. Other source-empty vials also acquire colored contents.
- The source red→dark-purple sequence becomes substantially different generated
  blue/green/yellow/orange contents. The rising liquid surface is not a reliable
  continuation of source state.
- Much station geometry survives. Median whole-frame edge F1 within two pixels
  is **0.872**, illustrating why global edge agreement cannot approve liquid labels.

These are experimental inspection findings, not mentee-approved annotations.
Source geometry/timing cannot be copied as correct generated labels. No detector
was retrained, and no data/default/QA changes were made. The cool/warm jobs remain
unstarted because repeating the same failed control setup is not the next useful
experiment. One edge-controlled clip does not establish that Transfer is unsuitable
in every configuration.

### Frozen-model response

Both videos were resized identically to 640×480 and sampled at the first input
frame for each distinct held source frame: **29 samples per video**. The actual
frozen YOLO11n/Faster R-CNN/SAM2.1 pipeline ran separately for each video.

| Diagnostic | Source clip | Generated clip |
| --- | ---: | ---: |
| Vial prediction available | 29/29 | 22/29 |
| Liquid prediction available | 25/29 | 29/29 |
| Nonempty final sampling mask | 25/29 | 22/29 |
| Sample QA passes | 24/29 | 17/29 |
| First liquid model prediction, clip time | 0.8125 s | 0.0 s |
| First QA-valid sample, clip time | 1.0 s | 1.0 s |

The frame-zero prediction does **not** demonstrate improved early-liquid detection:
the generated data invents liquid before the source fill. These are model response
and availability counts, not annotation accuracy or real defect precision/recall.
Source results can differ from the live study because this source has been resized,
compressed and resampled. The 17 paired QA-valid raw Lab comparisons have median
dE2000 **45.85**; that diagnostic combines appearance and model sampling differences
and is not a color-calibration accuracy measurement.

### Local media and evidence

Under `C:/Users/Austin/ai-club-tools/results/cosmos-transfer-trial-20261021/`:

- `neutral-v4/outputs-neutral-v4/vial-neutral.mp4`: clean generated-only H.264 clip.
- `model-evaluation-v1/generated-inspection.mp4`: single-view generated clip with
  actual model proposals/sample masks; held overlays update only on causal samples.
- `review-neutral-v1/source-generated-contact-sheet.png`: aligned source/generated
  frames and active-vial crops, including empty-frame failure.
- `review-neutral-v1/review.json`: all 93 frame association/edge diagnostics.
- `model-evaluation-v1/comparison.json`, `paired-model-disagreement.json`: model
  response counts and disagreements; no generated ground-truth assumption.
- `verification-neutral-v1.json`: completed local verification.
- `neutral-v4/completed-neutral-artifacts-v1.manifest.json`: retrieved file hashes,
  runtime pins, logs, source identity and successful launch metadata.

The next useful design is stronger control over task-critical contents: preserve
empty intervals, liquid geometry and ingredient colors, and limit appearance
changes to reviewed lighting/background regions. A candidate would combine spatial
and low-frequency color controls with an explicit empty-vial context, rather than
repeating unconstrained edge-only stylization. This is a proposed next experiment;
no new control configuration has been run or validated. A background-only variant
would not itself solve the faint-liquid material gap found in the live study.

Review before any training: empty-frame hallucinations, early-pool presence,
vial/nozzle geometry, surface-height drift, liquid hue, stream timing, temporal
flicker and frame association. Generated masks/boxes need review; source masks
cannot simply be copied. Frozen-model consistency is a diagnostic, not human
truth. Keep a generated-data lineage distinct from source geometric labels.

If outputs pass review, compare identical training budgets with and without the
new data. Split by original trajectory **before** creating appearance variants so
siblings cannot leak across splits. The existing tiny source clip is a smoke
trial, not a sufficient training corpus. Keep Run 5 outside fitting, selection,
threshold tuning and sampling. Measure early-liquid presence, onset delay and
mask quality on independently reviewed real development clips before scaling.

## Reproduce preparation and access checks

```powershell
$analysis = "$HOME/ai-club-tools/envs/vision-alternatives-windows/Scripts/python.exe"
$source = "$HOME/ai-club-tools/results/twin-experiments-20261021"
$trial = "$HOME/ai-club-tools/results/my-transfer-trial"
& $analysis setup/vision/prepare_cosmos_transfer_trial.py --source "$source/study-v1" `
  --verification "$source/verification-v1" --out "$trial/inputs"
& $analysis setup/vision/probe_cosmos_transfer_access.py --out "$trial/access.json"
```

All output paths must be new. Neither command performs generation or training.
