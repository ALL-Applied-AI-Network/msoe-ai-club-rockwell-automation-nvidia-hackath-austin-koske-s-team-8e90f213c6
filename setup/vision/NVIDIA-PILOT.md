# NVIDIA integration pilot

October 20, 2026. Development branch: `austin-work`. This experiment adds an
offline event-explanation branch and benchmarks the existing YOLO vial branch.
Application defaults, detector weights, measurement QA and `main` are unchanged.

**Result: TensorRT is useful for the next development integration. Cosmos3-Edge
is not reliable enough to produce operator explanations or quality decisions.**
The narrower caption experiment is an optional review aid, with factual errors
still visible. No backend or VLM has been promoted into deployed defaults.

## Placement in the architecture

The CPU decodes video, preserves frame identity/timestamps, checks measurement
eligibility and aligns recipe/process context. GPU models provide vial detection,
liquid detection and SAM masks. The numerical recipe comparison and advisory
coordinator retain their existing explicit eligibility checks.

An event packet can also go to Cosmos asynchronously: a bounded recorded clip,
provisional image-derived telemetry, missing-input status and the numerical
policy result. Cosmos produces a draft explanation for a reviewer. Its output
does not enter the action coordinator, authorize geometry, establish composition,
or supply missing recipe/meter data. The first experiment explains whole clips
after they end; it does not establish live warning latency.

TensorRT is an alternative execution backend for a model component. It must be
benchmarked for both latency and output differences before any application switch.
This pilot covers only YOLO11n vial inference, not Faster R-CNN or SAM acceleration.

## Model and compute selection

- The initial proposed Reason2-2B checkpoint returned HTTP 403 with the existing
  account. Access was not bypassed or represented as granted.
- The repository already pins and previously smoke-tested `nvidia/Cosmos3-Edge`.
  Its accessible reasoning checkpoint is used at revision
  `344d602b128d1bbdacb43b08d0a3626f46343e29`, through Transformers 5.20.1.
  This is a different model from Reason2; do not label its results Reason2.
- Rosie hostname resolution failed through both Windows SSH and WSL. No job
  submission or login-node GPU work was attempted. The local RTX 4080 SUPER
  supplies GPU inference for this pilot.
- An isolated `envs/nvidia-pilot-windows` environment installs TensorRT
  10.14.1.48.post1, Transformers 5.20.1 and Accelerate. It references existing
  vision packages through a `.pth` file. Original vision environments are intact.
- The YOLO11n checkpoint SHA256 remains
  `10856c18a8a5c485e832adcc6b841b739e726a969dc043fe8db64c49163bb361`.
  A copy is exported in the experiment directory; the original is untouched.

Upstream references: [Cosmos3-Edge model card](https://huggingface.co/nvidia/Cosmos3-Edge),
[Reason2 examples and requirements](https://github.com/nvidia-cosmos/cosmos-reason2),
[TensorRT integration](https://docs.ultralytics.com/integrations/tensorrt/).
The Edge model card lists Linux as its tested OS; this Windows execution is a
project-specific compatibility experiment, not NVIDIA platform certification.

## Recorded event experiment

Six predeclared eight-second windows from Run 4 cover early pool appearance,
pauses, a checkpoint neighborhood, restart and later filling. Selection uses
already-inspected development evidence; no new onset or mask labels are created.
Each source MP4 export has 240 fully decoded frames. Exact native pixels and PTS
at 40 existing sampling positions per clip are compared with the frozen Stage 2
evidence. The VLM receives lossy exported clips; numerical telemetry remains from
the original source frames. Nothing is derived from Run 5.

Each clip has two paired inputs: video with a common statement of unavailable
process/reference evidence, and the same input plus actual compact telemetry.
Both arms know that recipe identity, acceptable reference, metered volume,
reviewed geometry and authoritative completion are unavailable. The model samples
eight uniformly spaced video frames and generates at most 512 tokens without
sampling. PyAV explicitly decodes those inputs; sampled RGB hashes, frame indices
and timestamps are recorded and independently reconstructed by the verifier.

The requested JSON contains a visible summary, unavailable quality assessment,
limitations and an operator explanation. A strict display contract rejects an
invalid schema or a claimed quality verdict. Free text still requires semantic
review: valid JSON does not establish that a visual statement is correct or that
an operator message is appropriate. No model response can issue equipment actions.

### Actual Cosmos results and refinement

The structured explanation experiment completed all 12 requests. Only **3/12**
meet the requested display contract: 3/6 video-only and 0/6 with telemetry.
Three telemetry responses hit the 512-token limit. The model invents details,
including a robotic arm, final completion and a speculative fluid identity.
Even syntactically correct responses are not established as factually correct.
Adding telemetry did not rescue this small model/prompt configuration.

| Structured explanation input | Latency p50 | Latency p95 | Display contract |
| --- | ---: | ---: | ---: |
| Video + common missing-evidence context | 2.06 s | 2.56 s | 3/6 |
| Same + actual compact telemetry | 8.69 s | 15.65 s | 0/6 |

A second frozen experiment narrows the task to one short visual caption and
constructs quality status/QA counts directly from recorded evidence. It compares
the full frame with a fixed station crop, with 96 maximum output tokens and a
1.1 repetition penalty. Both arms are new development configurations; this is not
a controlled attribution of the improvement to one prompt change.

| Caption input | Latency p50 | Latency p95 | Nonempty captions |
| --- | ---: | ---: | ---: |
| Full frame | 0.994 s | 1.229 s | 6/6 |
| Station crop, resized for the model | 0.606 s | 0.711 s | 6/6 |

All 12 caption requests complete without token-limit truncation. Full-frame
captions still guess container material/nozzle color and describe holder/nozzle
relationships imprecisely. The cropped arm invents an identity (`Brady Vial 9`)
and reverses transfer direction in examples. Nonempty output is **not accuracy**.
Cropping is therefore not selected as an operational reasoning improvement.
The review video shows all six full-frame captions as unverified drafts and
separately labels the deterministic unavailable status; it is not a success-only
selection. No real recipe/defect accuracy or semantic accuracy is claimed.

The practical choice is to keep the numerical policy authoritative, investigate
a stronger accessible reasoner later, and retain this small model only as an
optional caption experiment. It cannot close the missing reference/log evidence.

## Paired TensorRT experiment

Ninety-six uniformly spaced samples from the frozen 1,581-sample Run 4 inventory
are decoded from the original source and matched by RGB hash/native PTS. Both
backends receive identical BGR arrays, a fixed 480×640 input, vial class 0,
confidence 0.35 and NMS IoU 0.45. There is no INT8 calibration or threshold tuning.

The original PyTorch FP32 arm is compared with a TensorRT FP16 engine. Each arm
warms up 20 calls; three measured passes alternate arm order. Host timing includes
preprocessing, inference, postprocessing and copying boxes to CPU, with CUDA
synchronization. Ultralytics' reported GPU inference timing is recorded separately.
Decode, liquid inference, SAM and other pipeline components are excluded.

Box agreement uses one-to-one IoU matching, reports unmatched boxes, minimum/median
matched IoU and confidence differences. This measures backend agreement with
PyTorch, not detection accuracy against human labels. No engine is promoted.

The completed first run measures 288 calls per arm (96 frames × three passes):

| Vial branch | Host processing p50 | Host processing p95 | Reported inference p50 |
| --- | ---: | ---: | ---: |
| PyTorch FP32 | 9.64 ms | 12.55 ms | 7.79 ms |
| TensorRT FP16 | 3.08 ms | 3.65 ms | 1.06 ms |

Median host processing improves 3.13×; reported model inference improves 7.35×.
All 285 predicted boxes across the 288 paired frames match, with no unmatched
boxes on either backend. Median IoU is 0.99955, minimum 0.98259, and maximum
absolute confidence difference 0.000712. One frame per pass has no vial prediction
on either arm. These observations do not establish full-pipeline accuracy or speed.

The first frozen runner accidentally reused its TensorRT summary key for the
version string. Per-frame timings and predictions are intact and hash-pinned.
The verifier explicitly recomputes both arms from those originals; no original
evidence is rewritten and no duplicate GPU benchmark is needed. The source runner
now uses a separate `tensorrt_version` field for future fresh runs.

## Reproduce and inspect

Local root: `C:/Users/Austin/ai-club-tools/results/nvidia-pilot-20261020`.
Use the isolated environment's `Scripts/python.exe` from the repository root.

1. `prepare_nvidia_events.py`: create `event-cases-v1` from the original Run 4
   source and `stage2-20261020/full-run4-v1` evidence.
2. `benchmark_vial_tensorrt.py`: `--half`, original baseline checkpoint, source
   and evidence; output a fresh `tensorrt-fp16-v1` directory.
3. `cosmos_event_review.py`: `--cases` and fresh `--out`; default eight frames,
   six cases and 512 maximum new tokens. Inspect all response JSONs, including
   parsing errors and token-limit hits. An interrupted attempt must remain intact.
4. `cosmos_caption_probe.py`: `--cases` and fresh `--out`; caption-only full-frame
   and cropped arms, deterministic status construction, all 12 responses retained.
5. `render_cosmos_events.py`: original `--source`, `--cases`, completed caption
   `--reviews`, `--response-mode full_frame` and fresh `--out`. The local video labels draft
   explanations, missing quality evidence and retrospective timing.
6. `verify_nvidia_pilot.py`: recompute hashes, decode counts, backend matching,
   explanation contract and descriptive latency statistics.

Completed outputs:

- `event-cases-v1/`: six clips and immutable source/telemetry manifests.
- `tensorrt-fp16-v1/`: engine, original timings/box predictions and first report.
- `cosmos-review-v1/`: failed structured explanations, preserved in full.
- `cosmos-caption-v1/`: both caption arms and deterministic status packets.
- `cosmos-demo-v1/cosmos-event-review.mp4`: 48-second single-view review video;
  all 240 output frames fully decoded and matched to the recorded source trace.
- `verification-v3.json`: 68 artifact hashes verified, metrics recomputed and
  all VLM sampled pixels/timestamps reconstructed. The first verification is
  preserved; v3 adds manifest links, independent demo-trace/QA-count checking,
  recomputed box-agreement aggregates and a frozen verifier snapshot.

See the committed [verification](../verification/nvidia-pilot-20261020.json).

Focused checks: `python -m unittest discover -s setup/vision -p test_nvidia_pilot.py -v`.
Run with an existing Windows vision environment; the bare WSL interpreter lacks
the project's video dependencies.

## Later work

Cosmos Transfer can be evaluated with Cameron/Hayden's synthetic-data work:
review generated geometry and label preservation, then measure first-liquid
detection/mask quality on independently reviewed real clips. It is not part of
this execution. Predict-generated continuations cannot commission recipe colors,
mixing physics or flush effectiveness. DeepStream becomes useful when the video
ingestion/deployment scope warrants it. Real recipe references and synchronized
process evidence remain necessary for the operational demonstration.
