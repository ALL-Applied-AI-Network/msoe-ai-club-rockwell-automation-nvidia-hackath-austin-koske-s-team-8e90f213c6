# GB10 versus local station: October 22 benchmark

The GB10 is faster on the tested station request path, but moving live Isaac
rendering there does not make the whole pipeline run at 30 fps. A persistent
inference worker also removes substantial overhead on the local machine.

## Matched cached-input results

All arms used the same 111 pinned 640×480 video samples, YOLO11n / Faster R-CNN
ResNet50-FPN / SAM2.1 Tiny model hashes, measurement QA and four CPU threads.
Each arm completed a full 111-sample warm-up followed by 222 measured samples.
Cached input never substitutes cached predictions.

| Timed boundary | RTX 4080 Super median / p95 | GB10 median / p95 |
| --- | ---: | ---: |
| Warm model/measurement processing | 111 / 122 ms | 101 / 103 ms |
| Direct station processing, including overlays/logging | 126 / 139 ms | 113 / 116 ms |
| Existing per-request HTTP processing, loopback | 272 / 323 ms | 130 / 148 ms |
| Experimental persistent-worker HTTP processing, loopback | 148 / 169 ms | 120 / 122 ms |

The persistent worker reduced local median request time by about 46%. It uses
the same model processor and a bounded single inference thread behind the
existing request boundary. The current operator app was **not changed** to use
it. These are isolated experimental servers, not an automatic deployment.

Windows and Linux differ, as do CUDA 12.8 and 13.0. PyTorch 2.11, TorchVision 0.26,
Ultralytics 8.4.182, NumPy 2.5.2 and the frozen models match. These are comparisons
of the tested systems, not proof of hardware-only speedup. The existing local
simulated source was paused during local tests and restored afterward; its
resident model and desktop processes remained present.

## Paced source and the actual link

The paced source uses the existing sample schedule in the 21.7-second video.
It skips older inputs rather than building a stale queue.

| Test | Source-to-result age median / p95 | Delivered inputs |
| --- | ---: | ---: |
| Direct local station, no HTTP/display | 148 / 170 ms | 110/111 |
| Direct GB10 station, no HTTP/display | 114 / 118 ms | 111/111 |
| Local PNG upload over campus VPN/SSH → GB10 worker → ACK | 320 / 410 ms | 102/111; nine skipped |
| Separate paced upload with actual browser polling/image loading | 556 / 675 ms to loaded image | 93/111 acknowledged; 84 unique images observed in browser |

In the browser run, 17 scheduled inputs were skipped and one final input was
unprocessed when the clip ended. The browser coalesced some results between
updates. The maximum measured age of a loaded image was 734 ms. Each observed
image's source identity was joined to actual remote inference. This measures
the **sampled CV view**, not 30 fps encoded video streaming or camera-exposure-to-
physical-monitor latency. The browser and simulated source clock were on the same
Windows machine; remote clock synchronization was not assumed.

The link test uploads PNGs from the local computer. A camera/file source located
beside the GB10 would have a different transport path. Sending encoded display
video/results to a remote browser also needs its own benchmark; do not call
the 120 ms loopback result the remote viewing latency.

## Rendering plus inference

The same supplied station/CAD and current procedural liquid helper were exercised
at 8 samples per pixel, two subframes and two settled captures per input frame.
This was RGB-only: no expensive private geometric-label captures. Each run saved
37 actual RGB frames; the first five were excluded from timing summaries.

| GB10 test | Median / p95 per step | Measured steady rate |
| --- | ---: | ---: |
| Rendering only | 360 / 697 ms | 2.37 frames/s |
| Sequential render → inference acknowledgement | 485 / 877 ms | 1.81 processed frames/s |

The combined test uses the current source's acknowledged sequential flow. It is
not a concurrent asynchronous renderer/inference throughput study. The GB10's
installed Isaac reported **6.0.0-rc.59**; local source development used 6.1.0-rc.26.
No renderer version, source asset, camera pose or optical parameter was replaced.
Image-quality parity across those runtimes is unclaimed.

The first render attempt stopped before producing images because its subprocess
omitted the existing onboarding EULA environment flag. That failed directory/log
is retained; fresh `v2` directories completed using the existing launch guidance.

## Numerical agreement and verification

- Across 222 matched local/GB10 offline samples, every sample-QA decision agreed.
  Final masks were identical on 216/222 samples. For 194 pairs with nonempty masks,
  median mask IoU was 1.0 and minimum IoU was 0.9612.
- Within each machine, existing HTTP versus persistent-worker HTTP produced
  identical sample-QA decisions and final masks on all 222 measured samples.
- Independent checks joined 2,784 actual model rows to source identities and
  recomputed 2,397 sampling colors. Repeated warm-up/test passes are not
  independent accuracy examples. All 74 saved native-render RGB hashes and
  original asset hashes were verified.
- No fitting, Run 5 use, approved label generation, industrial connection or
  hardware action occurred. All benchmark GPU processes stopped, the shared GPU
  lock was released and the temporary SSH tunnel closed. The existing app resumed.

Agreement is numerical consistency, not reviewed liquid-mask or recipe-defect
accuracy. Raw source hashes, clocks, incomplete/unavailable cases and both
successful and failed evidence remain preserved.

## Deployment decision

Prefer **pre-rendered test input or an approved camera/file source → persistent
inference worker → Part 2 → operator UI**. Keep source acquisition and inference
co-located when practical. The GB10 is viable for the tested 5 Hz CV profile;
the frozen high-quality live-render loop is the bottleneck. A small explicit
display buffer can align image/results; it cannot remove transport or inference
time. Validate warning timing and dropped-frame behavior before promoting a
worker or changing the sample cadence.

Local artifacts: `C:/Users/Austin/ai-club-tools/results/gb10-station-benchmark-20261022`.
Remote artifacts: `/home/hackathon8/ai-club-tools/experiments/gb10-station-benchmark-20261022`.
See [verification and exact timings](../verification/gb10-station-benchmark-20261022.json).
