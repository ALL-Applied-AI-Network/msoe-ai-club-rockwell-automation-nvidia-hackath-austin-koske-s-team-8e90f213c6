# Live twin experiments: process, camera and stream transfer

October 21, 2026. Follow-up to [the first live pilot](LIVE-TWIN.md), on `austin-work`.

The experiment plan was frozen before the batch rendered. It contains **21
deterministic development trajectories**: one nominal reference, ten nominal
controls with process/camera variations, six initial-residue perturbations, one
scripted prime-to-drain comparison and three separate appearance probes. Each
case runs the actual frozen YOLO11n, Faster R-CNN ResNet50-FPN and SAM2.1 models
against newly rendered frames. No predictions or masks are substituted for inference.

## Questions and frozen changes

1. **Why did image-only fill starts fail?** The first pilot's maximum stream
   contrast was 1.926, below the existing start threshold 3.0 and peak requirement
   6.0. Probe jet radius 1.0→1.5 mm and shared absorption depth 25→5 mm, separately
   and together. These are uncalibrated appearance probes; they do not authorize
   a new liquid material or establish physically correct jet visibility.
2. **Does exact meter matching discard evidence when flow changes?** Compare the
   existing exact-progress reference with bounded within-phase Lab interpolation.
   Never extrapolate, cross phases or bridge a reference gap larger than 2.5 mL.
   The nominal reference cadence is 1.2 mL per active 0.2-second sample. Test flow
   scale 0.75 and 1.25 at the same final commanded quantities.
3. **Can background normalization preserve warnings under camera variation?**
   Compare raw color with sRGB gain correction from three fixed background patches,
   and with the existing experimental health bounds applied. Keep all methods at
   **5 dE / three consecutive qualified samples**. Camera perturbations are 15%
   green/blue attenuation, 10% encoded exposure reduction, native light intensity
   0.8/1.2, and one obscured background patch. Test slow flow plus green attenuation
   for both nominal and different-residue initial conditions.

The path model retains its assumed 2 mL mixed compartment. Initial green residue
is either 2 mL or a 0.5 mL equivalent diluted with nominal first ingredient; a
separate blue donor tests a different constituent. The 8 mL simulated purge uses
first ingredient and drains before filling. CSI's actual procedure is unknown.
Residue amount is a model perturbation, not a product rejection tolerance.

The three optical profiles change the reference-to-observation appearance model.
Keep them separate from nominal camera controls when discussing warning counts.
No physical camera calibration, approved recipe reference or human mask/onset
label is introduced. The operational recipe gate remains unavailable.

## Evidence and interpretation

Local root: `C:/Users/Austin/ai-club-tools/results/twin-experiments-20261021/`.

- `plan-v1.json`: complete predeclared cases; the source plan is never overwritten.
- `smoke-v1/`: 48 live frames across six partial trajectories, including flow,
  sensor-gain and optical variants. This does not score full fills.
- `study-v1/`: complete batch, separate source RGB/private geometric masks,
  exact acknowledgements, model outputs and all four diagnostic methods.
- Verification recomputes source associations, private process trajectories,
  masks/decisions, media decoding and every diagnostic from saved RGB/model masks.

The four methods are `exact_raw`, `metered_raw`, `metered_normalized` and
`metered_normalized_guarded`. Reference capture is excluded from warning-rate
denominators. Every method reports eligible comparisons and unavailable reasons;
no comparisons is not a successful no-warning result. All cases are reported,
including missed perturbations and blocked camera evidence.

The station's carrier remains frozen. Report image-derived **fill starts** and
stream availability separately from completed-vial lifecycle: EOF cannot stand
in for a physical departure. Phase/meter metadata remains simulator-supplied;
it does not prove that the image-only detector recognized dispensing.

These cases extend development coverage; one trajectory per condition is not
a production accuracy estimate or a held-out real-world test. Native-renderer
capture variation between launches remains in the records. Run 5 is unused.
Original CAD, teammate renderer, labels, QA, stream thresholds and app defaults
remain unchanged. No hardware or autonomous intervention is connected.

## Completed results

The full study completed **21 trajectories / 803 live RGB-to-inference frames**.
`verification-v1/` passed RGB/ACK, private process, native-mask and full video
checks. `analysis-v1/` independently recomputed all four diagnostics from saved
RGB and actual model sampling masks. No new fit was launched.

| Method | Nominal warnings / cases | Nominal cases with comparisons | Residue warnings / cases | Residue cases with comparisons |
| --- | ---: | ---: | ---: | ---: |
| Exact meter / raw color | 1/10 | 8/10 | 2/6 | 5/6 |
| Bounded meter interpolation / raw | 1/10 | 8/10 | 4/6 | 5/6 |
| Interpolation / normalized color | 0/10 | 8/10 | 4/6 | 5/6 |
| Interpolation / normalized + camera guard | 0/10 | 8/10 | 4/6 | 5/6 |
| Adaptive replay: guarded correction **before models**, then interpolation | 0/10 | 10/10 | 5/6 | 6/6 |

These are warnings on declared simulated perturbations, not measured real defect
precision/recall. The prior nominal reference is excluded. The scripted 8 mL
prime-to-drain case did not warn in any method. The three appearance probes are
reported separately; two warn because their optics differ from the reference.

Meter interpolation recovered eligible comparisons under slow flow (nominal
11→31) and fast flow (8→19). It recovered the fast-residue warning that exact
matching missed. Normalization removed the blue-attenuation nominal warning.
However, a green-channel perturbation caused the upstream detector to lose
geometry: correcting only sampled color could not recover missing masks.

### Adaptive preprocessing follow-up

After seeing that failure, `preprocess-replay-v1/` reran the **actual frozen
neural models** on corrected saved study RGB. Three prior background patches
estimate sRGB gains. Unhealthy anchors block inference; the gain estimate never
uses private liquid composition, geometric masks or the known perturbation gain.
Raw and corrected pixel identities are recorded separately. This is an adaptive
development follow-up on the same cases, **not a new live or held-out study**.

- Green-attenuation nominal QA: **7/37→25/37**; comparisons **0→25**, no warning.
- Joint slow-flow/green-attenuation residue QA: **1/46→27/46**; comparisons
  **0→27**, with a provisional warning at simulation **2.0 s**.
- Joint slow-flow/green-attenuation nominal QA: **9/46→33/46**; comparisons
  **0→31**, no warning.
- **787/803 frames processed; 16 unhealthy-anchor frames blocked.** The anchor
  occlusion case has 7 corrected QA samples rather than 25 original samples;
  blocked inference also interrupts tracking/history. Blocking has an availability cost.
- The **half-mL-equivalent perturbation still does not warn**, despite 25 eligible
  comparisons. No threshold was tuned to recover it.

`preprocess-analysis-v2/` independently rechecks every source/corrected pixel
identity, clock, correction estimate, blocked-frame boundary and saved diagnostic.
It verifies retained model outputs; it does not claim a second neural-inference
reproduction. The reference and frozen helper hashes are checked as well.

### Stream visibility remains an optics question

Widening the jet alone peaks at **2.83**, below the unchanged start threshold 3.
Shortening the shared absorption depth to 5 mm peaks at **11.19**, and combining
that with the wider jet peaks at **28.94**. Both profiles put all **25/25 active
frames** above the start threshold, with **0 inactive frames** above it. The
image-only fill-start event becomes available at **1.0 s**, **0.4 s after declared
flow starts**. That is simulator-flow timing, not human first-visible-liquid delay.

Those same profiles reduce sample QA from **25/37 to 14/37** and create a
reference-color warning. This is evidence that optics matter, not an approved
renderer fix. Cameron/Hayden and CSI still need to establish plausible glass,
liquid and stream appearance. No complete-vial event is validated: the stationary
carrier never departs.

### Local media and evidence

Under `C:/Users/Austin/ai-club-tools/results/twin-experiments-20261021/`:

- `analysis-v1/comparison.json`, `RESULTS.md`, `experiment-scorecard.png`:
  all predeclared cases and availability counts.
- `analysis-v1/twin-experiment-examples.mp4`: H.264, 166 frames at 5 fps;
  actual live-study outputs, including the missing-geometry case.
- `preprocess-analysis-v2/comparison.json`, `RESULTS.md`,
  `preprocessing-scorecard.png`: adaptive replay results for every case.
- `preprocess-analysis-v2/corrected-pipeline-examples.mp4`: H.264, **120 frames /
  24 s**, single view; corrected nominal, residue warning and blocked-camera cases.
- `preprocess-analysis-v2/corrected-pipeline-warning.png`: retained warning frame.
- `published-media-verification-v2.json`: final H.264 decoding/hashes.

For appearance-transfer preparation and its current checkpoint-access blocker,
see [the Cosmos Transfer trial](COSMOS-TRANSFER-TRIAL.md).

## Reproduce a new batch

From PowerShell in this checkout, with the existing Windows environments:

```powershell
$python = "$HOME/ai-club-tools/envs/nvidia-pilot-windows/Scripts/python.exe"
$plan = "$HOME/ai-club-tools/results/my-twin-study-plan.json"
$out = "$HOME/ai-club-tools/results/my-twin-study"
& $python setup/vision/prepare_twin_experiments.py --out $plan
& $python setup/vision/run_live_twin.py --out $out --plan $plan
```

Every output path must be fresh. The default four-case pilot remains available
by omitting `--plan`. After completion:

```powershell
$analysis = "$HOME/ai-club-tools/envs/vision-alternatives-windows/Scripts/python.exe"
& $analysis setup/vision/verify_live_twin.py --run $out --out "$out-verification"
& $analysis setup/vision/analyze_twin_experiments.py --run $out `
  --verification "$out-verification" --out "$out-analysis"
```

The adaptive replay and independent analysis use fresh output paths too:

```powershell
& $python setup/vision/replay_twin_preprocessing.py --source $out --out "$out-corrected"
& $analysis setup/vision/analyze_twin_preprocessing.py --source $out `
  --replay "$out-corrected" --analysis "$out-analysis" `
  --verification "$out-verification" --out "$out-corrected-analysis"
```

Study preparation/diagnostics are isolated experiments, with no new runtime
installation, fitting job or deployment. Preserve partial directories on failure.
