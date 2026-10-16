# Setup verification

October 16, 2026. Reports and selected smoke images establish which software
paths actually ran. Weights, environments, credentials and full logs live outside
the repository. These checks do not measure detection accuracy or calibrate optics.

| Path | Executed result |
| --- | --- |
| GB10 SSH | Dedicated public key works; account password unchanged |
| Windows data | Bootstrap/import checks passed; all 15 supplied videos decode a first frame; Run 5 has 30 label rows across 10 recipes |
| GB10 data | NumPy, pandas, SciPy, Matplotlib, Pillow, PyAV, OpenCV, scikit-image, scikit-learn and colour-science imports passed |
| GB10 CUDA | Actual tensor calculation passed on GB10 capability 12.1, torch 2.11.0+cu130 |
| GB10 Isaac | Pinned 6.0.0.1 rendered the primitive scene at 320 by 240; first cold start took about 182 seconds |
| GB10 CSI twin | Adapter rendered all 12 supplied examples at 640 by 480 in a warm run of about 32 seconds; 12 unique images and visible orange/blue/height changes |
| GB10 Cosmos | Pinned Cosmos3-Edge downloaded and correctly described the orange/blue rectangle test image |
| Rosie H100 | Slurm jobs 284886 and 284899 passed the Singularity CUDA tensor check on capability 9.0, CUDA 12.9 |
| Rosie Cosmos | Job 284887 passed initially; final environment passed again in job 284894 (COMPLETED, exit 0, 38 seconds including startup) |
| Rosie data | Separate isolated environment passed all ten data imports with NumPy 2.5.4; site CUDA environment retains its own NumPy |
| Offline checks | Five hosted-client/fixture checks and two stale-capture regression checks passed; all Python, Bash and PowerShell scripts parsed |

The original GB10 twin renderer silently produced twelve identical empty-liquid
images. Its `render_status.complete` flag only meant files were captured. The
setup adapter places an unanimated liquid proxy at the frozen world pose instead
of inheriting the source liquid parent's animation. The original renderer and
CAD files remain untouched. `vial-proxy-orange.png` and `vial-proxy-blue.png` were
visually inspected. `vial-capture-check.json` records the full fixture variation
check. The older `vial-render.png`, `vial-render-color.png` and one-image status
are pre-fix diagnostic evidence, **not validated synthetic training examples**.

The check rejects stale or imperceptibly different captures for strongly
different input colors. It does not certify exact fill-volume or optical
accuracy; all source fixture labels remain `uncalibrated_hypothesis`.

Exact package sets are under `requirements/`. The Rosie Cosmos lock records
only the writable overlay; the site SIF still supplies PyTorch and CUDA.
`models.json` records model revisions and the site image path.

Repeated native GB10 launches exposed a Kit startup hang before logging, including
the primitive scene. A debugger launch succeeded; testing the corresponding
address-layout change isolated a working startup path. The ARM launcher now uses
`setarch aarch64 -R` for Isaac only. Consecutive full-fixture launches passed on
that path. This is an empirical workaround for the observed build, not a claim
that every Isaac or GB10 version requires it. Host settings and drivers are unchanged.

Hosted NVIDIA calls were tested through payload validation and mocked transport,
not a real API key. Nano, video generation, Brev provisioning and optional tools
in the resource catalog have not been executed. No training or plant connection
was started. This onboarding does not establish sim-to-real transfer or valid
causal explanations.
