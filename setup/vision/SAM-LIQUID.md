# Prompted masks in the liquid pipeline

SAM is a useful candidate for the **mask-proposal stage**. The first-visible-pool
decision and the color-sampling decision should remain separate: a tentative
liquid detection can exist before enough clean pixels/stability are available
for measurement. Keep the current detector, app defaults and QA while testing.

```text
recorded RGB -> current vial / track -> liquid presence + localization
                                         |
                                         v
                               SAM raw mask proposal
                                  /             \
                    early-presence evidence    sampling interior + existing QA
```

This is a proposed experimental composition, not an implemented production path.
The replay already has an independent segmentation proposal interface in
`replay/segmentation.py`. Its existing tiny U-Net predicts an inset sampling
interior, which is distinct from a full fluid outline. Its earlier trial passed
8/30 Run 5 windows versus the unchanged baseline's 13/30; see
[the segmentation audit](SEGMENTATION.md). Better-looking outlines do not by
themselves validate color sampling or physical onset timing.

## SAM2 and SAM3 roles

- [SAM2/SAM2.1](https://github.com/facebookresearch/sam2) supports point/box/mask
  prompting and video propagation. Use a current-frame liquid prompt for outline
  refinement, then test causal forward propagation with state reset per vial.
- [SAM3](https://github.com/facebookresearch/sam3) adds concept/text prompting.
  It is a candidate for finding a liquid region without requiring an existing
  liquid box. Whether short/pale liquid is recognized is an empirical question.
  Its official checkpoint requires access approval; no SAM3 access request,
  authentication change, term acceptance or inference was performed here.
- A vial-only prompt asks a prompt-conditioned model for a region. It does not
  establish fluid presence. The experiment below demonstrates why a separate
  presence/localization decision is necessary for that prompt strategy.

For onset evaluation, use only current/past RGB and prompts available then.
Prompting a later filled image and propagating backward leaks future evidence.
Do not carry memory across vials, initialize from teacher boxes, or use reviewed
timing annotations as inference inputs. Future human mask review may support
training a compact supervised mask head; raw SAM output is not approved training
truth and is not exported as corrected annotations here.

## Executed SAM2.1 Run 4 probe

Root: `C:/Users/Austin/ai-club-tools/results/prompted-liquid-20261020/run4-probe`.
`probe_prompted_liquid.py` ran **30 original Run 4 stills**: all 18 supplied liquid
targets <=64 pixels high, plus the first 12 chronological images without a
supplied liquid target. Their absence is provisional, not reviewed empty glass.
Run 5 was not read. Original image/label hashes were verified before inference.

Model: official Meta **SAM2.1 Hiera Tiny**, accessed through the installed
Ultralytics 8.4.182 image predictor. The checkpoint is byte-pinned as
`7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69`.
It came from the [official checkpoint link](https://github.com/facebookresearch/sam2#download-checkpoints),
not a substituted model. Source/transform details, script snapshot and runtime
are retained in `probe-plan.json`, `executed-probe.py` and `comparison.json`.

All prompts use **predictions from the same frozen baseline on the current
image**, never teacher boxes. The vial prompt follows the existing active-station
profile. The lower-vial search prompt covers x10%–90%, y55%–100% of that vial;
these are a fixed search prior, not fluid-presence labels or a new QA rule.
SAM input is 1024. All raw returned masks/scores are retained at score floor 0,
without selecting a fluid mask by the model's mask-quality score.

| Prompt | Supplied-positive cases yielding masks | Label-absence cases yielding masks | Observation |
| --- | ---: | ---: | --- |
| Baseline liquid box | 15/18 | 0/12 | Lower colored regions in inspected examples; inherits the detector's missing prompts |
| Full vial box | 18/18 | 12/12 | Predominantly vial/glass outlines in inspected examples |
| Lower-vial search box | 18/18 | 12/12 | Predominantly lower glass regions; presence remains unresolved |

These are **output counts, not mask accuracy, sensitivity or specificity**.
There is no reviewed mask truth or first-visible-liquid timing in this probe.
The liquid-box arm generated nothing in three positive cases because the
baseline supplied no eligible liquid prompt. It cannot recover those misses by
refining the same absent boxes. Full-vial prompts returned quality scores around
.92–.94 on several label-absence examples while outlining the glass; mask quality
is not a calibrated probability of fluid presence.

The [probe verification](../verification/sam2-liquid-probe-20261020.json) records
hash checks and inspected cases. All individual masks are source-aligned 640×480
binary PNGs under `masks/`. Thirty four-panel overlays and five contact sheets
show source RGB, liquid-box, full-vial and lower-vial outputs. Their geometry
statistics describe raw masks; they do not adjudicate fluid pixels or pass QA.

## Offline pipeline video — October 20

`render_liquid_pipeline.py` applies the completed Faster R-CNN ResNet50 liquid
pilot, fixed baseline vial detector and SAM2.1 Tiny to the **first 60 seconds of
Run 4**, independently on every sixth decoded frame. Native PTS and exact input
RGB hashes are retained. Playback is 5 fps, preserving source elapsed time.
This is a development demonstration with no source labels supplied to inference.

Root: `C:/Users/Austin/ai-club-tools/results/liquid-pipeline-demo-20261020/render-v1`.
Open `liquid-pipeline.mp4`; `pipeline-frame-0600.png` shows a later filled region,
`pipeline-frame-0420.png` shows an oversized prompt failure, and
`decoded-storyboard.png` contains six frames decoded from the actual MP4.

Four synchronized panels show source pixels, detector proposals, current-frame
SAM masks, and existing per-frame sampling QA. An offline adapter supplies the
SAM proposals to the unchanged geometry/measurement gates. Geometry evidence is
explicitly identified as SAM rather than the generic adapter's TinyUNet default.
There is no temporal SAM propagation, future-frame prompting, reference-bank
reuse, full-window QA, recipe decision or physical vial identification.
Multiple contained masks remain ambiguous; mask-quality scores do not decide
fluid presence. A pixel hash guard rejects stale-frame proposals.

All **300 output frames** fully decode; all 300 source pixel hashes/native PTS
were independently rechecked. There are 103 frames with liquid detections,
89 with eligible prompts, 109 raw masks, 68 with nonempty final sampling masks,
and 49 passing existing per-frame gates. These are execution counts, **not
accuracy, mask IoU, onset recall or product-quality outcomes**. The first
detector proposal at source PTS 10.8 s is not a reviewed physical-onset time.
At 14 s an oversized prompt includes glass/upper structure; at 20 s a tighter
prompt follows the filled region. Approved outlines/timing remain human work.

`frames.jsonl` retains original floating-point detections, prompt boxes, raw mask
RLE, per-frame quality flags and final sample masks. `manifest.json` pins source,
models, executed script, QA code/configuration and output hashes. The
[verification record](../verification/liquid-pipeline-video-20261020.json)
captures checks and limits. Three focused tests cover stale-frame rejection,
mask clearing/copying and ambiguity rejection through existing geometry.

Reproduce with the vision-alternatives-windows Python, using a fresh output:

```powershell
python setup/vision/render_liquid_pipeline.py --source datasets/dataset2/csi-vial-fill-dataset/vial_fill/videos/vial_fill_video_4.mp4 --experiment C:/Users/Austin/ai-club-tools/results/early-liquid-20261020 --baseline C:/Users/Austin/ai-club-tools/results/crop-clahe-detector-20261019/baseline-best.pt --sam C:/Users/Austin/ai-club-tools/results/prompted-liquid-assets-20261020/sam2.1_t.pt --out C:/Users/Austin/ai-club-tools/results/liquid-pipeline-demo-20261020/NEW-RENDER
```

## Next controlled comparison

1. Compare detector-to-SAM refinement with the existing box/heuristic-mask path
   on identical development frames, using the completed detector pilots.
2. Keep a separate presence arm so missing/ambiguous liquid boxes remain visible
   failures. A SAM3 concept arm is an option once authorized weights are present.
3. Compare forward-only video propagation against independent-frame masks. Reset
   state at vial changes and log every prompt's source frame/availability.
4. Evaluate reviewed pool outlines, leakage onto glass/label/holder/stream,
   first-visible-pool delay, absent-liquid alarms, temporal flicker and runtime.
   Retain hard pale/clear cases and label uncertainty. Existing unreviewed boxes
   and mask-quality scores cannot supply those ground-truth outcomes.
5. Compare exact inset masks under unchanged window-level measurement QA. The
   demonstration above exercises only the per-frame gates. No detector or
   segmenter is promoted here.

Reproduction, using the existing vision-alternatives-windows Python and a fresh
output path:

```powershell
python setup/vision/probe_prompted_liquid.py --experiment C:/Users/Austin/ai-club-tools/results/liquid-specialists-20261020 --baseline C:/Users/Austin/ai-club-tools/results/crop-clahe-detector-20261019/baseline-best.pt --checkpoint C:/Users/Austin/ai-club-tools/results/prompted-liquid-assets-20261020/sam2.1_t.pt --out C:/Users/Austin/ai-club-tools/results/prompted-liquid-20261020/NEW-PROBE
```
