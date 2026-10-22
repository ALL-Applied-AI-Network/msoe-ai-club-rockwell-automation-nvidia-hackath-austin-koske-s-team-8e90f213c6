# Team 3 SAM2.1 four-frame pilot

`run_sam_pilot.py` verifies the SAM frame manifest entries and image hashes,
then prompts SAM2.1 Hiera Tiny with each valid class 1 `liquid` box for the
selected Run 4 development frames. It does not use class 0 `vial` boxes as pool
prompts. Every proposal is kept as an independent candidate.

The runner records source frame identity and native PTS, prompt and detector
provenance, checkpoint hash, SAM score and elapsed time. It writes one 0/255
full-frame PNG mask and one overlay per prompted proposal, plus `result.json`.
When no liquid proposal exists, it records `no_liquid_proposal` and writes no
mask. Malformed or out-of-bounds proposals are reported as unavailable.

## Spark run

The frame package should be staged outside Git at
`~/ai-club-tools/datasets/liquid-team-data-v1/sam_starter/`. Use the provisioned
`cosmos` environment through the Spark launcher to serialize GPU work:

```bash
bash ~/ai-club-onboarding/setup/spark/run.sh cosmos \
  ~/fluid-mixing-pipeline/team3/run_sam_pilot.py \
  --data-root ~/ai-club-tools/datasets/liquid-team-data-v1 \
  --checkpoint ~/fluid-mixing-pipeline/checkpoints/sam2.1_hiera_tiny.pt \
  --output-dir "$HOME/ai-club-tools/results/$USER/team3-sam-pilot-run01" \
  --device cuda
```

The default frame indices are 330, 450, 750 and 900. Override them with
`--frames`. Change `run01` to a fresh name for each run; the runner refuses to
overwrite an existing directory. Keep generated results and model weights out
of Git.

These are unreviewed development proposals. SAM scores are not calibrated mask
quality, and no mask-accuracy claim is supported without reviewed references.

## Compare with reviewed masks

Use `evaluate_masks.py` after copying a pilot result's `masks/` directory next
to its `result.json`. It checks source dimensions, reports IoU and Dice for
each box proposal, and writes `mask-evaluation.json` beside the pilot result:

```bash
python ~/fluid-mixing-pipeline/team3/evaluate_masks.py \
  --pilot-dir ~/ai-club-tools/results/$USER/team3-sam-pilot-run01 \
  --annotation-dir ~/ai-club-tools/results/$USER/team3-mask-review/run4
```

The annotation manifest must contain one 0/255 mask PNG per source frame and
declare the split. Run 4 masks are development references for prompt comparison
and evaluation; do not use them to fine-tune model weights.
