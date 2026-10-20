# Model continuation — October 20, 2026

Use the primary checkout on **austin-work**. Do not push or merge into main.
Read [the consolidated handoff](../../HANDOFF.md) first.

## Completed comparisons

- [Liquid specialists](LIQUID-SPECIALISTS.md): Slurm 286264, four successful fits,
  four evaluations, 12 fully decoded MP4s and final verification complete.
- [Architecture/resolution pilots](EARLY-LIQUID.md): Slurm 286291, three successful
  fits, all Run 4 subset evaluations and the eight-detector assembly complete.
- [SAM liquid masks](SAM-LIQUID.md): SAM2.1 Tiny probe complete on 30 Run 4 stills,
  with 75 raw masks, 30 overlays and five contact sheets. SAM3 is untested and
  needs authorized checkpoint access.

Do not rerun completed fits or evaluations. Local results live under
C:/Users/Austin/ai-club-tools/results/ in liquid-specialists-20261020,
early-liquid-20261020 and prompted-liquid-20261020/run4-probe.

The completed detector pilots' final development-run4 directory includes
comparison.json, model-scorecard.png and short-target PNGs. Native selected
epochs for nano-960 / small-640 / ResNet are 29 / 27 / 7. Their common-adapter
Run 4 AP50–95 is 53.81 / 53.66 / 53.50 percent. Short supplied-target matches
at .35 are 15/18 / 15/18 / 17/18; there is no <=32px development target.

## Verified artifact retrieval

VPN is restored. Initial management-node reads returned missing/stale metadata
and a NUL-filled completed CSV; original logs and the incomplete local retrieval
were preserved. CPU-only Slurm job 286305 read the originals on dh-dgxh100-2,
verified all three complete training curves/checkpoints and produced
completed-artifacts-node.tar.gz, SHA256
31b86fe156e70d839daa90389030c0f836046682a2ae8d56dfed5dd7ca489a94.

That bundle is local, verified and extracted. collection-read.json binds original
file hashes and source node. retrieval-verification.json records the preserved
incomplete local CSV under retrieval-first-attempt/. No source file was sanitized
or fitted model replaced. The polling collector was stopped after node packaging.

## Next model work

The SAM2 probe supports liquid-box-to-mask refinement in inspected examples,
but vial-only prompts mostly outline glass, including label-absence cases.
A mask-quality score is not fluid-presence probability. Compare prompts from
stronger detectors and a separate presence/localization arm, then forward-only
video masks with resets per vial. Never prompt on later frames to score early
presence. Human pool/stream outlines and first-visible-liquid timing remain
separate evaluation inputs; all current masks and targets are provisional.

Use vision-alternatives-windows locally. Keep Runs 1–3 for fitting, Run 4 for
model development, and Run 5 out of settings/sampling/checkpoint selection.
App defaults, measurement QA, source labels and frozen fitting snapshots remain
unchanged. No industrial connection, automatic adjudication, synthetic rendering,
model promotion, browser preview or subagents without an explicit request.
