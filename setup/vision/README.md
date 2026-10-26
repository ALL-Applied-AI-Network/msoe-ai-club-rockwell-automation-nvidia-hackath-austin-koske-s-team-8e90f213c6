# Vision models

The selected liquid model is the paired-difference U-Net in `train_pair_masks.py`.
`temporal_pool.py` loads the frozen checkpoint and runs current/reference RGB
inference; `verify_temporal_handoff.py` replays the released example predictions.
Other scripts support the completed evaluation and stress experiments.

[Architecture](../../docs/architecture.md#liquid-model) ·
[Run instructions](../../docs/development.md#run-liquid-inference) ·
[Data and checkpoints](../../docs/data.md) · [Results](../../docs/evaluation.md)

Experiment scripts may require the original external bundles, exact source
clocks and configurations. They are not interchangeable with the model-only CLI.
