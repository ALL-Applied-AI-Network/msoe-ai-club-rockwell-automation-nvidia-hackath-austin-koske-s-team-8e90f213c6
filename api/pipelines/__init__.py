"""Long-lived frame pipeline adapters.

MockInspectionPipeline keeps UI/API development deterministic. In model mode,
TemporalInspectionPipeline runs the released asset bundle (YOLO vial geometry, paired
temporal U-Net, measurement QA); without a bundle, TemporalCnnInspectionBridge loads the
configured checkpoints through the shared batch pipeline. Both keep measurement-QA state
per inspection run.
"""

