"""Long-lived frame pipeline adapters.

MockInspectionPipeline keeps UI/API development deterministic. The configured
TemporalCnnInspectionBridge loads the shared batch pipeline models once and
preserves measurement-QA state independently for each inspection run.
"""

