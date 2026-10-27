"""Long-lived frame pipeline adapters.

Replace MockInspectionPipeline with adapters around the Team 1 detector, Team 2
liquid detector, and Team 3 SAM refiner. Models should be loaded once at startup,
not launched as a new CLI process for every frame.
"""

