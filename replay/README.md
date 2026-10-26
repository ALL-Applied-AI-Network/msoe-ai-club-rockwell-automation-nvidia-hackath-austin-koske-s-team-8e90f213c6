# Measurement and reference checks

`geometry.py` associates fill-position geometry and builds the sampling mask.
`measurement.py` extracts raw-source RGB/Lab and reports QA outcomes.
`operator/commissioning.py` checks source/reference and recipe qualification.

[Architecture](../docs/architecture.md#vial-association-and-measurement-qa) ·
[Development and tests](../docs/development.md#checks)

These modules do not start a video worker or operator server.
