# Recipe-aware liquid inspection

Vision tools for inspecting vial fills at CSI's Fluid Mixing Station 3. The
selected pipeline uses YOLO for vial geometry and a paired temporal U-Net for
liquid masks, followed by sampling QA and recipe-related operator evidence.

The current release provides the frozen models, paired-image inference,
measurement/reference checks and human evaluation tools. The FastAPI/React app
is still a widgets skeleton; the complete video worker and operator loop need
integration. See the [architecture](docs/architecture.md) for component status.

## Get started

- [Run liquid inference](docs/development.md#run-liquid-inference)
- [Start the API and client](docs/development.md#app-skeleton)
- [Download data and models](docs/data.md)
- [Read measured results](docs/evaluation.md)
- [Prepare the supervised test](docs/operations.md)

[All documentation](docs/README.md) · [Contributing](CONTRIBUTING.md)

## Results

The latest GB10 comparison measures 16.37 ms median CV processing and 105 / 264
QA-eligible samples on inspected Run4 frames. Timing excludes decode, transport
and display. QA eligibility is sample availability, not segmentation or recipe
accuracy. Independent real mask/onset evaluation and physical deployment remain
unverified.

## Project

Built for the Rockwell Automation / NVIDIA Physical AI hackathon with MSOE and
UWM CSI. [Organizer documentation](https://msoe-maic.com/library/?article=Rockwell-NVIDIA-Fall-2026)
provides the event and supplied-data context. This system produces operator
advice; it has no direct industrial control path.
