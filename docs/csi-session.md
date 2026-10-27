# CSI session quick reference (data collection and live test)

Full plan: [CSI visit guide](../team_handoff/integration/CSI-VISIT-20261027.md)
(PDF: `CSI-Tuesday-Visit-Guide.pdf` on the `temporal-cnn-v1.0.0` release; blank forms in
`guide-v3.zip`). This page adds the tools finished since that guide was written.

## Priorities

1. Raw recording of every fill, unedited, native frames and timestamps.
2. A log of every vial: recipe and mL amounts, previous recipe, physical vial ID, fill
   start/end, completion, recipe changes, who confirmed each item.
3. Reference evidence: a human-confirmed empty vial in position; accepted-good endpoint
   vials per recipe, with the selection rule written down **before** filling.
4. Live app test (secondary). Models and settings stay frozen.

New footage is evaluation-only. Do not connect to line equipment, stage defects or record
approvals that did not happen.

## Set up the GB10

```bash
tar -xzf linewatch-gb10-v1.1.0.tar.gz && cd linewatch-gb10
# No internet: copy wheels/ from linewatch-gb10-offline.tar.gz into this folder first.
bash START.sh --check
bash START.sh --selftest      # must print "Self-test PASSED: 40/40"
bash START.sh                 # http://127.0.0.1:8788 in a browser on the GB10
```

Details: [setup/gb10](../setup/gb10/README.md). Check framing with LIVE CAMERA first: the vial
must sit under the nozzle as in the Run5 clips.

## Record each vial

```bash
ffmpeg -f v4l2 -input_format mjpeg -video_size 640x480 -framerate 30 -i /dev/video0 -c copy vial-A1.mkv
```

`-c copy` keeps the camera's frames and timestamps. Stop with `q`. Start just before the fill
so the first frame shows the empty vial. Name files after the vial log.

## Live test in the app

1. Select the recipe (or **Custom amounts…**) and press **SET RECIPE**.
2. **RECORDING** → drop the vial's clip. Its first frame becomes the empty-vial reference.
3. Read the vial verdict and simulated action; confirm or override.

If the line runs one of recipes 5–14, first test whether the built-in Run5 colour holds (one
vial → IN SPEC?). For new recipes, accept the pre-declared reference vial (e.g. the 3rd, not
the first after a change) with **USE AS REFERENCE**. Vials judged before their reference was
accepted are re-checked offline afterwards. LIVE CAMERA mode does not yet capture an
empty-vial reference, so prefer recordings.

## Ask CSI

- Approved feed/export; may originals and notes leave the site (including GitHub storage)?
- Who confirms empty reference, acceptable endpoint, vial ID and completion?
- Teledyne's per-recipe RGB limits (Recipe Setup) and where/when Teledyne measures.
- What the sorter did with each vial, actual tank contents, any meter/delivery records.

## Close out

Replay a sample of every recording; copy the session folder (`raw/`, `notes/`,
`references/`) and `~/linewatch-data` to the USB and a second location. If CSI approved
off-site storage, from the laptop afterwards:

```bash
bash setup/gb10/upload-session.sh /path/to/csi-20261027/session-01 --dry-run
bash setup/gb10/upload-session.sh /path/to/csi-20261027/session-01
```

This uploads checksummed, split archives to a draft (unpublished) release.
