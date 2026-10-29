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

## Demo-day checklist

The demo runs the LINE page (`/`) on the CSI GB10 with the station's two cameras. `main` and the
`linewatch-v1.2.0` kit both carry it.

**Before leaving**
- [ ] USB (ext4 or exFAT preferred): `linewatch-gb10-v1.2.0.tar.gz` + `.sha256`,
      `linewatch-gb10-offline.tar.gz` (wheels), `temporal-cnn-assets-v1.zip`, the `run5-vial0*.mp4`
      clips and `linewatch-vial-flow-demo.mp4`.
- [ ] Laptop fallback running: Docker app at http://localhost:8788 ([laptop demo](demo.md)), with
      the LINE page playing Run 5 through *Cameras* → a recorded run.
- [ ] `gh auth login` works on the laptop (to fetch anything missing on site).

**On the GB10 (allow 20 minutes before presenting)**
1. Close CSI's Linewatch recorder (one program per camera).
2. Get and verify the kit ([Get the software](#get-the-software)), then
   [set it up](#set-up-the-gb10): `--check`, `--selftest` (must pass), `bash START.sh`.
3. In the `START.sh` output, every camera link must resolve to a `/dev/videoN`, not
   `NOT FOUND`. If one is missing, check `ls /dev/v4l/by-id/` and the USB hub.
4. Open http://127.0.0.1:8788. LINE → **Live** → *Cameras (live, models now)* →
   *Station cameras* → **Start**. The vial region and tray slots drawn on the feeds must cover the
   vial under the nozzle and the tray; re-aim the camera, not the overlay.
5. Set the recipe order (MONITOR → recipe plan; LINE → **RECIPE ORDER**) before the first vial:
   vial n takes recipe n, counted from **Start**.

**If something fails:** no cameras → *Cameras* → a recorded run (Run 5 through the same input);
kit fails → `--native` / `--docker` via `setup/gb10/onboard.sh`, or `linewatch-v1.1.0`; GB10
unusable → laptop fallback.

Not yet verified on the CSI GB10's own cameras, and the v1.2.0 kit has not been self-tested on GB10
hardware (spark8 was unreachable): run step 2 first.

## Get the software

The repository and its releases are private: the GB10 needs a GitHub login, or bring the files
on USB. Current release: `linewatch-v1.2.0`.

**A. Release download with `gh`** (simplest when `gh` is installed):

```bash
gh auth login        # GitHub.com → HTTPS → "Login with a web browser"; enter the code on any device
cd ~ && gh release download linewatch-v1.2.0 \
  -R ALL-Applied-AI-Network/msoe-ai-club-rockwell-automation-nvidia-hackath-austin-koske-s-team-8e90f213c6 \
  -p 'linewatch-gb10-v1.2.0.tar.gz*' -p 'run5-vial0*.mp4'
sha256sum -c linewatch-gb10-v1.2.0.tar.gz.sha256
```

**B. Without `gh`:** use a fine-grained personal access token with read access to this
repository's contents (create it on a laptop, then `export GITHUB_TOKEN=...` on the GB10):

```bash
REPO=ALL-Applied-AI-Network/msoe-ai-club-rockwell-automation-nvidia-hackath-austin-koske-s-team-8e90f213c6
for name in linewatch-gb10-v1.2.0.tar.gz linewatch-gb10-v1.2.0.tar.gz.sha256; do
  id=$(curl -fsSL -H "Authorization: Bearer $GITHUB_TOKEN" https://api.github.com/repos/$REPO/releases/tags/linewatch-v1.2.0 |
       python3 -c "import json,sys; print(next(a['id'] for a in json.load(sys.stdin)['assets'] if a['name']==sys.argv[1]))" "$name")
  curl -fsSL -H "Authorization: Bearer $GITHUB_TOKEN" -H "Accept: application/octet-stream" \
    https://api.github.com/repos/$REPO/releases/assets/$id -o "$name"
done
sha256sum -c linewatch-gb10-v1.2.0.tar.gz.sha256
```

Or download the same files from the release page in a logged-in browser and copy them over.

**C. Git clone + Docker onboarding** (if the account can use Docker):

```bash
git clone https://github.com/ALL-Applied-AI-Network/msoe-ai-club-rockwell-automation-nvidia-hackath-austin-koske-s-team-8e90f213c6.git linewatch
cd linewatch && bash setup/gb10/onboard.sh     # fetches temporal-cnn-v1.0.0 models; or --assets <USB zip>
```

HTTPS clone asks for a username and the token as password (or run `gh auth login` first).
Onboarding tries Docker GPU, then Docker CPU, then the native launcher, runs the self-test and
saves a device report under `~/linewatch-onboarding/`.

**No internet:** use the USB copy of the kit; for packages, copy `wheels/` from
`linewatch-gb10-offline.tar.gz` into the extracted kit folder.

## Set up the GB10

Copy the kit from the USB to the GB10's own disk and extract it there. A FAT32 stick cannot
hold the bundled Python's symlinks and permissions, so do not run it from the USB.

```bash
cp /media/$USER/<USB>/linewatch-gb10-v1.2.0.tar.gz ~ && cd ~
tar -xzf linewatch-gb10-v1.2.0.tar.gz && cd linewatch-gb10
# No internet: copy wheels/ from linewatch-gb10-offline.tar.gz into this folder first.
bash START.sh --check
bash START.sh --selftest      # must print "Self-test PASSED: 40/40"
bash START.sh                 # http://127.0.0.1:8788 in a browser on the GB10
```

Details: [setup/gb10](../setup/gb10/README.md). Check framing with LIVE CAMERA first: the vial
must sit under the nozzle as in the Run5 clips.

## Record each vial

The app does not record continuous video. In LIVE CAMERA mode it keeps only the frames it
samples (every 1.5 s by default), which is not enough for data collection. Record the native
stream with `ffmpeg` to the GB10's local disk, not the USB:

```bash
mkdir -p ~/csi-20261027/session-01/{raw,notes,references} && cd ~/csi-20261027/session-01/raw
ffmpeg -f v4l2 -input_format mjpeg -video_size 640x480 -framerate 30 -i /dev/video0 -c copy vial-A1.mkv
```

`-c copy` keeps the camera's frames and timestamps. Stop with `q`. Start just before the fill
so the first frame shows the empty vial. Name files after the vial log. Keep each file under
4 GB (per-vial recordings are far smaller) because FAT32 sticks cannot hold larger files.

The camera can usually be opened by only one program at a time: while `ffmpeg` records, the
browser's LIVE CAMERA cannot use it, and the reverse. Record first, then process the clip.

## What is saved where

| Location | Contents |
| --- | --- |
| `~/csi-20261027/session-01/raw/` | Native recordings from `ffmpeg` (the data-collection record) |
| `~/csi-20261027/session-01/notes/` | Vial log, forms, CSI answers, photos of Recipe Setup |
| `~/linewatch-data/` | App results: each processed frame's original, overlay and mask, SQLite results/QA reasons, a copy of every dropped recording, operator decisions |
| `~/linewatch-data/selftest-*/` | Self-test reports |

## Live test in the app

**LINE page (main):** follow steps 4–5 of the [demo-day checklist](#demo-day-checklist). Each
session records both feeds and the model samples and saves itself as a run; **Stop** links to it
in Review. This does not replace the `ffmpeg` raw recording above when collecting data.

**MONITOR page (`/monitor`, single vial):**

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

Replay a sample of every recording, then copy both folders to the USB and a second location:

```bash
cp -r ~/csi-20261027 ~/linewatch-data /media/$USER/<USB>/
``` If CSI approved
off-site storage, from the laptop afterwards:

```bash
bash setup/gb10/upload-session.sh /path/to/csi-20261027/session-01 --dry-run
bash setup/gb10/upload-session.sh /path/to/csi-20261027/session-01
```

This uploads checksummed, split archives to a draft (unpublished) release.
