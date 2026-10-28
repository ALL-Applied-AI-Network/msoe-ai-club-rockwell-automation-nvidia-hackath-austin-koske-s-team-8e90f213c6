# Deploying the line runner on GB10s at CSI

Plan for running the whole pipeline on site:

1. capture of the fill and tray cameras;
2. per-vial verdicts and actions;
3. tray reconciliation;
4. annotated video;
5. RCA.

Everything runs on a CSI GB10, without CSI's inspection software. It builds on the existing
GB10 kit ([setup/gb10](../setup/gb10/README.md), `linewatch-gb10-v1.1.0`). That kit already
runs on spark8 without Docker or root and passes its self-test on CUDA. The kit serves the
per-frame app only. The line runner (`replay.line`, `replay.tray`, `replay.annotate`,
`replay.explain`) is not in it yet.

The plan makes no connection to the line: no PLC, no sorter, no industrial network. Actions are
recommendations shown to the operator and logged. Cameras are UVC devices plugged into the GB10.

## Target layout

```
fill camera (USB, 640×480 MJPEG) ──┐
                                   ├─► GB10-A  capture → line runner → tray → ledger/actions
tray camera (USB, cap_seal view) ──┘           annotated video, rule RCA, operator page
                                                    │  (copy finished vials' packets)
                                                    ▼
                                    GB10-B (optional)  Cosmos/Qwen RCA, spare for A
```

- **GB10-A runs everything that decides.**
  - Two 640×480 streams are light.
  - The kit's models run at about 90 ms per frame on CUDA, HTTP included.
  - The runner samples dwells, not every frame.
- **GB10-B is optional.**
  - It runs the VLM explainer, so a 7B model at about 17 s per vial never competes with the
    decision path for the GPU.
  - It is also the spare if A fails.
- **With only one GB10,** the VLM runs on A after the session, or not at all.
  - Rule RCA is the default anyway. On Run 5 it was consistent with the evidence on 20/20
    vials; zero-shot Cosmos was consistent on 6/20 (see [evaluation](evaluation.md)).
- **Both cameras use the GB10's clock.**
  - The capture step stamps every frame with `CLOCK_MONOTONIC` and wall time.
  - This removes the burned-in-clock matching we needed for Run 5. There, the two clocks agreed
    within about 20 ms.

## Software on the box

| Piece | Status | Work |
| --- | --- | --- |
| Python 3.12, torch cu130, OpenCV, numpy | In the kit lock (`lock-cuda.txt`) | none |
| Fill models (YOLO11n + temporal U-Net) | In the kit; self-test 40/40 on spark8 | none |
| `replay.line/decide/tray/annotate/explain` | On `austin/line-runner` | Add the modules to `build-kit.sh`, plus `integration/runtime.py`, which is already copied |
| Headless capture | Missing | New `replay/capture.py`: OpenCV V4L2, MJPEG 640×480 at 30 fps per camera. Writes 60 s MJPEG segments and a `frames.jsonl` of `(camera, frame, mono_ns, wall)` |
| H.264 output | Missing: spark8 has no `ffmpeg`, and the kit avoids apt | `annotate.py` keeps its OpenCV `mp4v` output when no `ffmpeg` is present. Add `imageio-ffmpeg`, which ships an aarch64 binary, to the lock for H.264 |
| Kit entry points | `START.sh` serves the app only | `START.sh capture`, `START.sh line`, `START.sh selftest-line` |
| VLM explainer (GB10-B) | Ran on spark8 in a separate env (torch 2.11, transformers 5.20) | A separate `cosmos` lock and an offline weight bundle of about 16 GB per 7B model (Cosmos-Reason1-7B, Qwen2.5-VL-7B), carried by USB on exFAT. Load by snapshot path, because model-ID lookup fails offline |

The Linewatch recorder (`Linewatch-recorder`, the Tk capture GUI) stays a laptop tool. On a
GB10 it would need `python3-tk` and `ffmpeg` from apt. We should not assume root on CSI's
GB10s, and our account on spark8 has no Docker or apt. Its vial log format (`vial_log.csv`) is
still the right way to write down ground truth. `capture.py` should write the same columns.

## Calibration on site (per camera mount)

The fill ROI and the Good/Bad tray slot grid (`GOOD_SLOTS`, `TRAYS` in `replay/tray.py`) are
pixel constants measured on the Run 5 view. A new mount needs new values. The plan uses a
reviewed file, not live auto-detection:

1. Run `START.sh capture --snapshot` with the trays empty, then again with the vial under the
   nozzle.
2. Run `python -m replay.tray --calibrate snapshot.png`. It writes `station.json` with the tray
   boxes and the 6×5 slot grid, proposed from the tray outline, and renders an overlay PNG.
3. A person checks the overlay and edits `station.json` if needed. Commit it with the session
   notes.
4. `line`, `tray` and `annotate` read `--station station.json`. The defaults stay the Run 5
   values.

Thresholds such as `OCCUPIED_DIFF`, `CAPPED_YELLOW` and the ΔE limits are **not** retuned on
site. They stay frozen from synthetic data. If they fail on CSI's view, we write that down,
not refit.

## Two run modes

**Mode 1: batch per run (day 1, lowest risk).** Record a full run, then process it:

```bash
bash START.sh capture --fill /dev/v4l/by-id/<fill> --tray /dev/v4l/by-id/<tray> --out ~/csi/<session>/run-01
# ... line runs; Ctrl-C ends the recording
bash START.sh line --run ~/csi/<session>/run-01 --schedule schedule.csv --station station.json
#  -> ledger.csv/json, actions.jsonl, reconciliation.json, rca/, line-and-tray.mp4
```

This is exactly the path we validated on Run 5. The only change is the input source.

**Mode 2: live (day 2, once mode 1 has passed on site).** The runner tails the capture
segments and decides each vial when its dwell closes:

- `replay.line` today calls `decide()` once, after all dwells. That needs to become a per-dwell
  call that appends to `actions.jsonl`. `decide` is already per vial with its previous vial as
  context, so this is a loop change, not a model change.
- `replay.tray` matches each arrival as it happens. The DP already handles in-transit tails.
- The operator page reads the ledger file, which is Josiah's app work.
- Budget: the deadline is the vial's arrival at the tray, at least 100 s after fill end on Run 5
  (median 128 s). The goal is ≤ 2 s from fill end to action, so there is a large margin.

## Validation gates (in order; stop at the first failure)

| Gate | Pass condition | Where |
| --- | --- | --- |
| 1. Kit self-test | `START.sh --selftest`: 40/40 on CUDA | Done on spark8 |
| 2. Line self-test | A Run 5 clip and its samples rebuild the committed ledger and `reconciliation.json` byte for byte, and `line` run with the models matches the verdicts | spark8 before travel |
| 3. Throughput | Runner on the full Run 5 video keeps up with real time (≥ 10 frames/s as sampled) on GB10 CUDA. Record the measured speed in [evaluation](evaluation.md) | spark8 before travel |
| 4. Camera check | Both `/dev/video*` open at 640×480 MJPEG, no dropped frames over 10 min, `mono_ns` gaps < 50 ms | CSI GB10 |
| 5. Framing | Calibration overlay reviewed and `station.json` saved | CSI GB10 |
| 6. Dry run | 3 vials with a known recipe give the right verdicts, all 3 reach the tray in the reconciliation, and the annotated video is readable | CSI GB10 |

Only after gate 6: record the real session, mode 1, and then mode 2 if time allows. The CSI
session is real data. It is evaluation only and is never used for tuning, like Run 5.

## Work items

| # | Item | Size | Who |
| --- | --- | --- | --- |
| 1 | `replay/capture.py`: headless two-camera capture, segments plus frame clock log, `--snapshot` | ½ day | Austin |
| 2 | `--station station.json` in `line`/`tray`/`annotate`, and `tray --calibrate` with an overlay | ½ day | Austin |
| 3 | `annotate.py` falls back to `mp4v`, and `imageio-ffmpeg` goes in the lock (regenerate the locks, rerun the self-test) | 2 h | Austin |
| 4 | `build-kit.sh` copies the line-runner modules; `START.sh capture/line/selftest-line`; kit v1.2.0 | ½ day | Austin |
| 5 | Gates 2 and 3 on spark8, with results recorded in `setup/verification/` | 2 h | Austin |
| 6 | Per-dwell `decide` (live mode) and ledger tail in the app | 1 day | Austin + Josiah |
| 7 | Offline Cosmos bundle (lock, weights, `explain.py --backend hf`) for GB10-B, used for the LoRA once Zoe and Jev have one | ½ day | Zoe/Jev |

Items 1–5 are enough for mode 1 at CSI. Item 6 is the live demo. Item 7 is optional.

## Ask CSI before the visit

- How many GB10s, and is one free for us for the whole session? Is there a user account with
  USB camera access (the `video` group)? Is there internet, or do we bring the offline kit?
- Can we mount our two USB cameras in the Run 5 positions, or use CSI's existing cameras over
  USB (UVC)? We do not use network cameras on their network.
- The recipe schedule for each run, in order (`seq,recipe_id`), and the Recipe Setup limits.
- Can recordings and results leave the site? This decides whether `upload-session.sh` is used.
