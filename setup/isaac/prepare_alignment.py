"""Extract one fixed Run 1 camera-alignment frame, without evaluation labels."""
import argparse
from pathlib import Path

import cv2

from box_annotations import sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.video.name != "vial_fill_video_1.mp4":
        parser.error("This fixed alignment reference requires vial_fill_video_1.mp4")
    if args.out.exists():
        parser.error("Use a new output PNG")
    capture = cv2.VideoCapture(str(args.video))
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        if abs(fps-30) > 1e-6:
            raise ValueError("Expected native 30 fps MP4")
        # Sequential decode avoids keyframe seeking ambiguity for the review reference.
        for _ in range(2101):
            ok, frame = capture.read()
            if not ok:
                raise ValueError("Could not decode zero-based frame 2100")
        if frame.shape != (480, 640, 3):
            raise ValueError("Expected native 640x480 camera image")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(args.out), frame):
            raise OSError("Could not save alignment PNG")
        write_json(args.out.with_suffix(".json"), dict(source_video=str(args.video.resolve()),
            source_video_sha256=sha256(args.video), source_run=1, frame_index_zero_based=2100,
            mp4_timestamp_seconds=70.0, fps=fps, image_sha256=sha256(args.out),
            usage="visual alignment review only; no evaluation labels or camera fitting"))
    finally:
        capture.release()


if __name__ == "__main__":
    main()
