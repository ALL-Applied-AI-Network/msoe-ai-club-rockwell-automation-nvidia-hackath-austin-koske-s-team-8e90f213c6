"""Frame-input boundary. Only an existing-recording adapter is provided."""
from dataclasses import dataclass
from typing import Iterator

import av
import numpy as np


@dataclass(frozen=True)
class FramePacket:
    frame_index: int
    video_s: float | None
    rgb: np.ndarray
    pts: int | None = None
    time_base: str | None = None
    estimated_wall_elapsed_s: float | None = None


class RecordingSource:
    """An adapter owns decode/transport; monitoring owns all inspection logic."""
    def __init__(self, camera, stride, clock=None):
        self.camera, self.stride, self.clock = camera, stride, clock
        self.decoded_frames = 0

    def __iter__(self) -> Iterator[FramePacket]:
        with av.open(self.camera['path']) as container:
            stream = container.streams.video[0]
            stream.thread_type = 'AUTO'
            for index, frame in enumerate(container.decode(stream)):
                self.decoded_frames = index + 1
                if index % self.stride:
                    continue
                yield FramePacket(index,
                    float(frame.pts * frame.time_base) - self.camera['start_pts_s'] if frame.pts is not None else None,
                    frame.to_ndarray(format='rgb24'), int(frame.pts) if frame.pts is not None else None,
                    str(frame.time_base), float(self.clock[index]) if self.clock is not None else None)
