from dataclasses import asdict, dataclass
import hashlib
import json
import math
import re


@dataclass(frozen=True)
class Config:
    schema_version: int = 1
    vision_model_path: str | None = None
    vision_model_sha256: str | None = None
    sample_stride: int = 6
    image_size: tuple = (640, 480)
    color_roi: tuple = (293, 345, 325, 385)
    holder_roi: tuple = (264, 231, 338, 253)
    nozzle_roi: tuple = (298, 70, 314, 135)
    valid_fraction_min: float = .6
    holder_dark_fraction_min: float = .35
    max_motion: float = 3.0
    max_roi_sensitivity_de2000: float = 8.0
    max_temporal_de2000: float = 5.0
    review_threshold_de2000: float = 5.0
    window_start_after_end_s: float = .5
    window_end_after_end_s: float = 1.0
    minimum_window_samples: int = 3
    stream_center_roi: tuple = (302, 75, 311, 130)
    stream_left_roi: tuple = (290, 75, 298, 130)
    stream_right_roi: tuple = (316, 75, 324, 130)
    stream_on_contrast: float = 3.0
    stream_off_contrast: float = 2.5
    stream_peak_min: float = 6.0
    stream_confirm_s: float = .4
    minimum_stream_s: float = .6
    departure_motion: float = 8.0
    departure_confirm_s: float = .2
    carrier_settle_s: float = .6

    def __post_init__(self):
        if bool(self.vision_model_path) != bool(self.vision_model_sha256):
            raise ValueError('Detector path and SHA256 must be supplied together')
        if self.vision_model_sha256 and not re.fullmatch(r'[0-9a-f]{64}',self.vision_model_sha256):
            raise ValueError('Detector SHA256 must be 64 lowercase hexadecimal characters')
        if (not isinstance(self.sample_stride, int) or not isinstance(self.minimum_window_samples, int) or
                self.sample_stride < 1 or self.minimum_window_samples < 1):
            raise ValueError('Stride and sample count must be positive')
        numeric = [value for value in asdict(self).values() if isinstance(value, (int, float))]
        if not all(math.isfinite(value) for value in numeric):
            raise ValueError('Configuration numbers must be finite')
        if not 0 <= self.valid_fraction_min <= 1 or not 0 <= self.holder_dark_fraction_min <= 1:
            raise ValueError('QA fractions must lie between zero and one')
        if not 0 <= self.window_start_after_end_s < self.window_end_after_end_s:
            raise ValueError('Measurement window must follow fill completion')
        if self.review_threshold_de2000 <= 0:
            raise ValueError('Review threshold must be positive')
        if not 0 <= self.stream_off_contrast < self.stream_on_contrast <= self.stream_peak_min:
            raise ValueError('Stream hysteresis/peak thresholds are inconsistent')
        if min(self.stream_confirm_s, self.minimum_stream_s, self.departure_confirm_s, self.carrier_settle_s, self.departure_motion) <= 0:
            raise ValueError('Detector durations and motion threshold must be positive')
        width, height = self.image_size
        for x0, y0, x1, y1 in (self.color_roi, self.holder_roi, self.nozzle_roi,
                              self.stream_center_roi, self.stream_left_roi, self.stream_right_roi):
            if not 0 <= x0 < x1 <= width or not 0 <= y0 < y1 <= height:
                raise ValueError('ROI lies outside the configured image')
        if len({box[3] - box[1] for box in (self.stream_center_roi, self.stream_left_roi, self.stream_right_roi)}) != 1:
            raise ValueError('Stream regions must have equal heights')

    def to_dict(self):
        return {key: list(value) if isinstance(value, tuple) else value
                for key, value in asdict(self).items()}

    @property
    def fingerprint(self):
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()[:16]


def load_config(path=None):
    if path is None:
        return Config()
    with open(path, encoding='utf-8-sig') as handle:
        values = json.load(handle)
    for name in ('image_size', 'color_roi', 'holder_roi', 'nozzle_roi',
                 'stream_center_roi', 'stream_left_roi', 'stream_right_roi'):
        if name in values:
            values[name] = tuple(values[name])
    return Config(**values)
