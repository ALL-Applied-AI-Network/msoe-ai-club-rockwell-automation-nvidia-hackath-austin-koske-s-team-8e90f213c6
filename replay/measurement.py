"""Tracked liquid-mask measurement, past-frame QA; no evaluation labels."""
import cv2
import numpy as np
from skimage.color import deltaE_ciede2000
from .geometry import VialGeometry, decode_mask, encode_mask


def delta_e(left, right):
    return float(deltaE_ciede2000(np.asarray(left, dtype=float), np.asarray(right, dtype=float)))


def rgb_to_lab(rgb):
    # Same sRGB/D65 camera-feature convention as the earlier observation pilot.
    x = np.asarray(rgb, dtype=float) / 255
    linear = np.where(x <= .04045, x / 12.92, ((x + .055) / 1.055) ** 2.4)
    matrix = np.array([[.4124564, .3575761, .1804375],
                       [.2126729, .7151522, .0721750],
                       [.0193339, .1191920, .9503041]])
    xyz = linear @ matrix.T / [.95047, 1., 1.08883]
    delta = 6 / 29
    fx = np.where(xyz > delta ** 3, np.cbrt(xyz), xyz / (3 * delta ** 2) + 4 / 29)
    return np.array([116 * fx[1] - 16, 500 * (fx[0] - fx[1]), 200 * (fx[1] - fx[2])])


def optical_density(lab):
    """-ln(linear RGB) of a Lab colour, inverting rgb_to_lab. Dye mixtures add in this space."""
    L, a, b = np.asarray(lab, dtype=float)
    fy = (L + 16) / 116
    f = np.array([fy + a / 500, fy, fy - b / 200])
    delta = 6 / 29
    xyz = np.where(f > delta, f ** 3, 3 * delta ** 2 * (f - 4 / 29)) * [.95047, 1., 1.08883]
    matrix = np.array([[3.2404542, -1.5371385, -.4985314],
                       [-.9692660, 1.8760108, .0415560],
                       [.0556434, -.2040259, 1.0572252]])
    return -np.log(np.clip(matrix @ xyz, 1e-4, 1))


def crop(rgb, box):
    x0, y0, x1, y1 = box
    return rgb[y0:y1, x0:x1]


def color(pixels, trim=True):
    values = pixels.reshape(-1, 3)
    valid = (values.max(axis=1) < 245) & (values.max(axis=1) > 3)
    if not valid.any():
        return None, None, 0.
    selected = values[valid]
    luma = selected @ [.2126, .7152, .0722]
    if trim:
        low, high = np.percentile(luma, [5, 95])
        selected = selected[(luma >= low) & (luma <= high)]
    rgb = np.median(selected, axis=0)
    return rgb.tolist(), rgb_to_lab(rgb).tolist(), float(valid.mean())


class Extractor:
    def __init__(self, config, detector=None):
        self.config = config
        self.previous_thumb = None
        self.previous_nozzle = None
        self.geometry = VialGeometry(config,detector=detector)

    def measure(self, rgb):
        config = self.config
        if (rgb.shape[1], rgb.shape[0]) != tuple(config.image_size):
            raise ValueError(f'ROI profile expects {config.image_size}; received {rgb.shape[1::-1]}')
        thumb = cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), (32, 24)).astype(float)[10:23, 11:21]
        nozzle = crop(rgb, config.nozzle_roi).astype(float)
        center = crop(rgb, config.stream_center_roi).astype(float)
        sides = np.concatenate([crop(rgb, box) for box in (config.stream_left_roi, config.stream_right_roi)], axis=1)
        background = np.median(sides, axis=1)
        # Same-frame spatial contrast: works even when recording begins mid-flow.
        # A median across rows suppresses isolated drops; no label/template needed.
        stream_contrast = float(np.median(np.abs(center - background[:, None, :]).mean(axis=(1, 2))))
        motion = float(np.abs(thumb - self.previous_thumb).mean()) if self.previous_thumb is not None else None
        nozzle_change = float(np.abs(nozzle - self.previous_nozzle).mean()) if self.previous_nozzle is not None else None
        self.previous_thumb, self.previous_nozzle = thumb, nozzle
        geometry = self.geometry.update(rgb, moving=motion is not None and motion > config.max_motion)
        mask = decode_mask(geometry['mask'], rgb.shape[:2]).astype(bool)
        proposed_count = int(mask.sum())
        mask &= (rgb.max(axis=2) < 245) & (rgb.max(axis=2) > 3)
        fraction = float(mask.sum() / proposed_count) if proposed_count else 0.
        if mask.any():
            luma = rgb @ np.array([.2126, .7152, .0722])
            low, high = np.percentile(luma[mask], [5,95])
            mask &= (luma >= low) & (luma <= high)
        # Export the final actual sample mask, including clipping/luma trimming.
        geometry.update(mask=encode_mask(mask), mask_pixels=int(mask.sum()))
        median_rgb, lab, _ = color(rgb[mask], trim=False) if mask.any() else (None, None, 0.)
        holder_fraction = float((crop(rgb, config.holder_roi).max(axis=2) < 80).mean())
        sensitivity = None
        if lab is not None:
            shifts = []
            # Erosion sensitivity stays inside the candidate liquid, rather
            # than shifting a rectangle onto the wall or glass edge.
            for radius in (1, 2, 3):
                interior = cv2.erode(mask.astype(np.uint8), np.ones((2*radius+1, 2*radius+1),np.uint8)).astype(bool)
                _, shifted, _ = color(rgb[interior], trim=False) if interior.any() else (None,None,0.)
                if shifted is not None:
                    shifts.append(delta_e(lab, shifted))
            sensitivity = max(shifts, default=0.)
        flags = list(geometry['quality_flags'])
        if geometry['mask_pixels'] < 60:
            flags.append('insufficient_final_liquid_pixels')
        if fraction < config.valid_fraction_min:
            flags.append('insufficient_unclipped_pixels')
        if holder_fraction < config.holder_dark_fraction_min:
            flags.append('holder_geometry_not_confirmed')
        if motion is None:
            flags.append('no_prior_frame_for_motion_check')
        elif motion > config.max_motion:
            flags.append('carrier_motion')
        if sensitivity is not None and sensitivity > config.max_roi_sensitivity_de2000:
            flags.append('roi_sensitive_to_position')
        return dict(rgb=median_rgb, lab=lab, valid_fraction=fraction,
                    holder_dark_fraction=holder_fraction, motion=motion,
                    nozzle_change=nozzle_change, stream_contrast=stream_contrast, roi_sensitivity_de2000=sensitivity,
                    geometry=geometry, quality_flags=flags, measurement_valid=not flags and lab is not None)
