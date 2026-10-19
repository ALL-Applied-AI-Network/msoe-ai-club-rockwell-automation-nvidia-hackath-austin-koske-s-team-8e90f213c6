"""Read-only catalog. Recipe annotations are replay context, never defect labels."""
import csv
from dataclasses import asdict, dataclass
import hashlib
import math
from pathlib import Path

import av

CAMERAS = ('vial_fill', 'tank_level', 'cap_seal')
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = REPO_ROOT / 'datasets/dataset2/csi-vial-fill-dataset'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class Annotation:
    vial_id: int
    recipe_id: str
    start_frame: int
    end_frame: int
    volumes_ml: dict
    dispense_order: str
    position_in_batch: int
    previous_recipe_id: str | None

    def to_dict(self):
        return asdict(self)

    def context(self):
        # The monitor sees recipe availability and commands, not a future end.
        return dict(vial_id=self.vial_id, recipe_id=self.recipe_id,
                    available_frame=self.start_frame, volumes_ml=self.volumes_ml,
                    dispense_order=self.dispense_order, position_in_batch=self.position_in_batch,
                    previous_recipe_id=self.previous_recipe_id,
                    source='provided_recipe_annotation')


def read_annotations(path, frame_count):
    if path is None:
        return []
    with Path(path).open(newline='', encoding='utf-8-sig') as handle:
        rows = list(csv.DictReader(handle))
    annotations, seen = [], set()
    previous_recipe, previous_batch_recipe, previous_end, position = None, None, -1, 0
    for row in rows:
        vial_id = int(row['vial'])
        recipe = str(row['recipe_id']).strip()
        start, end = int(row['start_frame']), int(row['end_frame'])
        if not recipe or vial_id in seen or not previous_end < start <= end < frame_count:
            raise ValueError(f'Invalid, overlapping or duplicate annotation for vial {vial_id}')
        volumes = {f'T{tank}': float(row.get(f't{tank}_ml') or 0) for tank in ('100', '200', '400')}
        if any(not math.isfinite(value) or value < 0 for value in volumes.values()) or sum(volumes.values()) <= 0:
            raise ValueError(f'Invalid commanded volumes for vial {vial_id}')
        order = row['dispense_order'].split('>')
        if len(order) != len(set(order)) or set(order) != {tank for tank, volume in volumes.items() if volume > 0}:
            raise ValueError(f'Dispense order does not match the commanded tanks for vial {vial_id}')
        if recipe == previous_recipe:
            position += 1
        else:
            previous_batch_recipe, position = previous_recipe, 1
        annotations.append(Annotation(vial_id, recipe, start, end, volumes,
                                      row['dispense_order'], position, previous_batch_recipe))
        seen.add(vial_id)
        previous_recipe, previous_end = recipe, end
    return annotations


class Catalog:
    def __init__(self, data=DEFAULT_DATA, labels_dir=None, clock_map_dir=None, include_annotations=True):
        self.data = Path(data).expanduser().resolve()
        self.labels_dir = Path(labels_dir).expanduser().resolve() if labels_dir else None
        self.clock_map_dir = Path(clock_map_dir).expanduser().resolve() if clock_map_dir else None
        for folder in (self.labels_dir, self.clock_map_dir):
            if folder is not None and not folder.is_dir():
                raise FileNotFoundError(f'Override folder does not exist: {folder}')
        self.runs = {}
        for run_id in range(1, 6):
            cameras = {}
            for camera in CAMERAS:
                path = self.data / camera / 'videos' / f'{camera}_video_{run_id}.mp4'
                if not path.is_file():
                    continue
                with av.open(str(path)) as container:
                    stream = container.streams.video[0]
                    fps = float(stream.average_rate or stream.guessed_rate or 30)
                    start = float((stream.start_time or 0) * stream.time_base)
                    duration = float(stream.duration * stream.time_base) if stream.duration else container.duration / av.time_base
                    count = int(stream.frames or round(duration * fps))
                    cameras[camera] = dict(path=str(path), fps=fps, frames=count,
                                           width=stream.width, height=stream.height,
                                           duration_s=duration, start_pts_s=start,
                                           codec=stream.codec_context.name, sha256=sha256(path))
            if 'vial_fill' not in cameras:
                continue
            override = self.labels_dir / f'run{run_id}.csv' if self.labels_dir else None
            supplied = self.data / 'vial_fill/labels' / f'video{run_id}_vial_recipes.csv'
            annotation_path = (override if override and override.is_file() else supplied if supplied.is_file() else None) if include_annotations else None
            annotations = read_annotations(annotation_path, cameras['vial_fill']['frames'])
            clock = self.clock_map_dir / f'framemap_v{run_id}.npy' if self.clock_map_dir else None
            self.runs[run_id] = dict(run_id=run_id, cameras=cameras, annotations=annotations,
                                    annotation_path=str(annotation_path) if annotation_path else None,
                                    annotation_sha256=sha256(annotation_path) if annotation_path else None,
                                    clock_path=str(clock) if clock and clock.is_file() else None,
                                    clock_sha256=sha256(clock) if clock and clock.is_file() else None)
        if not self.runs:
            raise FileNotFoundError(f'No vial_fill recordings found in {self.data}')

    def public(self):
        return [dict(run_id=key, cameras={name: {k: v for k, v in camera.items() if k != 'path'}
                                         for name, camera in run['cameras'].items()},
                     annotations=[item.to_dict() for item in run['annotations']],
                     context_mode='annotated_replay' if run['annotations'] else 'unlabeled_recording',
                     clock_status='unreviewed OCR estimate' if run['clock_path'] else 'MP4 PTS only; wall time unavailable',
                     sync_status='container-time alignment; physical cross-camera alignment unverified')
                for key, run in self.runs.items()]
