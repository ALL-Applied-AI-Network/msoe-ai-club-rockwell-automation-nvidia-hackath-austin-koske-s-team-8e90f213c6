"""Read-only input check. Does not clean, relabel, or change the supplied data."""
import argparse
import csv
import json
from pathlib import Path
import av

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=Path, default=Path(__file__).resolve().parents[1] / 'datasets/dataset2/csi-vial-fill-dataset')
parser.add_argument('--out', type=Path)
args = parser.parse_args()
report = {'data': str(args.data.resolve()), 'videos': [], 'ok': True,
          'timing_note': 'Frame rate is container metadata, not verified wall-clock capture rate.'}
for camera in ['vial_fill', 'tank_level', 'cap_seal']:
    for run in range(1, 6):
        path = args.data / camera / 'videos' / f'{camera}_video_{run}.mp4'
        row = {'camera': camera, 'run': run}
        try:
            with av.open(str(path)) as container:
                stream = container.streams.video[0]
                frame = next(container.decode(stream))
                row.update(ok=True, width=frame.width, height=frame.height,
                           first_frame_pts=frame.pts, container_rate=str(stream.average_rate))
        except Exception as exc:
            row.update(ok=False, error=str(exc))
            report['ok'] = False
        report['videos'].append(row)
try:
    with (args.data / 'vial_fill/labels/video5_vial_recipes.csv').open(newline='', encoding='utf-8-sig') as handle:
        labels = list(csv.DictReader(handle))
    report['run5_label_rows'] = len(labels)
    report['run5_recipe_count'] = len({row['recipe_id'] for row in labels})
    report['ok'] = report['ok'] and len(labels) == 30
except Exception as exc:
    report['label_error'] = str(exc)
    report['ok'] = False
encoded = json.dumps(report, indent=2)
print(encoded)
if args.out:
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(encoded + '\n', encoding='utf-8')
raise SystemExit(0 if report['ok'] else 1)
