import argparse
import json
import os
from pathlib import Path

from .config import load_config
from .dataset import Catalog, DEFAULT_DATA
from .pipeline import analyze, export_csv, write_json


def main():
    parser = argparse.ArgumentParser(description='Replay the supplied recordings without modifying data or labels.')
    parser.add_argument('command', choices=['inspect', 'analyze', 'serve', 'validate-detection'], nargs='?', default='serve')
    parser.add_argument('--fill-source', choices=['image', 'annotations'], default='image', help='Image-only detection (default), or the supplied-annotation baseline')
    parser.add_argument('--data', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--labels-dir', type=Path, help='Reviewed overrides named run1.csv .. run5.csv')
    parser.add_argument('--clock-map-dir', type=Path, help='Optional unreviewed framemap_vN.npy estimates')
    parser.add_argument('--config', type=Path)
    parser.add_argument('--out', type=Path, default=Path(os.environ.get('AI_CLUB_HOME', str(Path.home() / 'ai-club-tools'))) / 'results/replay')
    parser.add_argument('--runs', default='5', help='Comma-separated IDs, or all')
    parser.add_argument('--references', type=Path)
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    config = load_config(args.config)
    catalog = Catalog(args.data, args.labels_dir, args.clock_map_dir,
                      include_annotations=args.command in ('inspect', 'validate-detection') or args.fill_source == 'annotations')
    if args.command == 'inspect':
        print(json.dumps(dict(runs=catalog.public(), config_id=config.fingerprint), indent=2))
        return
    run_ids = list(catalog.runs) if args.runs == 'all' else [int(item) for item in args.runs.split(',')]
    if any(run_id not in catalog.runs for run_id in run_ids):
        parser.error('Requested recording is missing')
    from .server import Manager, ReplayServer
    manager = Manager(catalog, config, args.out, args.references, args.fill_source)
    try:
        if args.command in ('analyze', 'validate-detection'):
            for run_id in run_ids:
                print(f'Analyzing run {run_id} ...', flush=True)
                source = 'image' if args.command == 'validate-detection' else args.fill_source
                data = analyze(catalog.runs[run_id], config, manager.output / 'cache', fill_source=source)
                output = manager.output / f'run{run_id}'
                output.mkdir(parents=True, exist_ok=True)
                last = data.payload['manifest']['frame_count'] - 1
                write_json(output / 'manifest.json', data.payload['manifest'])
                if args.command == 'validate-detection':
                    from .validation import evaluate
                    report = evaluate(data.payload['detections'], catalog.runs[run_id]['annotations'], data.payload['manifest'])
                    write_json(output / 'detection-validation.json', report)
                    print(json.dumps(report['summary']), flush=True)
                (output / 'features.jsonl').write_text(''.join(json.dumps(row, allow_nan=False) + '\n' for row in data.features), encoding='utf-8')
                (output / 'events.jsonl').write_text(''.join(json.dumps(row, allow_nan=False) + '\n' for row in data.events(last, manager.references)), encoding='utf-8')
                (output / 'observations.csv').write_text(export_csv(data.completed(last, manager.references)), encoding='utf-8', newline='')
                print(json.dumps(dict(run_id=run_id, samples=len(data.features), completed_observations=len(data.observations),
                                      qa_valid=sum(item['measurement_valid'] for item in data.observations),
                                      analysis_seconds=data.payload['manifest']['analysis_seconds'], output=str(output))), flush=True)
            return
        for run_id in run_ids:
            manager.ensure(run_id)
        server = ReplayServer(('127.0.0.1', args.port), manager)
        print(f'Replay app: http://127.0.0.1:{server.server_port}', flush=True)
        print(f'Results and development references: {manager.output}', flush=True)
        print('Ctrl+C to stop. Preparing the selected recording in the background.', flush=True)
        try:
            server.serve_forever(poll_interval=.25)
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()
    finally:
        manager.close()


if __name__ == '__main__':
    main()
