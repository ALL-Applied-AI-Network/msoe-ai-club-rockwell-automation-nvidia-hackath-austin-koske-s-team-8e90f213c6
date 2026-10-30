import argparse
import os
from pathlib import Path

from .evidence import Evidence
from .store import OperatorStore
from .server import OperatorServer


def main():
    tools = Path(os.environ['AI_CLUB_HOME']) if os.environ.get('AI_CLUB_HOME') else Path.home()/'ai-club-tools'
    parser = argparse.ArgumentParser(description='Operator-only recorded-data workspace; no line I/O or model inference.')
    parser.add_argument('--evidence', type=Path, default=tools/'results/pipeline-acceleration-20261020/full-run4-v1')
    parser.add_argument('--video', type=Path, default=Path('datasets/dataset2/csi-vial-fill-dataset/vial_fill/videos/vial_fill_video_4.mp4'))
    parser.add_argument('--out', type=Path, default=tools/'results/operator-workspace-run4')
    parser.add_argument('--port', type=int, default=8766)
    args = parser.parse_args()
    if not 0 <= args.port <= 65535:
        parser.error('--port must be between 0 and 65535')
    evidence = Evidence(args.evidence, args.video)
    store = OperatorStore(args.out, evidence)
    server = OperatorServer(('127.0.0.1', args.port), store)
    print(f'Operator workspace: http://127.0.0.1:{server.server_port}', flush=True)
    print(f'Append-only practice journal: {store.path}', flush=True)
    print('Recorded demonstration. No sensor, actuator or industrial network connections. Ctrl+C to stop.', flush=True)
    try:
        server.serve_forever(poll_interval=.25)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
