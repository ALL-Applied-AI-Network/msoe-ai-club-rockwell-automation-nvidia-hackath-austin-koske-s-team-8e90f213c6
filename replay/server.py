"""Loopback-only app, range-capable recording delivery and shared replay rules."""
from concurrent.futures import ThreadPoolExecutor
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
import threading
from urllib.parse import parse_qs, urlsplit
import uuid

from .pipeline import analyze, export_csv, validate_references, write_json

STATIC = Path(__file__).parent / 'static'


def byte_range(header, size):
    if header is None:
        return 0, size - 1, False
    match = re.fullmatch(r'bytes=(\d*)-(\d*)', header.strip())
    if not match or not any(match.groups()):
        raise ValueError('Only a single byte range is supported')
    left, right = match.groups()
    if left:
        start, end = int(left), min(int(right), size - 1) if right else size - 1
    else:
        suffix = int(right)
        if suffix <= 0:
            raise ValueError('Invalid suffix range')
        start, end = max(0, size - suffix), size - 1
    if size <= 0 or not 0 <= start <= end < size:
        raise ValueError('Byte range is outside the file')
    return start, end, True


class Manager:
    def __init__(self, catalog, config, output, references_path=None, fill_source='image'):
        self.catalog, self.config = catalog, config
        self.fill_source = fill_source
        self.output = Path(output).expanduser().resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='recording-analysis')
        self.jobs, self.data, self.progress = {}, {}, {}
        self.reference_path = self.output / 'development-references.json'
        source = Path(references_path) if references_path else self.reference_path
        if references_path and not source.is_file():
            raise FileNotFoundError(f'Reference bank does not exist: {source}')
        self.references = validate_references(json.loads(source.read_text(encoding='utf-8-sig'))['references']) if source.is_file() else []

    def ensure(self, run_id):
        if run_id not in self.catalog.runs:
            raise ValueError('Recording does not exist')
        with self.lock:
            if run_id not in self.jobs:
                self.jobs[run_id] = self.pool.submit(self._prepare, run_id)
            job = self.jobs[run_id]
            if not job.done():
                return dict(status='preparing', progress=self.progress.get(run_id, 0))
            error = job.exception()
            if error:
                return dict(status='error', error=str(error))
            return dict(status='ready', progress=1.)

    def _prepare(self, run_id):
        def progress(value):
            with self.lock:
                self.progress[run_id] = value
        data = analyze(self.catalog.runs[run_id], self.config, self.output / 'cache', progress, self.fill_source)
        with self.lock:
            self.data[run_id] = data

    def publish(self, run_id, vial_id, frame):
        with self.lock:
            if run_id not in self.data:
                raise ValueError('Wait for recording preparation')
            data = self.data[run_id]
            if not 0 <= frame < data.payload['manifest']['frame_count']:
                raise ValueError('Reference availability lies outside the recording')
            item = next((item for item in data.observations if item['vial_id'] == vial_id), None)
            if item is None or item['available_frame'] > frame or not item['measurement_valid']:
                raise ValueError('Choose an already available, QA-valid completed observation')
            if item['recipe_id'] is None:
                raise ValueError('A recipe context source is required before publishing a recipe reference')
            reference = dict(reference_id='dev-' + uuid.uuid4().hex[:12], recipe_id=item['recipe_id'],
                             lab=item['lab'], rgb=item['rgb'], scope='same_run_development',
                             measurement_config_id=item['measurement_config_id'], source_run_id=run_id,
                             source_video_sha256=item['source_video_sha256'],
                             source_annotation_sha256=item['source_annotation_sha256'],
                             source_event_id=item['event_id'], available_frame=frame,
                             source_description='Manually selected observed example; not certified clean product')
            validate_references([reference])
            self.references.append(reference)
            write_json(self.reference_path, dict(schema_version=1, references=self.references))
            return reference

    def close(self):
        self.pool.shutdown(wait=True)


class ReplayServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, manager):
        self.manager = manager
        super().__init__(address, Handler)


class Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def log_message(self, *_):
        pass

    @property
    def manager(self):
        return self.server.manager

    def respond(self, value, status=200, mime='application/json; charset=utf-8', filename=None):
        body = json.dumps(value, allow_nan=False).encode() if mime.startswith('application/json') else value.encode() if isinstance(value, str) else value
        self.send_response(status)
        self.send_header('Content-Type', mime)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Referrer-Policy', 'no-referrer')
        self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; media-src 'self'; connect-src 'self'; frame-ancestors 'none'")
        if filename:
            self.send_header('Content-Disposition', f'attachment; filename="{filename}"')
        self.end_headers()
        if self.command != 'HEAD':
            self.wfile.write(body)

    def serve_media(self, path):
        size = path.stat().st_size
        try:
            start, end, partial = byte_range(self.headers.get('Range'), size)
        except ValueError:
            self.send_response(416)
            self.send_header('Content-Range', f'bytes */{size}')
            self.send_header('Content-Length', '0')
            self.end_headers()
            return
        self.send_response(206 if partial else 200)
        self.send_header('Content-Type', 'video/mp4')
        self.send_header('Accept-Ranges', 'bytes')
        self.send_header('Content-Length', str(end - start + 1))
        if partial:
            self.send_header('Content-Range', f'bytes {start}-{end}/{size}')
        self.end_headers()
        if self.command == 'HEAD':
            return
        with path.open('rb') as handle:
            handle.seek(start)
            remaining = end - start + 1
            while remaining:
                block = handle.read(min(256 * 1024, remaining))
                if not block:
                    break
                self.wfile.write(block)
                remaining -= len(block)

    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        try:
            parsed = urlsplit(self.path)
            path, query = parsed.path, parse_qs(parsed.query)
            assets = {'/': ('index.html', 'text/html; charset=utf-8'),
                      '/app.js': ('app.js', 'text/javascript; charset=utf-8'),
                      '/style.css': ('style.css', 'text/css; charset=utf-8')}
            if path in assets:
                name, mime = assets[path]
                return self.respond((STATIC / name).read_bytes(), mime=mime)
            if path == '/favicon.ico':
                return self.respond(b'', 204, mime='image/x-icon')
            if path == '/api/catalog':
                return self.respond(dict(runs=self.manager.catalog.public(), config=self.manager.config.to_dict(),
                                         config_id=self.manager.config.fingerprint,
                                         fill_source=self.manager.fill_source,
                                         output_directory=str(self.manager.output)))
            media = re.fullmatch(r'/media/(\d+)/(vial_fill|tank_level|cap_seal)\.mp4', path)
            if media:
                run = self.manager.catalog.runs.get(int(media[1]), {})
                camera = run.get('cameras', {}).get(media[2])
                if camera:
                    return self.serve_media(Path(camera['path']))
            if path == '/api/references':
                with self.manager.lock:
                    return self.respond(dict(schema_version=1, references=self.manager.references), filename='references.json')
            if path in ('/api/state', '/api/status', '/api/events.jsonl', '/api/observations.csv', '/api/manifest', '/api/fills'):
                run_id = int(query.get('run', ['5'])[0])
                ready = self.manager.ensure(run_id)
                if ready['status'] != 'ready' or path == '/api/status':
                    return self.respond(ready, 202 if ready['status'] == 'preparing' else 500 if ready['status'] == 'error' else 200)
                with self.manager.lock:
                    data = self.manager.data[run_id]
                    frame = int(query.get('frame', [str(data.payload['manifest']['frame_count'] - 1)])[0])
                    if not 0 <= frame < data.payload['manifest']['frame_count']:
                        raise ValueError('Frame lies outside the recording')
                    references = list(self.manager.references)
                if path == '/api/state':
                    return self.respond(data.snapshot(frame, references))
                if path == '/api/manifest':
                    return self.respond(data.payload['manifest'])
                if path == '/api/fills':
                    # Full-recording index for explicit replay navigation, never decision input.
                    return self.respond(dict(purpose='replay_navigation_only', fills=[
                        {key: item[key] for key in ('vial_id', 'recipe_id', 'start_frame', 'end_frame', 'available_frame')}
                        for item in data.observations]))
                if path == '/api/events.jsonl':
                    body = ''.join(json.dumps(item, allow_nan=False) + '\n' for item in data.events(frame, references))
                    return self.respond(body, mime='application/x-ndjson', filename=f'run{run_id}-events-through-{frame}.jsonl')
                return self.respond(export_csv(data.completed(frame, references)), mime='text/csv; charset=utf-8', filename=f'run{run_id}-observations-through-{frame}.csv')
            return self.respond(dict(error='Not found'), 404)
        except (ValueError, KeyError, json.JSONDecodeError) as exc:
            self.respond(dict(error=str(exc)), 400)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as exc:
            self.respond(dict(error=str(exc)), 500)

    def do_POST(self):
        try:
            # Browser mutations must originate from this loopback app.
            origin = self.headers.get('Origin')
            if origin and origin != 'http://' + self.headers.get('Host', ''):
                return self.respond(dict(error='Origin mismatch'), 403)
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length <= 16384:
                return self.respond(dict(error='Expected a small JSON request'), 413)
            body = json.loads(self.rfile.read(length))
            if self.path != '/api/references':
                return self.respond(dict(error='Not found'), 404)
            reference = self.manager.publish(int(body['run_id']), int(body['vial_id']), int(body['frame_index']))
            return self.respond(reference, HTTPStatus.CREATED)
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            return self.respond(dict(error=str(exc)), 400)
        except (BrokenPipeError, ConnectionResetError):
            pass
