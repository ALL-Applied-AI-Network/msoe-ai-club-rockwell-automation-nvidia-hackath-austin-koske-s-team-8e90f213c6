"""Loopback operator UI, immutable evidence media and manual journal API."""
import csv
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
from pathlib import Path
import re
import secrets
from urllib.parse import parse_qs, urlsplit

from .store import Conflict

STATIC = Path(__file__).parent/'static'


def byte_range(header, size):
    if header is None:
        return 0, size-1, False
    match = re.fullmatch(r'bytes=(\d*)-(\d*)', header.strip())
    if not match or not any(match.groups()):
        raise ValueError('A single byte range is required')
    left, right = match.groups()
    if left:
        start, end = int(left), min(int(right), size-1) if right else size-1
    else:
        suffix = int(right)
        if suffix <= 0:
            raise ValueError('Invalid suffix')
        start, end = max(0, size-suffix), size-1
    if not 0 <= start <= end < size:
        raise ValueError('Range outside file')
    return start, end, True


class OperatorServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, store):
        self.store = store
        self.csrf = secrets.token_urlsafe(32)
        super().__init__(address, Handler)


class Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def log_message(self, *_):
        pass

    def handle(self):
        try:
            super().handle()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass # A browser may cancel a media request or close a keep-alive socket.

    @property
    def store(self):
        return self.server.store

    def valid_host(self):
        return self.headers.get('Host') in {f'127.0.0.1:{self.server.server_port}', f'localhost:{self.server.server_port}'}

    def respond(self, value, status=200, mime='application/json; charset=utf-8', filename=None):
        body = json.dumps(value, allow_nan=False).encode() if mime.startswith('application/json') else value.encode() if isinstance(value, str) else value
        self.send_response(status)
        self.send_header('Content-Type', mime)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Referrer-Policy', 'no-referrer')
        self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self'; media-src 'self'; connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'; form-action 'self'")
        if filename:
            self.send_header('Content-Disposition', f'attachment; filename="{filename}"')
        self.end_headers()
        if self.command != 'HEAD':
            self.wfile.write(body)

    def media(self):
        path = self.store.evidence.video; size = path.stat().st_size
        try:
            start, end, partial = byte_range(self.headers.get('Range'), size)
        except ValueError:
            self.send_response(416)
            self.send_header('Content-Range', f'bytes */{size}')
            self.send_header('Content-Length', '0')
            self.end_headers(); return
        self.send_response(206 if partial else 200)
        self.send_header('Content-Type', 'video/mp4')
        self.send_header('Accept-Ranges', 'bytes')
        self.send_header('Content-Length', str(end-start+1))
        if partial:
            self.send_header('Content-Range', f'bytes {start}-{end}/{size}')
        self.end_headers()
        if self.command == 'HEAD':
            return
        with path.open('rb') as f:
            f.seek(start); remaining = end-start+1
            while remaining:
                block = f.read(min(256*1024, remaining))
                if not block:
                    break
                self.wfile.write(block); remaining -= len(block)

    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        try:
            if not self.valid_host():
                return self.respond(dict(error='Use the loopback workspace URL'), 403)
            parsed = urlsplit(self.path); query = parse_qs(parsed.query); path = parsed.path
            assets = {'/': ('index.html', 'text/html; charset=utf-8'),
                      '/operator.js': ('operator.js', 'text/javascript; charset=utf-8'),
                      '/operator.css': ('operator.css', 'text/css; charset=utf-8')}
            if path in assets:
                name, mime = assets[path]
                return self.respond((STATIC/name).read_bytes(), mime=mime)
            if path == '/favicon.ico':
                return self.respond(b'', 204, 'image/x-icon')
            if path == '/video.mp4':
                return self.media()
            if path == '/api/session':
                return self.respond(dict(source=self.store.evidence.public, session_id=self.store.session_id,
                                         csrf_token=self.server.csrf, journal_path=str(self.store.path)))
            seconds = float(query.get('at_s', ['0'])[0])
            if path == '/api/state':
                return self.respond(self.store.snapshot(seconds, query.get('case_id', [None])[0]))
            if path == '/api/evidence':
                frame = int(query['frame'][0])
                if frame > self.store.evidence.at(seconds)['source_frame']:
                    raise ValueError('Evidence is ahead of the requested playhead')
                return self.respond(self.store.evidence.preview(frame, query.get('kind', ['overlay'])[0]), mime='image/png')
            if path in ('/api/export.json', '/api/export.csv'):
                with self.store.lock:
                    data = self.store.export(seconds)
                if path.endswith('.json'):
                    return self.respond(data, filename='operator-session.json')
                buffer = io.StringIO(newline='')
                fields = ['case_id','physical_vial_id','batch_id','recipe_id','identity_confirmed','start_frame',
                          'completed_frame','lifecycle','review_state','measurement_valid','assessment',
                          'diagnostic_status','delta_e2000','last_operator_action','reported_station_route',
                          'route_semantics','independent_quality_outcome','source_video_sha256','source_mode']
                writer = csv.DictWriter(buffer, fieldnames=fields); writer.writeheader()
                for case in data['cases']:
                    row = {key: case.get(key) for key in fields}
                    action = case['actions'][-1] if case['actions'] else {}
                    row.update(measurement_valid=case['endpoint']['measurement_valid'] if case['endpoint'] else None,
                               assessment=case['assessment']['status'], diagnostic_status=case['assessment']['diagnostic_status'],
                               delta_e2000=case['assessment']['delta_e2000'], last_operator_action=action.get('action'),
                               reported_station_route=action.get('reported_station_route'), route_semantics='unconfirmed',
                               independent_quality_outcome='not_verified', source_video_sha256=data['source_identity']['source_video_sha256'],
                               source_mode='recorded_demonstration')
                    writer.writerow({k: "'"+v if isinstance(v, str) and v.startswith(('=', '+', '-', '@')) else v for k,v in row.items()})
                return self.respond(buffer.getvalue(), mime='text/csv; charset=utf-8', filename='operator-vials.csv')
            return self.respond(dict(error='Not found'), 404)
        except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            self.respond(dict(error=str(exc)), 400)
        except (BrokenPipeError, ConnectionResetError):
            pass

        except Exception as exc:
            self.respond(dict(error=str(exc)), 500)

    def do_POST(self):
        try:
            if not self.valid_host() or self.headers.get('Origin') not in (None, 'http://'+self.headers.get('Host', '')):
                return self.respond(dict(error='Origin mismatch'), 403)
            if not secrets.compare_digest(self.headers.get('X-Operator-CSRF', ''), self.server.csrf):
                return self.respond(dict(error='Workspace token required'), 403)
            if self.path != '/api/command':
                return self.respond(dict(error='Not found'), 404)
            if self.headers.get('Content-Type', '').split(';')[0] != 'application/json':
                return self.respond(dict(error='JSON input required'), 415)
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length <= 16384:
                return self.respond(dict(error='Expected a small command'), 413)
            result = self.store.command(json.loads(self.rfile.read(length)))
            return self.respond(result, 200 if result['duplicate'] else 201)
        except Conflict as exc:
            self.respond(dict(error=str(exc)), 409)
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            self.respond(dict(error=str(exc)), 400)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as exc:
            self.respond(dict(error=str(exc)), 500)

