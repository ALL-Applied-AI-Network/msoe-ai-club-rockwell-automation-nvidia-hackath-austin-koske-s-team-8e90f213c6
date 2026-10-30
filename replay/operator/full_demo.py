"""Opt-in two-station evidence replay and demonstration response API. No inference."""
import argparse
from copy import deepcopy
import hashlib
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
import secrets
from threading import Lock
from urllib.parse import urlsplit

from .rehearsal import DemonstrationJournal,verify_journal

STATIC=Path(__file__).parent/'static'


def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


class DemoStore:
    def __init__(self,bundle,workspace):
        self.bundle=bundle.resolve();self.workspace=workspace.resolve();self.lock=Lock()
        manifest=read(bundle/'manifest.json')
        for name,digest in manifest['files'].items():
            path=(bundle/name).resolve()
            if not path.is_relative_to(self.bundle) or sha(path)!=digest:raise ValueError('Demo input differs or escapes bundle')
        self.asset_hashes=manifest['files'];self.data=read(bundle/'data.json');self.known={r['id']:r for r in self.data['fill']+self.data['cap']}
        self.state=dict(revision=0,reviews=[],context=None,demonstration=True,quality_decision=None,hardware_commands=0)
        self.commands={};workspace.mkdir(parents=True,exist_ok=True);identity=workspace/'identity.json'
        if identity.exists():
            if read(identity)['bundle_sha256']!=sha(bundle/'manifest.json'):raise ValueError('Different source bundle; preserve workspace and choose a new one')
            events=verify_journal(workspace/'demonstration-events.jsonl');self.journal=DemonstrationJournal.__new__(DemonstrationJournal)
            self.journal.path=workspace/'demonstration-events.jsonl';self.journal.seq=len(events);self.journal.previous=events[-1]['event_hash'] if events else None
            for event in events:self.apply(event['payload'])
        else:
            if any(workspace.iterdir()):raise ValueError('Nonempty unrecognized workspace; choose a fresh directory')
            identity.write_text(json.dumps(dict(bundle_sha256=sha(bundle/'manifest.json'),demonstration=True)))
            self.journal=DemonstrationJournal(workspace/'demonstration-events.jsonl')

    def snapshot(self):
        with self.lock:return deepcopy(self.state)

    def export(self):
        with self.lock:return dict(schema='full-app-demo-export/v1',state=deepcopy(self.state),
            events=verify_journal(self.workspace/'demonstration-events.jsonl'),sources=self.data['sources'],
            demonstration=True,eligible_for_team_review=False,product_release_authorized=False,hardware_commands=0)

    def apply(self,payload):
        if payload['kind']=='context':self.state['context']=payload['value']
        elif payload['kind']=='request':self.state['reviews'].append(payload['review'])
        else:
            review=next(r for r in self.state['reviews'] if r['id']==payload['review_id'])
            review['status']={'acknowledge':'acknowledged','respond':'responded','close':'closed'}[payload['kind']]
            review['history'].append({k:payload[k] for k in ('kind','actor','note')})
        self.state['revision']+=1;self.commands[payload['command_id']]=True

    def command(self,value):
        with self.lock:
            command_id=value.get('command_id')
            if not isinstance(command_id,str) or not 8<=len(command_id)<=120:raise ValueError('Command identity required')
            if command_id in self.commands:return deepcopy(self.state)
            if value.get('revision')!=self.state['revision']:raise ValueError('State changed; refresh and retry')
            actor=str(value.get('actor','')).strip();note=str(value.get('note','')).strip();kind=value.get('kind')
            if not actor or len(actor)>80 or not note or len(note)>1000:raise ValueError('Name and explanation required')
            payload=dict(kind=kind,command_id=command_id,actor=actor,note=note)
            if kind=='context':payload['value']=dict(recipe=str(value.get('recipe','')).strip()[:100],actor=actor,note=note,origin='demonstration_operator_entry',reference_qualified=False)
            elif kind=='request':
                row=self.known.get(value.get('frame_id'))
                if row is None:raise ValueError('Unknown evidence frame')
                slot=value.get('slot')
                if row['station']=='final_fixture' and slot not in {r['id'] for r in row['slots']}:raise ValueError('Unknown target')
                if row['station']=='vial_fill' and slot is not None:raise ValueError('Filling target cannot claim a fixture slot')
                mode=value.get('mode','sam')
                if mode not in ('sam','direct'):raise ValueError('Unknown model branch')
                review=dict(id='review-'+str(self.state['revision']+1),frame_id=row['id'],station=row['station'],domain=row['domain'],slot=slot,mode=mode,
                    source_identity=row.get('source_identity',dict(sha256=row.get('source_sha256'))),status='requested',
                    actor=actor,note=note,quality_decision=None,body_condition='unassessed',product_release_authorized=False,
                    history=[dict(kind='request',actor=actor,note=note)])
                payload['review']=review
            elif kind in ('acknowledge','respond','close'):
                review=next((r for r in self.state['reviews'] if r['id']==value.get('review_id')),None)
                if review is None:raise ValueError('Unknown review')
                expected={'acknowledge':'requested','respond':'acknowledged','close':'responded'}[kind]
                if review['status']!=expected:raise ValueError('Acknowledge, record response, then close follow-up')
                payload['review_id']=review['id']
            else:raise ValueError('Unknown command')
            self.journal.append('demo_'+kind,payload);self.apply(payload)
            return deepcopy(self.state)


class Server(ThreadingHTTPServer):
    daemon_threads=True
    def __init__(self,address,store):self.store=store;self.csrf=secrets.token_urlsafe(32);super().__init__(address,Handler)


class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args):pass
    def valid_host(self):return self.headers.get('Host') in (f'127.0.0.1:{self.server.server_port}',f'localhost:{self.server.server_port}')
    def send(self,body,mime='application/json',status=200):
        if not isinstance(body,bytes):body=json.dumps(body,allow_nan=False).encode()
        self.send_response(status);self.send_header('Content-Type',mime);self.send_header('Content-Length',str(len(body)))
        self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff')
        self.send_header('Content-Security-Policy',"default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self'; connect-src 'self'; frame-ancestors 'none'; object-src 'none'; base-uri 'none'")
        self.end_headers()
        try:self.wfile.write(body)
        except (BrokenPipeError,ConnectionResetError):pass
    def do_GET(self):
        if not self.valid_host():return self.send(dict(error='Loopback host required'),status=403)
        path=urlsplit(self.path).path
        if path=='/api/data':return self.send(dict(data=self.server.store.data,csrf=self.server.csrf,state=self.server.store.snapshot()))
        if path=='/api/state':return self.send(self.server.store.snapshot())
        if path=='/api/export':return self.send(self.server.store.export())
        files={'/':STATIC/'full_demo.html','/full_demo.js':STATIC/'full_demo.js','/full_demo.css':STATIC/'full_demo.css'}
        if path.startswith('/assets/'):
            name=path.lstrip('/');manifest=self.server.store.asset_hashes
            if name in manifest:
                file=self.server.store.bundle/name
                if sha(file)!=manifest[name]:return self.send(dict(error='Pinned evidence changed'),status=412)
                files[path]=file
        file=files.get(path)
        if file is None:return self.send(dict(error='Not found'),status=404)
        self.send(file.read_bytes(),mimetypes.guess_type(file.name)[0] or 'application/octet-stream')
    def do_POST(self):
        origin=self.headers.get('Origin');allowed=(f'http://127.0.0.1:{self.server.server_port}',f'http://localhost:{self.server.server_port}')
        if not self.valid_host() or origin not in (None,*allowed) or self.headers.get('X-Demo-Token')!=self.server.csrf:
            return self.send(dict(error='Same-origin session required'),status=403)
        if urlsplit(self.path).path!='/api/command':return self.send(dict(error='Not found'),status=404)
        try:
            length=int(self.headers.get('Content-Length','0'))
            if not 0<length<=10000:raise ValueError('Command too large or empty')
            if self.headers.get_content_type()!='application/json':raise ValueError('JSON required')
            state=self.server.store.command(json.loads(self.rfile.read(length)));self.send(state)
        except (ValueError,TypeError,KeyError) as error:self.send(dict(error=str(error)),status=400)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--bundle',type=Path,required=True);parser.add_argument('--out',type=Path,required=True);parser.add_argument('--port',type=int,default=8870)
    args=parser.parse_args();store=DemoStore(args.bundle,args.out);server=Server(('127.0.0.1',args.port),store)
    print(f'Full app evidence demo: http://127.0.0.1:{server.server_port}',flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()


if __name__=='__main__':main()
