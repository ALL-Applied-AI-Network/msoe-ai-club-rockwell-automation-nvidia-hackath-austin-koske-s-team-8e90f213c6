"""Loopback-only review API. Dataset/prelabel files are never writable endpoints."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from urllib.parse import urlparse, parse_qs

from .core import Catalog, Reviews


def create_server(dataset,prelabels,workspace,cases=None,port=8775,demo=False):
    catalog=Catalog(dataset,prelabels,cases)
    reviews=Reviews(Path(workspace)/'reviews.sqlite',catalog,demo)
    static=Path(__file__).parent/'static'

    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):
            pass

        def send(self,data,status=200,kind='application/json'):
            body=json.dumps(data,allow_nan=False).encode() if kind=='application/json' else data
            self.send_response(status)
            self.send_header('Content-Type',kind)
            self.send_header('Content-Length',str(len(body)))
            self.send_header('Cache-Control','no-store')
            self.send_header('X-Content-Type-Options','nosniff')
            self.end_headers();self.wfile.write(body)

        def do_GET(self):
            try:
                url=urlparse(self.path);query=parse_qs(url.query)
                if url.path in ('/','/app.js','/style.css'):
                    path=static/({'/':'index.html','/app.js':'app.js','/style.css':'style.css'}[url.path])
                    self.send(path.read_bytes(),kind={'/':'text/html; charset=utf-8','/app.js':'application/javascript','/style.css':'text/css'}[url.path])
                elif url.path=='/api/config':
                    self.send(dict(classes=catalog.classes,demo=demo,reviewers=['Aleyna','Zoe','Reagan'],
                        cases=catalog.case_manifest,coordinate_convention='normalized_full_image_xyxy',
                        note='Reviewer identity is self-reported; run locally per reviewer. Confidence never approves a candidate.'))
                elif url.path=='/api/catalog':
                    current={r['sample_id'] for r in reviews.history()}
                    self.send([dict(id=s['id'],camera=s['source']['camera'],run=s['source']['run'],kind=s['source']['kind'],
                        reviewed=s['id'] in current,case=s.get('case')) for s in catalog.samples.values()])
                elif url.path in ('/api/sample','/api/image'):
                    sample=catalog.samples[query['id'][0]]
                    if url.path=='/api/sample':
                        self.send(dict(sample=sample,current=reviews.current(sample['id']),history=reviews.history(sample['id'])))
                    else:
                        self.send(catalog.path(sample).read_bytes(),kind='image/jpeg')
                elif url.path=='/api/export':
                    self.send(reviews.export())
                else:
                    self.send({'error':'Not found'},404)
            except (ValueError,KeyError,FileNotFoundError) as exc:
                self.send({'error':str(exc)},400)

        def do_POST(self):
            try:
                origin=self.headers.get('Origin')
                if origin and origin!=f'http://{self.headers.get("Host")}':
                    raise ValueError('Cross-origin writes are disabled')
                if self.headers.get('Content-Type','').split(';')[0]!='application/json':
                    raise ValueError('JSON is required')
                length=int(self.headers.get('Content-Length','0'))
                if not 0<length<=30_000_000:
                    raise ValueError('Invalid request size')
                data=json.loads(self.rfile.read(length))
                if self.path=='/api/review':
                    self.send(reviews.save(data))
                elif self.path=='/api/import':
                    self.send({'imported_revisions':reviews.import_bundle(data)})
                else:
                    self.send({'error':'Not found'},404)
            except (ValueError,KeyError,TypeError,FileNotFoundError) as exc:
                self.send({'error':str(exc)},400)

    server=ThreadingHTTPServer(('127.0.0.1',port),Handler)
    server.catalog,server.reviews=catalog,reviews
    return server
