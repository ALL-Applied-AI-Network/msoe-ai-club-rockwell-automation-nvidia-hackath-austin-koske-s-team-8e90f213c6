import argparse
import json
from pathlib import Path
from .core import Catalog, Reviews
from .server import create_server


def main():
    p=argparse.ArgumentParser(description='Independent review of saved prelabels; no API key, GPU or Node needed')
    p.add_argument('command',choices=('serve','export','import'))
    p.add_argument('--dataset',type=Path,required=True,help='Extracted csi-vial-fill-dataset root')
    p.add_argument('--prelabels',type=Path,required=True,help='Existing bulk output OR extracted labels-only archive')
    p.add_argument('--workspace',type=Path,required=True,help='Separate writable review directory, not raw data')
    p.add_argument('--cases',type=Path,help='Optional prioritized review cases manifest')
    p.add_argument('--mapping',type=Path,help='Explicit reviewed/proposed still-to-video associations CSV')
    p.add_argument('--port',type=int,default=8775)
    p.add_argument('--demo',action='store_true',help='All revisions are demonstration-only and ineligible as reviewed labels')
    p.add_argument('--file',type=Path,help='Correction JSON input/output')
    args=p.parse_args()
    for source in (args.dataset,args.prelabels):
        if args.workspace.resolve().is_relative_to(source.resolve()):
            p.error('Keep review workspace outside the dataset and original prelabels')
    if args.command=='serve':
        server=create_server(args.dataset,args.prelabels,args.workspace,args.cases,args.port,args.demo,args.mapping)
        print(f'Annotation review: http://127.0.0.1:{server.server_port} (Ctrl+C to stop)',flush=True)
        try:server.serve_forever()
        except KeyboardInterrupt:pass
        finally:server.server_close()
    else:
        if not args.file:p.error('--file is required')
        catalog=Catalog(args.dataset,args.prelabels,args.cases)
        if args.mapping:
            from .alignment import load_mappings
            load_mappings(catalog,args.mapping)
        reviews=Reviews(args.workspace/'reviews.sqlite',catalog,args.demo)
        if args.command=='export':
            args.file.parent.mkdir(parents=True,exist_ok=True)
            args.file.write_text(json.dumps(reviews.export(),indent=2)+'\n',encoding='utf-8')
        else:print('Imported revisions:',reviews.import_bundle(json.loads(args.file.read_text(encoding='utf-8'))))


if __name__=='__main__':main()
