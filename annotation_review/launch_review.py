"""Copy this launcher into a packaged batch root; Python 3.11+ only."""
import argparse
from pathlib import Path
import subprocess
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reviewer',choices=('Aleyna','Zoe','Reagan'),required=True)
    p.add_argument('--batch',type=Path,default=Path(__file__).resolve().parent)
    p.add_argument('--dataset',type=Path)
    p.add_argument('--workspace',type=Path)
    p.add_argument('--port',type=int,default=8775)
    args=p.parse_args();root=args.batch.resolve()
    command=[sys.executable,'-m','annotation_review','serve','--dataset',str((args.dataset or root/'data').resolve()),
        '--prelabels',str(root/'prelabels'),'--cases',str(root/'cases.json'),
        '--workspace',str((args.workspace or root.parent/('review-work-'+args.reviewer)).resolve()),'--port',str(args.port)]
    print('Choose '+args.reviewer+' in the editor. Your workspace is separate from the source package.',flush=True)
    raise SystemExit(subprocess.call(command,cwd=root))


if __name__=='__main__':main()
