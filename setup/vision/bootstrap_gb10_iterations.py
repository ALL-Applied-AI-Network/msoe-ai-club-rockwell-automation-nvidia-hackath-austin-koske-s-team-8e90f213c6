"""Verify a completed transfer before extracting and detaching the GB10 runner."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile
import time


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main(root):
    if (root/'launch.json').exists():raise ValueError('Inspect previous launch rather than duplicate it')
    transfer=json.loads((root/'transfer.json').read_text())
    archive=root/'input.tar.gz'
    if sha(archive)!=transfer['archive_sha256'] or archive.stat().st_size!=transfer['bytes']:
        raise ValueError('Archive transfer is incomplete or differs')
    bundle=root/'bundle';bundle.mkdir(exist_ok=False)
    with tarfile.open(archive,'r:gz') as tar:
        for member in tar.getmembers():
            p=Path(member.name)
            if p.is_absolute() or '..' in p.parts or not member.isfile():raise ValueError('Unsafe archive entry')
        tar.extractall(bundle,filter='data')
    manifest=json.loads((bundle/'manifest.json').read_text())
    for name,expected in manifest.items():
        if sha(bundle/name)!=expected:raise ValueError('Extracted file differs: '+name)
    python=Path.home()/'ai-club-tools/experiments/gb10-station-benchmark-20261022/vision-env/bin/python'
    command=[str(python),'-u',str(bundle/'repo/setup/vision/run_synthetic_iterations.py'),'remote','--out',str(root)]
    with (root/'job.log').open('w') as log:
        child=subprocess.Popen(command,cwd=root,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    metadata=dict(pid=child.pid,verified_files=len(manifest),archive_sha256=sha(archive),started=time.time(),command=command,
        gpu_location='GB10',local_gpu_used=False)
    (root/'launch.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps(metadata),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path)
    main(p.parse_args().root)
