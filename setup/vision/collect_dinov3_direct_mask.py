"""Validate completed GB10 outputs and create a hash-verifiable result bundle."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())


def main(root):
    archive=root/'completed-artifacts.tar.gz'
    if archive.exists():raise ValueError('Preserve existing collection')
    run=root/'run';evaluation=root/'comparison-v1'
    assert (run/'completed.json').exists() and (evaluation/'completed.json').exists()
    assert not (run/'failed.json').exists() and not (evaluation/'failed.json').exists()
    train=read(run/'experiment-plan.json');frozen=read(run/'fitted-v1/all-models-frozen.json')
    assert sha(run/'experiment-plan.json')==frozen['plan_sha256']
    assert sha(run/'fitted-v1/all-models-frozen.json')==read(run/'completed.json')['model_metadata_sha256']
    for name,digest in train['helpers'].items():assert sha(run/name)==digest
    for name,digest in train['backbone_files'].items():assert sha(run/'pretrained'/name)==digest
    for mode,meta in frozen['models'].items():assert sha(run/'fitted-v1'/mode/'best.pt')==meta['checkpoint_sha256']
    for name,digest in read(evaluation/'completed.json')['artifacts'].items():assert sha(evaluation/name)==digest
    items=['comparison-v1','run/fitted-v1','run/experiment-plan.json','run/completed.json',
        'run/cpu-smoke.json','launch.json','comparison-launch.json','train.log','comparison.log']
    items.extend('run/'+name for name in train['helpers'])
    with tarfile.open(archive,'x:gz',compresslevel=1) as tar:
        for name in items:tar.add(root/name,arcname=name)
    result=dict(sha256=sha(archive),bytes=archive.stat().st_size)
    (root/'completed-artifacts.sha256.json').write_text(json.dumps(result,indent=2)+'\n');print(result)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);main(p.parse_args().root.resolve())
