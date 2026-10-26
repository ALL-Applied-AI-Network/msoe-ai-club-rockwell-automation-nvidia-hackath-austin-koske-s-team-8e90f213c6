"""Wait for existing GB10 jobs, verify completion artifacts and bundle reports.

Does not launch training/evaluation jobs. Model checkpoints stay on GB10;
completed report manifests retain their hashes and source locations.
"""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile
import time


def sha(p):
    digest=hashlib.sha256()
    with Path(p).open('rb') as source:
        for block in iter(lambda:source.read(8*1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def read(p): return json.loads(Path(p).read_text())
def write(p,v): Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')


def main(a):
    a.out.mkdir(parents=True,exist_ok=False)
    paths={'long-original':a.v2/'long-original-v1','long-paired':a.v2/'long-paired-v1',
        'duration-controls':a.v2/'duration-controls-v1','synthetic-stability':a.v1/'mask-stability-v1',
        'real-stability':a.v1/'real-mask-stability-v1','pipeline-benchmark':a.v1/'temporal-pipeline-benchmark-v1'}
    write(a.out/'plan.json',dict(roots={k:str(v) for k,v in paths.items()},helper_sha256=sha(__file__),
        waits_only=True,resubmits=False,weights_copied=False))
    deadline=time.monotonic()+a.timeout
    while not all((p/'completed.json').exists() for p in paths.values()):
        pending=[name for name,p in paths.items() if not (p/'completed.json').exists()]
        write(a.out/'waiting.json',dict(pending=pending,checked_unix=time.time()))
        if (a.v2/'failed.json').exists():raise RuntimeError('Duration suite failed; inspect preserved logs')
        if time.monotonic()>deadline:raise TimeoutError('Existing jobs did not all complete; no replacements submitted')
        time.sleep(5)
    assert (a.v2/'completed-long.json').exists()
    files={};remote_weights={}
    for name,root in paths.items():
        complete=read(root/'completed.json')
        for rel,digest in complete['artifacts'].items():
            if sha(root/rel)!=digest:raise ValueError('Completed bytes differ: '+str(root/rel))
            if rel.endswith('.pt'):remote_weights[str(root/rel)]=digest
        for p in root.iterdir():
            if p.is_file() and p.suffix in ['.json','.jsonl','.png','.csv']:
                files[name+'/'+p.name]=p
    for p in [a.v2/'suite-plan.json',a.v2/'completed-long.json',
              a.v1/'probe_pair_mask_stability.py',a.v1/'probe_real_temporal_stability.py',
              a.v1/'benchmark_temporal_pipeline.py',a.v2/'evaluate_synthetic_duration_controls.py']:
        files['executed-helpers/'+p.name]=p
    manifest=dict(files={name:sha(p) for name,p in files.items()},remote_weights=remote_weights,
        source_roots={k:str(v) for k,v in paths.items()},run5_used=False,approved_labels=False,
        local_gpu_used=False,app_changed=False,all_artifact_hashes_verified=True)
    write(a.out/'manifest.json',manifest);files['manifest.json']=a.out/'manifest.json'
    archive=a.out/'completed-reports.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        for name,p in sorted(files.items()):tar.add(p,arcname=name)
    write(a.out/'completed.json',dict(archive_sha256=sha(archive),archive_bytes=archive.stat().st_size,
        files=len(files),finished_unix=time.time(),source_roots={k:str(v) for k,v in paths.items()}))
    print(json.dumps(read(a.out/'completed.json')),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['v1','v2','out']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--timeout',type=int,default=5400)
    a=p.parse_args()
    try:main(a)
    except BaseException as e:
        if a.out.exists():write(a.out/'failed.json',dict(type=type(e).__name__,error=str(e)))
        raise
