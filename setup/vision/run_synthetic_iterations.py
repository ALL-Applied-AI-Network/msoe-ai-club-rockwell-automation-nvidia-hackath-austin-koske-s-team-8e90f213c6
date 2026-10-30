"""Portable GB10 experiment input and detached sequential synthetic-only fits."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def package(a):
    a.out.mkdir(parents=True,exist_ok=False)
    bundle=a.out/'bundle';bundle.mkdir();(bundle/'specs').mkdir()
    sources={'original':a.tools/'results/synthetic-liquid-20261022/fasterrcnn-v1',
             'paired':a.tools/'results/targeted-synthetic-20261022/fasterrcnn-paired-v1'}
    for name,root in sources.items():
        plan=read(root/'experiment-plan.json')
        assert not plan['training_real_runs'] and not plan['selection_real_runs'] and not plan['run5_used']
        converted=copy.deepcopy(plan)
        for row in converted['samples']:
            assert row['origin']=='synthetic' and row['source_run'] is None
            kind='paired' if row['source_id'].startswith('pair-') else 'original'
            for key in ['image','label']:
                source=root/row[key];assert sha(source)==row[key+'_sha256']
                rel=f'data/{kind}/{key}s/{source.name}';target=bundle/rel
                target.parent.mkdir(parents=True,exist_ok=True)
                if target.exists():assert sha(target)==row[key+'_sha256']
                else:shutil.copy2(source,target)
                row[key]=rel
        converted['parent_plan_sha256']=sha(root/'experiment-plan.json')
        write(bundle/'specs'/(name+'.json'),converted)
    renders=a.tools/'results/targeted-synthetic-20261022/paired-render-v1'
    native=[json.loads(l) for l in (renders/'manifest.jsonl').read_text().splitlines()]
    (bundle/'data/paired/masks').mkdir()
    for row in native:
        shutil.copy2(renders/'masks'/(row['id']+'.png'),bundle/'data/paired/masks'/(row['id']+'.png'))
    write(bundle/'specs/paired-native.json',native)
    real=a.tools/'results/real-synthetic-detector-20261019/data/real'
    prov=read(real/'provenance.json');rows=[r for r in prov['samples'] if r['split']=='val']
    assert len(rows)==157 and all(r['source_run']==4 and not r.get('parent_id') for r in rows)
    target_real=bundle/'tools/results/real-synthetic-detector-20261019/data/real';target_real.mkdir(parents=True)
    for row in rows:
        for key,hashkey in [('image','image_sha256'),('labels','actual_label_sha256')]:
            path=real/row[key];assert sha(path)==row[hashkey]
            target=target_real/row[key];target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
    write(target_real/'provenance.json',dict(samples=rows,source_provenance_sha256=sha(real/'provenance.json'),
        purpose='Post-freeze evaluation only; no training images; Run4 only.'))
    for source in sorted(HERE.glob('*.py')):
        target=bundle/'repo/setup/vision'/source.name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
    for source in sorted((REPO/'replay').glob('*.py')):
        target=bundle/'repo/replay'/source.name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
    review=a.tools/'results/pipeline-acceleration-20261020/early-fill-review-v1'
    for name in ['onset-review.json','request.json']:
        shutil.copy2(review/name,bundle/'specs'/name)
    video=list((REPO/'datasets/dataset2/csi-vial-fill-dataset').rglob('vial_fill_video_4.mp4'))
    assert len(video)==1
    target=bundle/'evaluation/vial_fill_video_4.mp4';target.parent.mkdir();shutil.copy2(video[0],target)
    write(bundle/'specs/real-video.json',dict(sha256=sha(video[0]),file='evaluation/vial_fill_video_4.mp4',run_id=4,
        onset_status='Unreviewed predeclared navigation; no approved empty reference or timing labels.'))
    manifest={str(p.relative_to(bundle)).replace('\\','/'):sha(p) for p in sorted(bundle.rglob('*')) if p.is_file()}
    write(bundle/'manifest.json',manifest)
    archive=a.out/'input.tar.gz'
    with tarfile.open(archive,'w:gz',compresslevel=1) as tar:
        for p in sorted(bundle.rglob('*')):
            if p.is_file():tar.add(p,arcname=str(p.relative_to(bundle)))
    write(a.out/'transfer.json',dict(archive_sha256=sha(archive),files=len(manifest),bytes=archive.stat().st_size,
        training_real_runs=[],run5_used=False,gpu_work='GB10 only'))
    print(json.dumps(read(a.out/'transfer.json')),flush=True)


def remote(a):
    import fcntl
    root=a.out;bundle=a.bundle if a.bundle else root/'bundle'
    root.mkdir(parents=True,exist_ok=True)
    lock=(Path.home()/'ai-club-tools/spark-gpu.lock').open('a+')
    if (root/'queued.json').exists():raise ValueError('Inspect existing queued experiment')
    write(root/'queued.json',dict(pid=os.getpid(),queued=time.time(),gpu='GB10'))
    try:fcntl.flock(lock,fcntl.LOCK_EX if a.wait_for_gpu else fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise SystemExit('Shared GPU is busy; no experiments launched')
    if (root/'running.json').exists():raise ValueError('Inspect prior run; do not duplicate')
    for rel,expected in read(bundle/'manifest.json').items():
        if sha(bundle/rel)!=expected:raise ValueError('Transferred bytes differ: '+rel)
    write(root/'running.json',dict(pid=os.getpid(),started=time.time(),gpu_lock=True,host=os.uname().nodename))
    write(root/'suite-plan.json',dict(arms=['original','paired'],epochs=15,seed=19,
        selection='Synthetic AP50-95 at epochs3/6/9/12/15; earliest tie.',retain_epoch3_control=True,
        checkpoint_real_evaluation='After each selected checkpoint freezes; no real feedback changes later arm plans.',
        bundle_manifest_sha256=sha(bundle/'manifest.json'),runner_sha256=sha(__file__),
        training_real_runs=[],selection_real_runs=[],run5_used=False,local_gpu_used=False))
    cached=Path.home()/'ai-club-tools/experiments/gb10-station-benchmark-20261022/bundle-v1/tools/results/early-liquid-20261020/pretrained/fasterrcnn-coco.pth'
    env=dict(os.environ,PYTHONPATH=str(bundle/'repo/setup/vision')+':'+str(bundle/'repo'),PYTHONUNBUFFERED='1')
    if a.helper_overlay:
        for name,expected in read(a.helper_overlay/'manifest.json').items():
            if sha(a.helper_overlay/name)!=expected:raise ValueError('Dependency repair overlay changed')
    try:
        for arm in ['original','paired']:
            out=root/('long-'+arm+'-v1');out.mkdir()
            (out/'helpers').mkdir();(out/'pretrained').mkdir()
            plan=read(bundle/'specs'/(arm+'.json'))
            for row in plan['samples']:
                for key in ['image','label']:row[key]=str(bundle/row[key])
            assert sha(cached)==plan['pretrained_sha256']
            shutil.copy2(cached,out/'pretrained/fasterrcnn-coco.pth')
            helpers={}
            for p in (bundle/'repo/setup/vision').glob('*.py'):
                shutil.copy2(p,out/'helpers'/p.name);helpers[p.name]=sha(p)
            if a.helper_overlay:
                for p in a.helper_overlay.glob('*.py'):
                    shutil.copy2(p,out/'helpers'/p.name);helpers[p.name]=sha(p)
            plan.update(epochs=15,selection_policy='synthetic_ap50_95',retain_epoch3_control=True,
                checkpoint_selection='Highest synthetic development AP50-95 at epochs3/6/9/12/15; earliest tie. No real checkpoint selection.',
                kind='gb10_synthetic_duration_'+arm,augmentation=plan.get('augmentation','none'),helpers=helpers,
                input_view=None,duration_control='Save epoch3 before continuing to15; evaluate controls only after all checkpoints freeze.',
                platform='GB10 aarch64; prior desktop metrics are historical; epoch3 retained as same-platform control.')
            plan['limitations']+=['Adaptive research using inspected Run4 development references; not blind accuracy.',
                '15-epoch duration test selects only on synthetic geometry. Real reference scope remains provisional.']
            if 'matched_budget' in plan:plan['matched_budget'].update(epochs=15,total_steps=2310)
            write(out/'experiment-plan.json',plan);write(out/'preparation.json',dict(plan_sha256=sha(out/'experiment-plan.json'),helpers=helpers))
            with (root/(out.name+'.log')).open('w') as log:
                result=subprocess.run([sys.executable,'-u',str(out/'helpers/train_synthetic_liquid_pilot.py'),'fit',
                    '--out',str(out),'--tools',str(bundle/'tools')],cwd=bundle/'repo',env=env,stdout=log,stderr=subprocess.STDOUT)
            if result.returncode:raise RuntimeError(f'{arm} failed, exit {result.returncode}')
            print('COMPLETED',arm,flush=True)
        write(root/'completed-long.json',dict(finished=time.time(),host=os.uname().nodename,arms=['original','paired']))
    except BaseException as error:
        write(root/'failed.json',dict(type=type(error).__name__,error=str(error),finished=time.time()));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['package','remote'])
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--bundle',type=Path)
    p.add_argument('--helper-overlay',type=Path)
    p.add_argument('--wait-for-gpu',action='store_true')
    p.add_argument('--tools',type=Path,default=Path('/mnt/c/Users/Austin/ai-club-tools'))
    args=p.parse_args();(package if args.action=='package' else remote)(args)
