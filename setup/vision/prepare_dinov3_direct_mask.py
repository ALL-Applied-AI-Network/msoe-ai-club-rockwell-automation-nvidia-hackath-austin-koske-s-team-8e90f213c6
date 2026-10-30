"""Pin a SAM-free pilot payload: synthetic images/masks, generic backbone, helpers."""
import argparse
from pathlib import Path
import shutil
import tarfile

from train_pair_masks import read,write,sha


def prepare(parent,backbone,baseline,smoke,out):
    out.mkdir(parents=True,exist_ok=True)
    if (out/'payload-v1').exists() or (out/'input.tar.gz').exists():raise ValueError('Preserve existing experiment input')
    payload=out/'payload-v1';payload.mkdir();training=payload/'training';(training/'specs').mkdir(parents=True)
    rows=read(parent/'specs/paired-native.json');original_manifest=read(parent/'manifest.json');manifest={}
    required={'specs/paired-native.json'}
    for row in rows:
        if row['split'] not in ('train','val') or not row['training_eligible']:raise ValueError('Unexpected training source')
        required.update(['data/paired/images/'+row['id']+'.png','data/paired/masks/'+row['id']+'.png'])
    for relative in sorted(required):
        source=parent/relative
        if sha(source)!=original_manifest[relative]:raise ValueError('Parent input differs: '+relative)
        target=training/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target);manifest[relative]=sha(target)
    write(training/'manifest.json',manifest)
    (payload/'pretrained').mkdir();backbone_hashes={}
    for name in ('config.json','preprocessor_config.json','model.safetensors'):
        shutil.copy2(backbone/name,payload/'pretrained'/name);backbone_hashes[name]=sha(backbone/name)
    helpers={}
    for name in ('dinov3_liquid_mask.py','train_dinov3_liquid_mask.py','train_pair_masks.py'):
        source=Path(__file__).with_name(name);shutil.copy2(source,payload/name);helpers[name]=sha(source)
    (payload/'baseline').mkdir()
    metadata=read(baseline/'frozen-model.json')
    if sha(baseline/'best.pt')!=metadata['checkpoint_sha256']:raise ValueError('Frozen U-Net differs')
    for name in ('best.pt','frozen-model.json'):shutil.copy2(baseline/name,payload/'baseline'/name)
    smoke_meta=read(smoke/'completed.json')
    if smoke_meta['model_helper_sha256']!=helpers['dinov3_liquid_mask.py']:raise ValueError('CPU smoke/model helper mismatch')
    shutil.copy2(smoke/'completed.json',payload/'cpu-smoke.json')
    write(payload/'experiment-plan.json',dict(schema='dinov3-direct-mask/v1',modes=['rgb','paired'],epochs=60,batch_size=8,seed=19,
        helpers=helpers,backbone_files=backbone_hashes,training_manifest_sha256=sha(training/'manifest.json'),
        train_images=144,development_images=48,train_families=36,development_families=12,synthetic_test_set_available=False,
        checkpoint_selection='Highest synthetic development presence accuracy plus positive geometric IoU, every5 epochs, earliest tie',
        mask_threshold=.5,presence_threshold=.5,backbone='Generic pretrained DINOv3 ConvNeXt Small, frozen',
        head='FPN32 multi-scale fusion +16-channel native RGB detail + mask and presence heads',
        normalization='ImageNet native mean/std; native112x192 station ROI; no224x224 default resize',
        optimizer='AdamW lr.001 weight_decay.0001, heads only',augmentation='Matched to earlier U-Net pilot: shared/independent RGB gain, offset, noise and1px reference jitter',
        SAM_used=False,real_training_runs=[],real_selection_runs=[],run5_used=False,
        baseline_checkpoint_sha256=metadata['checkpoint_sha256'],
        evaluation='Frozen models evaluated on same264 existing Run4 development frames with unchanged vial detector and measurement QA; no reviewed real mask accuracy',
        real_sampling_threshold_tuning=False,app_defaults_changed=False,local_GPU_training=False,
        limits=['Synthetic supervision is geometric liquid-body mask, not reviewed optical truth.',
            'Small study reuses inspected scene family and real development clips.',
            'Generic backbone is49.5M parameters; a small head does not make total model smaller than SAM2.1 Tiny.',
            'Paired mode retains reference qualification/alignment dependency.']))
    archive=out/'input.tar.gz'
    with tarfile.open(archive,'x:gz',compresslevel=1) as tar:
        for p in payload.rglob('*'):
            if p.is_file():tar.add(p,arcname=p.relative_to(payload).as_posix(),recursive=False)
    write(out/'transfer.json',dict(archive_sha256=sha(archive),bytes=archive.stat().st_size,plan_sha256=sha(payload/'experiment-plan.json'),training_real_runs=[]))
    print(read(out/'transfer.json'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('parent','backbone','baseline','smoke','out'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();prepare(a.parent,a.backbone,a.baseline,a.smoke,a.out)
