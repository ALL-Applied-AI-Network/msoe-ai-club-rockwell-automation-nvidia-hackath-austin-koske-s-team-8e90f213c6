"""Create private, pinned offline refs without changing shared model-cache refs."""
import argparse
import json
from pathlib import Path


def siglip(root):
    guard=root/'model-cache/hub/models--nvidia--Cosmos-Guardrail1/snapshots/d6d4bfa899a71454a700907664f3e88f503950cf'
    name='models--google--siglip-so400m-patch14-384'
    source=guard/'video_content_safety_filter'/name
    target=root/'model-cache/hub'/name
    if target.exists():raise ValueError('SigLIP cache already exists')
    revision='9fdffc58afc957d1a03a25b10dba0329ab15c2a3'
    if not (source/'snapshots'/revision/'model.safetensors').is_file():raise ValueError('Pinned guardrail SigLIP missing')
    target.mkdir()
    for part in ['blobs','snapshots']:(target/part).symlink_to(source/part,target_is_directory=True)
    (target/'refs').mkdir();(target/'refs/main').write_text(revision)


def main(a):
    if a.siglip_only:siglip(a.root);return
    manifest=json.loads((a.root/('context-downloads-completed.json' if a.add_context else 'downloads-completed.json')).read_text())
    target=a.root/'model-cache'
    if target.exists() and not a.add_context:raise ValueError('Private cache already exists; inspect it')
    target.mkdir(exist_ok=a.add_context)
    for model in manifest['repositories']:
        name='models--'+model['repo'].replace('/','--');shared=Path.home()/'.cache/huggingface/hub'/name
        dest=target/'hub'/name;dest.mkdir(parents=True)
        revision=model['revision']
        for file in model['files']:
            p=shared/'snapshots'/revision/file['file']
            if not p.is_file() or p.stat().st_size!=file['size']:raise ValueError('Verified cache file missing')
        for part in ['blobs','snapshots']:(dest/part).symlink_to(shared/part,target_is_directory=True)
        (dest/'refs').mkdir();(dest/'refs/main').write_text(revision)
    if a.add_context:return
    siglip(a.root)
    (a.root/'private-cache-completed.json').write_text(json.dumps(dict(kind='private_offline_hf_refs',
        source_manifest='downloads-completed.json',token_stored=False,shared_refs_modified=False),indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--siglip-only',action='store_true');p.add_argument('--add-context',action='store_true')
    main(p.parse_args())
