"""Download pinned trial dependencies using an ephemeral stdin token.

No login, token file, credential environment variable or credential output is
created on the shared compute account. This is CPU/network preparation only.
"""
import argparse
import fnmatch
import hashlib
import json
import os
from pathlib import Path
import sys
import time

from huggingface_hub import HfApi, snapshot_download

PINS=[
    ('nvidia/Cosmos-Transfer2.5-2B','b67b64abda3801a9aceddbff2bdb86126c06db74',
     ['general/edge/61f5694b-0ad5-4ecd-8ad7-c8545627d125_ema_bf16.pt']),
    ('nvidia/Cosmos-Predict2.5-2B','f176dc95b4a70f53ce01c4b302851595e7322b00',['tokenizer.pth']),
    ('nvidia/Cosmos-Reason1-7B','3210bec0495fdc7a8d3dbb8d58da5711eab4b423',None),
    ('nvidia/Cosmos-Guardrail1','d6d4bfa899a71454a700907664f3e88f503950cf',None),
    ('Qwen/Qwen3Guard-Gen-0.6B',None,None),
    ('Qwen/Qwen2.5-VL-7B-Instruct',None,'metadata_only'),
    ('google/siglip2-so400m-patch16-naflex',None,['*.json','*.txt','*.model','*.safetensors']),
]


def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        while b:=f.read(8*1024*1024):h.update(b)
    return h.hexdigest()


def main(a):
    prefix='context-' if a.public_context_only else ''
    if (a.root/(prefix+'downloads-completed.json')).exists():raise ValueError('Downloads already completed; inspect existing evidence')
    token=False if a.public_context_only else sys.stdin.readline().strip()
    if not a.public_context_only and not token:raise ValueError('Provide a token on stdin; it is never persisted')
    api=HfApi(token=token);items=[];started=time.time()
    for repo,revision,patterns in (PINS[-1:] if a.public_context_only else PINS):
        info=api.model_info(repo,revision=revision,files_metadata=True)
        selected=info.siblings
        if patterns=='metadata_only':
            selected=[s for s in selected if not s.rfilename.endswith(('.safetensors','.bin','.pt','.pth'))]
            patterns=[s.rfilename for s in selected]
        elif patterns is not None:selected=[s for s in selected if any(fnmatch.fnmatchcase(s.rfilename,p) for p in patterns)]
        print('DOWNLOAD',repo,info.sha,'files',len(selected),flush=True)
        path=Path(snapshot_download(repo,revision=info.sha,allow_patterns=patterns,token=token,max_workers=4))
        files=[]
        for s in selected:
            p=path/s.rfilename
            if not p.is_file() or p.stat().st_size!=s.size:raise ValueError('Downloaded file missing or wrong size')
            h=digest(p);expected=s.lfs.sha256 if s.lfs else None
            if expected is not None and h!=expected:raise ValueError('LFS hash mismatch')
            files.append(dict(file=s.rfilename,size=s.size,sha256=h,lfs_hash_verified=expected is not None))
        items.append(dict(repo=repo,revision=info.sha,files=files))
        (a.root/(prefix+'download-progress.json')).write_text(json.dumps(items,indent=2)+'\n')
        print('VERIFIED',repo,len(files),flush=True)
    token=None
    (a.root/(prefix+'downloads-completed.json')).write_text(json.dumps(dict(repositories=items,
        elapsed_s=time.time()-started,token_persisted=False,training=False),indent=2)+'\n')
    print('DOWNLOADS COMPLETED',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--public-context-only',action='store_true')
    try:main(p.parse_args())
    except Exception as e:
        # Auth/transport exceptions can contain signed URLs. Never print their text.
        print('DOWNLOAD FAILED',type(e).__name__,flush=True);sys.exit(1)
