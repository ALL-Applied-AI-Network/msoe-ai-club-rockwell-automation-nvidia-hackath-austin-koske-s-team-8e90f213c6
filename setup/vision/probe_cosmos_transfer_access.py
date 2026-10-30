"""Read-only checkpoint access check. Never records tokens, headers or redirects."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import urllib.request
import urllib.error

MODELS=[('Cosmos-Transfer2.5-2B','ce8440327c632d8313c3bde69db13b627ba5cae1',
    'general/edge/61f5694b-0ad5-4ecd-8ad7-c8545627d125_ema_bf16.pt'),
    ('Cosmos-Guardrail1','d6d4bfa899a71454a700907664f3e88f503950cf','video_content_safety_filter/safety_filter.pt'),
    ('Cosmos-Predict2.5-2B','f176dc95b4a70f53ce01c4b302851595e7322b00','tokenizer.pth'),
    ('Cosmos-Reason1-7B','3210bec0495fdc7a8d3dbb8d58da5711eab4b423','model-00001-of-00004.safetensors')]


def main(a):
    if a.out.exists():raise ValueError('Use a fresh access evidence file')
    token=os.environ.get('HF_TOKEN')
    if not token and a.token_file.exists():token=a.token_file.read_text().strip()
    results=[]
    for model,revision,file in MODELS:
        url=f'https://huggingface.co/nvidia/{model}/resolve/{revision}/{file}'
        req=urllib.request.Request(url,method='HEAD',headers={'Authorization':'Bearer '+token} if token else {})
        item=dict(model='nvidia/'+model,revision=revision,file=file)
        try:
            with urllib.request.urlopen(req,timeout=30) as r:item.update(status=r.status,accessible=r.status==200)
        except urllib.error.HTTPError as e:
            item.update(status=e.code,accessible=False,error_code=e.headers.get('X-Error-Code'))
        except (urllib.error.URLError,TimeoutError):item.update(status=None,accessible=False,error_code='network_unavailable')
        results.append(item)
    report=dict(checked_utc=datetime.now(timezone.utc).isoformat(),token_present=bool(token),
        checkpoints=results,all_accessible=all(r['accessible'] for r in results),downloaded_weights=False)
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--token-file',type=Path,default=Path.home()/'.cache/huggingface/token')
    main(p.parse_args())
