"""CPU replay of the original October20 prompt policy on matched comparison data."""
import argparse
from fractions import Fraction
import hashlib
import importlib.util
import json
from pathlib import Path


ORIGINAL_SHA='f1d533490623611dbb3a51fbb3240ab4fefad3073927b4d75a175f03c47d1dce'


def main(a):
    import av
    import cv2
    if hashlib.sha256(a.original.read_bytes()).hexdigest()!=ORIGINAL_SHA:raise ValueError('Original policy differs')
    if not (a.comparison/'completed.json').exists():raise ValueError('Wait for the comparison to finish')
    spec=importlib.util.spec_from_file_location('original_prompt_policy',a.original)
    original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)
    rows=[json.loads(s)['arms'] for s in (a.comparison/'pass-1.jsonl').read_text().splitlines()]
    by_frame={r['temporal']['source_frame']:r for r in rows};bridges={};case=None;count=0;prompt_ages=[]
    with av.open(str(a.video)) as container:
        for index,frame in enumerate(container.decode(video=0)):
            if index>max(by_frame):break
            if index not in by_frame:continue
            rgb=frame.to_ndarray(format='rgb24');record=by_frame[index]
            assert hashlib.sha256(rgb.tobytes()).hexdigest()==record['temporal']['source_rgb_sha256']
            if case!=record['temporal']['case_id']:
                case=record['temporal']['case_id'];bridges={arm:original.CausalPromptBridge() for arm in ['legacy_sam','synthetic_sam']}
            for arm in ['legacy_sam','synthetic_sam']:
                r=record[arm];p=original.pool_prompts(rgb,r['vials']+r['liquid_predictions'])
                assert p==r['prompt_policy'],f'Prompt refactor mismatch at {index}/{arm}'
                b=bridges[arm].update(p,index,float(r['pts']*Fraction(r['time_base'])),r['measurement']['motion'])
                assert b==r['bridge'],f'Bridge refactor mismatch at {index}/{arm}'
                if b['prompt_source_frame'] is not None:
                    assert b['prompt_source_frame']<=index and b['prompt_age_s']>=0
                    prompt_ages.append(b['prompt_age_s'])
                count+=1
    assert count==528
    result=dict(status='passed',frames=264,sam_policy_checks=count,
        original_executed_policy_sha256=ORIGINAL_SHA,
        refactored_policy_sha256=hashlib.sha256((a.comparison/'helpers/pool_prompt_policy.py').read_bytes()).hexdigest(),
        all_policies_and_bridges_exact=True,past_only_prompt_sources=True,
        maximum_prompt_age_s=max(prompt_ages,default=None),gpu_used=False,run5_used=False)
    if a.out.exists():raise FileExistsError(a.out)
    a.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['original','comparison','video','out']:p.add_argument('--'+name,type=Path,required=True)
    main(p.parse_args())
