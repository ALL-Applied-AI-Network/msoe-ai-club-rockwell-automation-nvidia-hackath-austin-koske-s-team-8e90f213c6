"""Join continuous source RGB, inference, captured review and operator events."""
import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from replay.operator.frame_input import FIELDS
from replay.geometry import decode_mask
from replay.measurement import rgb_to_lab,delta_e


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text())


def main(a):
    a.out.mkdir(parents=True,exist_ok=False)
    browser=read(a.browser/'verification.json');export=read(a.browser/'session-export.json')
    assert browser['status']=='passed'
    for name,expected in browser['artifacts'].items():assert sha(a.browser/name)==expected,name
    rows=[json.loads(s) for s in (a.run/'frames.jsonl').read_text().splitlines()]
    truth=[json.loads(s) for s in (a.run/'simulation/private-truth.jsonl').read_text().splitlines()]
    assert len(rows)==len(truth)==browser['captured_frames']
    reference_path=Path(export['state']['reference']['source'])
    assert sha(reference_path)==export['state']['reference']['source_sha256']
    refs={}
    for r in [json.loads(s) for s in reference_path.read_text().splitlines()]:
        h=r['public_source']
        if h['episode_id'].endswith('-reference') and r['measurement']['measurement_valid']:
            refs[(h['phase_key'],round(sum(h['metered_ml']),6))]=r['measurement']['lab']
    by={};previous=None;colors=0
    for i,(record,private) in enumerate(zip(rows,truth),1):
        h=record['source'];row=record['model_output'];m=row['measurement'];d=record['diagnostic']
        assert set(h)==FIELDS
        assert h['sequence']==private['sequence']==i-1
        assert h['segment_id']==private['segment_id']
        assert abs(float(h['pts']*Fraction(h['time_base']))-row['seconds'])<1e-9
        rgb=cv2.cvtColor(cv2.imread(str(a.run/'frames'/f'{i:08d}-source.png')),cv2.COLOR_BGR2RGB)
        assert hashlib.sha256(rgb.tobytes()).hexdigest()==h['rgb_sha256']==private['rgb_sha256']==row['source_rgb_sha256']
        assert private['ack']['accepted_sequence']==h['sequence'] and private['ack']['rgb_sha256']==h['rgb_sha256']
        mask=decode_mask(m['geometry']['mask'],(480,640)).astype(bool)
        saved=cv2.imread(str(a.run/'frames'/f'{i:08d}-mask.png'),cv2.IMREAD_GRAYSCALE)>0
        assert np.array_equal(mask,saved)
        if m['lab'] is not None:
            assert mask.any()
            median=np.median(rgb[mask],axis=0)
            assert np.allclose(m['rgb'],median,atol=1e-9)
            assert np.allclose(m['lab'],rgb_to_lab(median),atol=1e-8);colors+=1
        if record['case_id']!=previous:count=0;warned=False;phase=None
        previous=record['case_id'];telemetry=h['telemetry']
        if phase!=telemetry['phase_key']:count=0
        phase=telemetry['phase_key'];expected=refs.get((phase,round(sum(telemetry['metered_ml']),6)))
        if m['measurement_valid'] and expected is not None:
            score=delta_e(m['lab'],expected);assert abs(score-d['delta_e2000'])<1e-9
            count=count+1 if score>5 else 0;warned=warned or count>=3
        else:count=0;assert d['delta_e2000'] is None
        assert d['warning']==warned
        assert row['recipe_decision']['status']=='unavailable'
        assert not private['ack']['hardware_commands']
        by.setdefault(record['case_id'],[]).append(record)
    events=export['events'];cid=browser['reviewed_case_id']
    case=next(c for c in export['cases'] if c['case_id']==cid)
    assert case['review']=='closed' and not case['resolution']['product_release_authorized']
    evidence=case['evidence'];source=evidence['source'];match=next(r for r in rows if r['source']==source)
    livepng=a.run/'frames'/f'{source["sequence"]+1:08d}-overlay.png'
    assert sha(a.run/evidence['paths']['overlay'])==sha(livepng)
    assert evidence['measurement']==match['model_output']['measurement']
    response=[e for e in events if e['payload'].get('case_id')==cid]
    kinds=[e['kind'] for e in response]
    assert kinds.index('review_requested')<kinds.index('review_acknowledged')<kinds.index('operator_action_recorded')<kinds.index('follow_up_closed')
    assert not any(e['kind'].startswith('simulated_flush') for e in events)
    runtime=read(a.run/'simulation/runtime.json');assets=ROOT/'datasets/mixing_twin/assets/csi_digital_twin_v1'
    for name,expected in runtime['asset_sha256'].items():assert sha(assets/name)==expected
    launch=read(a.run/'launch.json')
    for name,expected in launch['source_sha256'].items():assert sha(a.run/'helpers'/name)==expected
    result=dict(status='passed',mode='continuous_inspection',verified_rgb_inference_ack_frames=len(rows),
        recomputed_sampling_colors=colors,diagnostics_recomputed=True,retained_review_rgb_and_measurement_verified=True,
        operator_actions_do_not_control_source=True,source_assets_unchanged=True,case_count=len(by),
        browser_checks=browser['checks'],hardware_commands=0,operational_recipe_assessment='unavailable',
        model_sha256=read(a.run/'models.json'),isaac_version=runtime['isaac_version'],
        evidence_sha256={str(f.relative_to(a.run)):sha(f) for f in a.run.rglob('*') if f.is_file() and f.suffix in ('.json','.jsonl','.png','.py') and f.name!='frame-input-token.json'},
        limits=['Development simulation and unreviewed model geometry; no real accuracy result.',
                'Separate independent simulated fills; not an operator-triggered process intervention.',
                'Phase and meter telemetry available only for the explicitly simulated source.'])
    (a.out/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='evidence_sha256'},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('run','browser','out'):p.add_argument('--'+name,type=Path,required=True)
    main(p.parse_args())
