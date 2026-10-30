"""Independently verify completed live operator episodes, never drive inference."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'setup/vision'))
from twin_contract import PUBLIC_FIELDS,default_plan,episode_states,digest
from replay.measurement import delta_e
from replay.geometry import decode_mask


def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main(a):
    a.out.mkdir(parents=True,exist_ok=False)
    exported=read(a.browser/'session-export.json');state=exported['state']
    assert state['step']=='complete' and state['review']=='closed'
    assert state['operational_recipe_assessment']=='unavailable' and state['hardware_commands_sent']==0
    events=exported['events'];assert [e['seq'] for e in events]==list(range(1,len(events)+1))
    kinds=[e['kind'] for e in events]
    for earlier,later in zip(['review_requested','review_acknowledged','simulated_flush_requested','simulated_flush_completed'],
                             ['review_acknowledged','simulated_flush_requested','simulated_flush_completed','follow_up_closed']):
        assert kinds.index(earlier)<kinds.index(later)
    assert not events[-1]['payload']['product_release_authorized']
    launch=read(a.run/'launch.json')
    for name,expected in launch['source_sha256'].items():assert sha(a.run/'helpers'/name)==expected,name
    runtime=read(a.run/'simulation/runtime.json')
    assets=ROOT/'datasets/mixing_twin/assets/csi_digital_twin_v1'
    for name,expected in runtime['asset_sha256'].items():assert sha(assets/name)==expected,name
    raw=[json.loads(s) for s in (a.run/'simulation/private-truth.jsonl').read_text().splitlines()]
    truth=[r for r in raw if r['kind']=='frame']
    rows=[json.loads(s) for s in (a.run/'frames.jsonl').read_text().splitlines()]
    assert len(rows)==len(truth)==111
    references={};by={};warning={};count=0;phase=None;episode=None
    for private,row in zip(truth,rows):
        h=row['public_source'];eid=h['episode_id'];m=row['measurement'];d=row['synthetic_color_diagnostic']
        assert set(h)==PUBLIC_FIELDS and h['session_id']==state['session_id']
        assert private['episode_id']==eid and private['state']['frame_index']==h['frame_index']
        image=np.array(Image.open(a.run/'simulation'/private['rgb_file']).convert('RGB'))
        assert image.shape==(480,640,3)
        assert digest(image.tobytes())==h['rgb_sha256']==row['source_rgb_sha256']==private['rgb_sha256']
        assert private['ack']==dict(type='frame_ack',episode_id=eid,frame_index=h['frame_index'],rgb_sha256=h['rgb_sha256'],hardware_commands=0)
        assert abs(row['seconds']-h['sim_time_s'])<1e-8
        assert row['recipe_decision']['status']=='unavailable'
        mask=decode_mask(m['geometry']['mask'],(480,640)).astype(bool)
        assert int(mask.sum())==m['geometry']['mask_pixels']
        geometric=np.array(Image.open(a.run/'simulation'/private['mask_file']))>0
        assert geometric.shape==(480,640)
        total=sum(private['state']['actual_components_ml'])
        assert bool(geometric.any())==(total>1e-8)
        assert abs(total-sum(h['metered_ml']))<1e-7
        if episode!=eid:
            episode=eid;phase=None;count=0;warning[eid]=False
        if phase!=h['phase_key']:count=0
        phase=h['phase_key'];key=(phase,round(sum(h['metered_ml']),6))
        role=eid.rsplit('-',1)[1]
        valid=bool(m['measurement_valid'] and m['lab'] is not None)
        if role=='reference':
            if valid:references[key]=m['lab']
            assert d['delta_e2000'] is None
        else:
            expected=references.get(key)
            if not valid or expected is None:
                count=0;assert d['delta_e2000'] is None
            else:
                score=delta_e(m['lab'],expected)
                assert abs(d['delta_e2000']-score)<1e-9
                count=count+1 if score>5 else 0
                warning[eid]=warning[eid] or count>=3
            assert d['warning']==warning[eid]
        by.setdefault(eid,[]).append(private)
    plan=default_plan()
    flush=next(r for r in raw if r['kind']=='operator_flush')
    test=next(records for eid,records in by.items() if eid.endswith('-test'))
    recovery=next(records for eid,records in by.items() if eid.endswith('-recovery'))
    assert flush['path_before_ml']==test[-1]['state']['path_components_ml']
    for i,before in enumerate(flush['path_before_ml']):
        expected=before*math.exp(-4)+(2*(1-math.exp(-4)) if i==0 else 0)
        assert abs(flush['path_after_ml'][i]-expected)<1e-10
        assert abs(before+(8 if i==0 else 0)-flush['path_after_ml'][i]-flush['result']['discharged_components_ml'][i])<1e-9
    assert recovery[0]['state']['path_components_ml']==flush['path_after_ml']
    for eid,records in by.items():
        role=eid.rsplit('-',1)[1]
        episode=dict(id=role,reference=role=='reference',prior_residue=role!='reference',flush_ml=0.)
        if role=='recovery':episode['prior_fraction']=[v/2 for v in flush['path_after_ml']]
        expected=list(episode_states(plan,episode))
        assert len(records)==len(expected)==37
        for item,wanted in zip(records,expected):
            for key in ('metered_ml','actual_components_ml','path_components_ml'):
                assert np.allclose(item['state'][key],wanted[key],atol=1e-10,rtol=0)
    browser=read(a.browser/'verification.json')
    assert browser['status']=='passed'
    for name,expected in browser['artifacts'].items():assert sha(a.browser/name)==expected
    report=dict(status='passed',kind='interactive_native_isaac_operator_walkthrough',
        live_rgb_inference_ack_frames=len(rows),episodes=state['episodes'],isaac_version=runtime['isaac_version'],
        operational_decisions=dict(Counter(r['recipe_decision']['status'] for r in rows)),
        operational_unavailable_reasons=dict(Counter(r['recipe_decision']['reason'] for r in rows)),
        actual_model_warning_origin=next(e['payload']['origin'] for e in events if e['kind']=='review_requested'),
        retained_path_before_flush_ml=flush['path_before_ml'],retained_path_after_flush_ml=flush['path_after_ml'],
        process_trace_and_mass_balance=True,operator_receipt_precedes_recovery=True,
        frame_diagnostics_independently_recomputed=True,private_truth_fields_in_public_frames=0,
        source_assets_unchanged=True,hardware_commands_sent=0,physical_calibration=False,run5_used=False,
        model_sha256=read(a.run/'models.json'),browser_checks=browser['checks'],
        source_run=str(a.run),browser_run=str(a.browser),
        evidence_sha256={str(f.relative_to(a.run)):sha(f) for f in a.run.rglob('*')
            if f.is_file() and f.suffix in ('.json','.jsonl','.png','.py')},
        limits=['One scenario walkthrough, not a statistical accuracy study.',
            'Nominal reference and optical/process assumptions are not physically calibrated.',
            'Geometric masks are evaluator evidence, not reviewed human masks.',
            'Operator-directed software simulation; no industrial output or approved procedure.',
            'Operational recipe assessment stays unavailable.'])
    (a.out/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='evidence_sha256'},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('run','browser','out'):p.add_argument('--'+name,type=Path,required=True)
    main(p.parse_args())
