"""Read-only asset checks for station onboarding; never authorizes deployment.

This checks local files and declared roles/domains, not optical truth, reviewer
identity, model accuracy or the contents of qualification/evaluation reports.
Use the existing capture inspector, review scorers and commissioning gates for
those stages. No model imports, GPU use, installers, servers or hardware calls.
"""
import argparse
import hashlib
import json
from pathlib import Path


REQUIRED={
    'camera_profile':'Confirm source dimensions, ROI, orientation and clock handling.',
    'synthetic_manifest':'Supply the frozen synthetic source/family split manifest.',
    'checkpoint':'Supply the selected liquid checkpoint and its expected hash.',
    'training_evidence':'Supply fitting/selection provenance, including generic pretraining.',
    'temporal_reference':'Capture and independently qualify the earlier aligned empty reference.',
    'real_evaluation_report':'Score independent human review; retain uncertainty and exclusions.',
    'live_rehearsal_report':'Measure the integrated inference/QA/operator loop, including failure cases.',
}
OPTIONAL={'accepted_recipe_reference','deployment_manifest'}


def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def inspect(request,base):
    if request.get('schema')!='station-onboarding-request/v1':raise ValueError('Unknown onboarding request schema')
    if request.get('task')!='vial_fill':raise ValueError('This pilot covers vial_fill; other tasks need their own profile and checks')
    if not isinstance(request.get('station_id'),str) or not request['station_id'].strip():raise ValueError('station_id required')
    if not isinstance(request.get('goal'),str) or not request['goal'].strip():raise ValueError('Inspection goal required')
    assets=request.get('assets',[])
    if not isinstance(assets,list):raise ValueError('assets must be an array')
    seen=set();records=[];issues=[]
    for asset in assets:
        if not isinstance(asset,dict):raise ValueError('Each asset must be an object')
        role=asset.get('role')
        if role not in REQUIRED.keys()|OPTIONAL:raise ValueError('Unknown asset role')
        if role in seen:raise ValueError('Duplicate role: '+role)
        seen.add(role)
        if not isinstance(asset.get('path'),str) or not asset['path']:raise ValueError('Asset path required')
        path=(base/asset['path']).resolve();row=dict(role=role,path=str(path),exists=path.is_file(),issues=[])
        declared=asset.get('data_domain');row['declared_data_domain']=declared
        if declared not in ('real','synthetic','mixed','not_applicable'):row['issues'].append('data_domain_not_declared')
        if role=='synthetic_manifest' and declared!='synthetic':row['issues'].append('new_CSI_training_requires_synthetic_data')
        expected=asset.get('sha256');row['expected_sha256']=expected
        valid_digest=isinstance(expected,str) and len(expected)==64 and all(c in '0123456789abcdefABCDEF' for c in expected)
        if not valid_digest:row['issues'].append('expected_sha256_missing_or_invalid')
        if path.is_file():
            before=path.stat();row.update(bytes=before.st_size,actual_sha256=sha(path));after=path.stat()
            if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):row['issues'].append('asset_changed_during_read')
            if valid_digest and expected.lower()!=row['actual_sha256']:row['issues'].append('sha256_mismatch')
        else:row['issues'].append('asset_missing')
        issues.extend(dict(role=role,reason=reason) for reason in row['issues']);records.append(row)
    for role in sorted(REQUIRED.keys()-seen):issues.append(dict(role=role,reason='required_asset_not_supplied'))
    # File presence/hashes must never become a physical deployment approval.
    return dict(schema='station-onboarding-readiness/v1',station_id=request['station_id'],task=request['task'],goal=request['goal'],
        status='asset_checks_incomplete' if issues else 'asset_checks_passed',assets=records,issues=issues,
        next_actions=[dict(role=r,action=REQUIRED.get(r,'Inspect the optional asset.'),reasons=[i['reason'] for i in issues if i['role']==r]) for r in sorted({i['role'] for i in issues})],
        qualification_checks_executed=False,real_accuracy=None,deployment_authorized=False,
        source_modified=False,training_started=False,GPU_used=False,hardware_connections=0,
        pending_validation=['Decode/source-clock and camera geometry checks','Synthetic family split and fitting/selection provenance inspection',
            'Reference condition, alignment, causality and human qualification','Reviewed mask/presence/onset scoring',
            'Integrated QA, operator response and end-to-end latency rehearsal','Explicit deployment authorization and approved feed arrangement'],
        limits=['Assets are hashed, not semantically validated by this helper.',
            'Domain declarations are attributed metadata, not independent proof of pixel origin.',
            'Existing commissioning/reference and review scorers remain authoritative for their checks.',
            'Even all asset checks passing does not approve references, product quality or deployment.'])


def run(request_path,out):
    request_path=request_path.resolve();out=out.resolve()
    if out.exists():raise ValueError('Use a fresh output directory')
    original=request_path.read_bytes();request=json.loads(original)
    # Keep outputs outside all supplied source directories, even on an incomplete request.
    protected=[request_path.parent]
    for asset in request.get('assets',[]):
        if isinstance(asset,dict) and isinstance(asset.get('path'),str):
            source=(request_path.parent/asset['path']).resolve();protected.append(source if source.is_dir() else source.parent)
    if any(out==parent or out.is_relative_to(parent) for parent in protected):raise ValueError('Output must be outside request/source directories')
    report=inspect(request,request_path.parent)
    if request_path.read_bytes()!=original:raise ValueError('Request changed during read')
    report['request_sha256']=hashlib.sha256(original).hexdigest()
    out.mkdir(parents=True)
    (out/'request.snapshot.json').write_bytes(original)
    encoded=json.dumps(report,indent=2,allow_nan=False)+'\n';(out/'readiness.json').write_text(encoded,encoding='utf-8')
    lines=['# Station onboarding asset check','',f'Station: {report["station_id"]}',f'Status: {report["status"]}',
        '', 'This report checks files and hashes. Qualification, real accuracy and deployment are not established.','', 'Next inputs:','']
    lines.extend('- '+a['role']+': '+a['action']+' Reasons: '+', '.join(a['reasons']) for a in report['next_actions'])
    if not report['next_actions']:lines.append('- Inspect semantic evidence and run the pending validation stages before any deployment decision.')
    (out/'NEXT-STEPS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (out/'completed.json').write_text(json.dumps(dict(artifacts={p.name:sha(p) for p in out.iterdir() if p.is_file()},
        status=report['status'],deployment_authorized=False),indent=2)+'\n',encoding='utf-8')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();result=run(args.request,args.out)
    print(json.dumps(dict(status=result['status'],issues=len(result['issues']),deployment_authorized=False,out=str(args.out)),indent=2))
    raise SystemExit(2 if result['issues'] else 0)
