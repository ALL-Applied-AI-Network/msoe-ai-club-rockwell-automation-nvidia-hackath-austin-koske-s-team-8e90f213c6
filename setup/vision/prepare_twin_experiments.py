"""Freeze a small factorial development study before any new rendering."""
import argparse
import json
from pathlib import Path
from twin_contract import default_plan,validate_plan


def plan():
    p=default_plan();p.update(kind='live_twin_process_camera_stream_ablations',diagnostic_suite=True,
        study_version=1,randomization=False,quality_ground_truth='none; synthetic perturbations only',
        stream_on_threshold_unchanged=3.,stream_peak_threshold_unchanged=6.,
        selection='all predeclared cases reported; no default promotion')
    def case(name,group='nominal',prior=False,flush=0.,**kw):
        return dict(id=name,group=group,reference=group=='reference',prior_residue=prior,flush_ml=flush,**kw)
    p['episodes']=[
        case('nominal-reference',group='reference'),case('nominal-repeat'),
        case('residue-2ml',group='residue',prior=True),
        case('residue-after-8ml-prime',group='purge',prior=True,flush=8.),
        case('nominal-flow-slow',flow_scale=.75),case('nominal-flow-fast',flow_scale=1.25),
        case('residue-flow-slow',group='residue',prior=True,flow_scale=.75),
        case('residue-flow-fast',group='residue',prior=True,flow_scale=1.25),
        case('nominal-sensor-green-low',sensor_gain=[1.,.85,1.]),
        case('nominal-sensor-blue-low',sensor_gain=[1.,1.,.85]),
        case('nominal-sensor-exposure-low',sensor_gain=[.9,.9,.9]),
        case('nominal-native-light-low',light_scale=.8),case('nominal-native-light-high',light_scale=1.2),
        case('nominal-anchor-occlusion',anchor_occlusion=True),
        case('residue-halfml-equivalent',group='residue',prior=True,prior_fraction=[.75,.25,0.]),
        case('residue-blue-donor',group='residue',prior=True,prior_fraction=[0.,0.,1.]),
        case('stream-wider',group='appearance_probe',stream_radius_m=.0015),
        case('shared-absorption-shorter',group='appearance_probe',absorption_depth_m=.005),
        case('stream-wider-shorter-absorption',group='appearance_probe',stream_radius_m=.0015,absorption_depth_m=.005),
        case('residue-slow-sensor-green-low',group='residue',prior=True,flow_scale=.75,sensor_gain=[1.,.85,1.]),
        case('nominal-slow-sensor-green-low',flow_scale=.75,sensor_gain=[1.,.85,1.])]
    validate_plan(p);return p


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True)
    a=parser.parse_args()
    if a.out.exists():raise ValueError('Preserve the frozen plan; use a new path')
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(plan(),indent=2)+'\n',encoding='utf-8')
    print('Prepared',len(plan()['episodes']),'predeclared synthetic cases')
