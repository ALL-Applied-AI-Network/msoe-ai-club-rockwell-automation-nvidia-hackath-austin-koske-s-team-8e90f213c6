"""Frozen Run 5 comparison with identical vial inputs, windows and QA rules."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from replay.config import load_config
from replay.dataset import Catalog,DEFAULT_DATA
from replay.pipeline import analyze,write_json,export_csv
from replay.recipes import RecipePlan
from replay.review import export_review


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline-cache',type=Path,required=True,help='Original 13/30 per-image cache, not annotation labels')
    p.add_argument('--data',type=Path,default=DEFAULT_DATA)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--recipe-plan',type=Path)
    p.add_argument('--baseline-base-url',default='http://127.0.0.1:8765')
    p.add_argument('--segmentation-base-url',default='http://127.0.0.1:8766')
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    original=json.loads(a.baseline_cache.read_text());cache_sha=hashlib.sha256(a.baseline_cache.read_bytes()).hexdigest()
    settings={name:json.loads(Path(f'replay/examples/{file}').read_text()) for name,file in
              [('baseline','bootstrap-vision.json'),('segmentation','segmentation-vision.json')]}
    for name,values in settings.items():
        values.update(image_proposal_cache_path=a.baseline_cache.resolve().as_posix(),image_proposal_cache_sha256=cache_sha)
        write_json(a.out/f'{name}-reuse.json',values)
    configs={name:load_config(a.out/f'{name}-reuse.json') for name in settings}
    plan=json.loads(Path('replay/verification/segmentation-experiment-plan.json').read_text())
    for config in configs.values():
        if any(config.to_dict()[key]!=value for key,value in plan['qa_config'].items()):
            raise ValueError('QA config differs from the frozen experiment plan')
    catalog=Catalog(a.data,include_annotations=False);run=catalog.runs[5]
    results={}
    recipes=RecipePlan.load(a.recipe_plan) if a.recipe_plan else RecipePlan()
    for name in settings:
        data=analyze(run,configs[name],a.out/name/'cache');data.recipe_plan=recipes;results[name]=data
        output=a.out/name/'run5';output.mkdir(parents=True,exist_ok=True)
        write_json(output/'manifest.json',data.payload['manifest'])
        write_json(output/'recipe-plan.json',recipes.to_dict())
        export_review(data,run,output/'qa-review',base_url=a.baseline_base_url if name=='baseline' else a.segmentation_base_url)
        completed=data.completed(data.payload['manifest']['frame_count']-1,[])
        write_json(output/'decisions.json',dict(observations=completed))
        (output/'observations.csv').write_text(export_csv(completed),encoding='utf-8',newline='')
    baseline,segmentation=results['baseline'],results['segmentation']
    if baseline.payload['detections']!=segmentation.payload['detections'] or baseline.payload['detection_events']!=segmentation.payload['detection_events']:
        raise ValueError('Fill detection changed; comparison is not controlled')
    for old,current,new in zip(original['features'],baseline.features,segmentation.features):
        for key in ('frame_index','source_rgb_sha256','rgb','lab','quality_flags','measurement_valid','geometry','fill_detection'):
            if old[key]!=current[key]: raise ValueError(f'Original baseline differs at {old["frame_index"]}: {key}')
        for key in ('vial_box','track_id','track_age'):
            if current['geometry'][key]!=new['geometry'][key]: raise ValueError('Vial localization/tracking changed')
    if len(baseline.features)!=len(original['features']) or len(baseline.features)!=len(segmentation.features):
        raise ValueError('Sample coverage changed')
    cases=[]
    for old,new in zip(baseline.observations,segmentation.observations):
        if old['event_id']!=new['event_id'] or old['sample_frames']!=new['sample_frames']:
            raise ValueError('Detected fill or measurement window changed')
        cases.append(dict(detection_id=old['event_id'],sample_frames=old['sample_frames'],
            baseline_qa_valid=old['measurement_valid'],segmentation_qa_valid=new['measurement_valid'],
            baseline_reasons=old['quality_flags'],segmentation_reasons=new['quality_flags'],
            baseline_lab=old['lab'],segmentation_lab=new['lab']))
    report=dict(schema_version=1,kind='controlled_sampling_mask_comparison',source_video_sha256=run['cameras']['vial_fill']['sha256'],
        baseline_cache_sha256=cache_sha,detected_fills=len(cases),samples=len(baseline.features),
        baseline_qa_valid=sum(c['baseline_qa_valid'] for c in cases),segmentation_qa_valid=sum(c['segmentation_qa_valid'] for c in cases),
        gained=[c['detection_id'] for c in cases if c['segmentation_qa_valid'] and not c['baseline_qa_valid']],
        lost=[c['detection_id'] for c in cases if c['baseline_qa_valid'] and not c['segmentation_qa_valid']],
        original_baseline_reproduced_exactly=True,vial_proposals_and_tracks_identical=True,
        measurement_windows_identical=True,fill_detection_and_events_identical=True,qa_config_unchanged=True,
        plan_sha256=hashlib.sha256(Path('replay/verification/segmentation-experiment-plan.json').read_bytes()).hexdigest(),
        model=json.loads(Path('replay/models/sampling-segmenter.json').read_text()),
        segmentation_rejection_counts=dict(Counter(f for c in cases if not c['segmentation_qa_valid'] for f in c['segmentation_reasons'])),
        independent_mask_accuracy=None,trusted_comparisons_enabled=False,
        limitations=['QA coverage is not segmentation accuracy.','Targets are unreviewed conservative sampling masks, not complete liquid outlines.',
                     'Positive-only source sampling and unseen colors/backgrounds can bias the model.','No fallback or union with the baseline mask.'],cases=cases)
    write_json(a.out/'comparison.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('cases','model','limitations')}),flush=True)


if __name__=='__main__': main()
