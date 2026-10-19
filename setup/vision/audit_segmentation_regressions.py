"""Post-hoc adapter audit of the eight original Run 5 segmentation regressions.

Reads saved replay evidence, not evaluation annotations. Re-infers only the
requested diagnostic frames; never fits a model or changes a QA decision.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from replay.config import load_config
from replay.geometry import decode_mask
from replay.measurement import rgb_to_lab,delta_e
from replay.segmentation import SamplingSegmenter,crop_input,INPUT_SIZE


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')


def save_image(path,array):
    if not cv2.imwrite(str(path),array): raise IOError(path)
    return dict(path=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def audit_review(root,document):
    """Check exported pixels/aggregates for every window, including rejections."""
    artifacts=samples=0
    for case in document['cases']:
        for frame in case['representative_frames']:
            for kind in ('source','mask','overlay'):
                path=root/frame['path' if kind=='source' else kind+'_path']
                sha=frame['sha256' if kind=='source' else kind+'_sha256']
                if hashlib.sha256(path.read_bytes()).hexdigest()!=sha: raise ValueError('Artifact hash mismatch')
                artifacts+=1
        for sample in case['samples']:
            frame=next(f for f in case['representative_frames'] if f['frame_index']==sample['frame_index'])
            rgb=cv2.cvtColor(cv2.imread(str(root/frame['path'])),cv2.COLOR_BGR2RGB)
            if hashlib.sha256(rgb.tobytes()).hexdigest()!=sample['source_rgb_sha256']: raise ValueError('Source pixels mismatch')
            mask=decode_mask(sample['geometry']['mask']).astype(bool)
            np.testing.assert_array_equal(cv2.imread(str(root/frame['mask_path']),0)>0,mask)
            if int(mask.sum())!=sample['geometry']['mask_pixels']: raise ValueError('Mask count mismatch')
            if mask.any():
                color=np.median(rgb[mask],axis=0)
                np.testing.assert_allclose(color,sample['measured_rgb'],atol=0)
                np.testing.assert_allclose(rgb_to_lab(color),sample['measured_lab'],atol=1e-10)
            elif sample['measured_rgb'] is not None or sample['measured_lab'] is not None:
                raise ValueError('Color without sampling pixels')
            stages=sample['geometry'].get('mask_stages',{})
            prior=None
            for name in ('model_proposal','physical_interior','hsv','opened','largest_component','eroded'):
                if name not in stages: continue
                current=decode_mask(stages[name]).astype(bool)
                if prior is not None and np.any(current & ~prior): raise ValueError('Postprocessing invented foreground')
                prior=current
            if prior is not None and np.any(mask & ~prior): raise ValueError('Final trim invented foreground')
            samples+=1
        accepted=[s for s in case['samples'] if s['qa_valid']]
        if accepted:
            np.testing.assert_allclose(np.median([s['measured_rgb'] for s in accepted],axis=0),case['measured_rgb'])
            center=np.median([s['measured_lab'] for s in accepted],axis=0)
            np.testing.assert_allclose(center,case['measured_lab'])
            np.testing.assert_allclose(max(delta_e(s['measured_lab'],center) for s in accepted),case['temporal_spread_de2000'])
    return dict(cases=len(document['cases']),window_samples=samples,hashed_pngs=artifacts,
                exact_sample_masks_colors_and_aggregates=True,no_foreground_added_by_postprocessing=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous-study',type=Path,required=True)
    parser.add_argument('--current-study',type=Path,required=True)
    args=parser.parse_args()
    previous=read(args.previous_study/'comparison.json')
    current=read(args.current_study/'comparison.json')
    before=read(args.previous_study/'segmentation/run5/qa-review/manifest.json')
    after=read(args.current_study/'segmentation/run5/qa-review/manifest.json')
    evidence_audit=audit_review(args.current_study/'segmentation/run5/qa-review',after)
    config=load_config(args.current_study/'segmentation-reuse.json')
    model=SamplingSegmenter(config.liquid_segmentation_model_path,config.liquid_segmentation_model_sha256)
    output=args.current_study/'regression-audit';output.mkdir(exist_ok=True)
    cases=[];tiles=[]
    for detection in previous['lost']:
        old=next(c for c in before['cases'] if c['detection_id']==detection)
        new=next(c for c in after['cases'] if c['detection_id']==detection)
        if old['sample_frames']!=new['sample_frames']: raise ValueError('Window changed')
        rows=[]
        for sample in new['samples']:
            index=sample['frame_index'];geometry=sample['geometry']
            representative=next(f for f in old['representative_frames'] if f['frame_index']==index)
            rgb=cv2.cvtColor(cv2.imread(str(args.previous_study/'segmentation/run5/qa-review'/representative['path'])),cv2.COLOR_BGR2RGB)
            source_sha=hashlib.sha256(rgb.tobytes()).hexdigest()
            if source_sha!=sample['source_rgb_sha256']: raise ValueError('Source RGB mismatch')
            vial=geometry['vial_box'];logits,box=model.predict(rgb,vial)
            crop,_=crop_input(rgb,vial);x0,y0,x1,y1=box
            probability=np.zeros(rgb.shape[:2],np.float32)
            probability[y0:y1,x0:x1]=1/(1+np.exp(-np.clip(logits,-30,30)))
            # Independent transform calculation: exact RGB crop, same axis order,
            # resize logits before thresholding. No letterbox inside vial crops.
            model.net.setInput(cv2.dnn.blobFromImage(crop,1/255.,INPUT_SIZE,swapRB=False))
            network_logits=model.net.forward()[0,0]
            projected=cv2.resize(network_logits,(x1-x0,y1-y0),interpolation=cv2.INTER_LINEAR)
            if not np.array_equal(logits,projected): raise ValueError('Projection mismatch')
            proposal=model.propose(rgb,vial)
            original_sample=next(s for s in old['samples'] if s['frame_index']==index)
            original_boxes=sorted(p['box'] for p in original_sample['geometry']['proposals'] if p['class_id']==1)
            components=proposal[0]['raw_components'] if proposal else []
            if original_boxes!=sorted(p['box'] for p in components):
                raise ValueError('Raw component projection changed from original experiment')
            raw=decode_mask(proposal[0]['candidate_mask']) if proposal else np.zeros(rgb.shape[:2],np.uint8)
            recorded=decode_mask(geometry.get('mask_stages',{}).get('model_proposal'))
            if not np.array_equal(raw,recorded): raise ValueError('Recorded proposal mismatch')
            physical=np.zeros_like(raw);vx,vy,vr,vb=vial;vw,vh=vr-vx,vb-vy
            physical[int(vy+.58*vh):int(vb-.07*vh),int(vx+.2*vw):int(vr-.2*vw)]=1
            stage_masks={'raw_threshold':(probability>.5).astype(np.uint8),
                         'retained_proposal':raw,'vial_domain_only':raw*physical}
            stage_masks.update({key:decode_mask(value) for key,value in geometry.get('mask_stages',{}).items()})
            stage_masks['final_sample']=decode_mask(geometry['mask'])
            stem=f'{detection}-f{index:06d}';artifacts={}
            artifacts['input_crop']=save_image(output/(stem+'-input.png'),cv2.cvtColor(crop,cv2.COLOR_RGB2BGR))
            artifacts['probability']=save_image(output/(stem+'-probability.png'),np.round(probability*65535).astype(np.uint16))
            for name,mask in stage_masks.items():
                artifacts[name]=save_image(output/(stem+'-'+name+'.png'),mask*255)
            bounds=geometry.get('sampling_bounds')
            area=max(0,bounds[2]-bounds[0])*max(0,bounds[3]-bounds[1]) if bounds else 0
            row=dict(frame_index=index,mp4_s=sample['mp4_s'],source_rgb_sha256=source_sha,
                     crop_box=box,input_size=list(INPUT_SIZE),projection_exact=True,
                     original_raw_component_boxes_identical=True,
                     source_image=representative['path'],source_root=str(args.previous_study/'segmentation/run5/qa-review'),
                     model_components=components,
                     sampling_bounds=bounds,sampling_rectangle_area=area,
                     coherent_area_fraction=float(stage_masks['eroded'].sum()/area) if area and 'eroded' in stage_masks else None,
                     stage_pixels={key:int(value.sum()) for key,value in stage_masks.items()},
                     qa_valid=sample['qa_valid'],qa_flags=sample['qa_flags'],artifacts=artifacts)
            rows.append(row)
            if index==new['sample_frames'][0]:
                panels=[]
                for title,mask in [('source',None),('model .5',stage_masks['raw_threshold']),
                                   ('vial domain',stage_masks['vial_domain_only']),
                                   ('HSV',stage_masks.get('hsv',np.zeros_like(raw))),
                                   ('final sample',stage_masks['final_sample'])]:
                    panel=rgb.copy()
                    if mask is not None:
                        panel=(panel.astype(float)*.35).astype(np.uint8)
                        panel[mask.astype(bool)]=rgb[mask.astype(bool)]
                        contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                        cv2.drawContours(panel,contours,-1,(0,255,255),1)
                    panel=cv2.resize(panel[max(0,vy-10):min(480,vb+10),max(0,vx-20):min(640,vr+20)],(150,480))
                    panel=cv2.cvtColor(panel,cv2.COLOR_RGB2BGR)
                    cv2.putText(panel,title,(3,18),0,.42,(255,255,255),1)
                    panels.append(panel)
                strip=np.hstack(panels)
                label=np.zeros((42,strip.shape[1],3),np.uint8)
                cv2.putText(label,f'{detection}  frame {index}  after QA: {sample["qa_valid"]}',(5,26),0,.6,(255,255,255),1)
                tiles.append(np.vstack([label,strip]))
        cases.append(dict(detection_id=detection,before_qa_valid=old['measurement_available'],
                          after_qa_valid=new['measurement_available'],after_reasons=new['rejection_reasons'],samples=rows))
    report=dict(schema_version=1,kind='posthoc_segmentation_adapter_debug',
                source_video_sha256=after['source']['video_sha256'],model_sha256=config.liquid_segmentation_model_sha256,
                previous_qa_valid=previous['segmentation_qa_valid'],current_qa_valid=current['segmentation_qa_valid'],
                baseline_qa_valid=current['baseline_qa_valid'],gained_vs_baseline=current['gained'],lost_vs_baseline=current['lost'],
                inspected_regressions=len(cases),inspected_frames=sum(len(c['samples']) for c in cases),
                all_window_evidence_audit=evidence_audit,
                probability_png_encoding='uint16 probability scaled by 65535; zero outside crop',
                qa_thresholds_windows_and_model_unchanged=True,not_a_new_blind_evaluation=True,
                safeguards=['No baseline mask fallback or foreground addition.','All original insets and morphology retained.',
                            'QA coverage is not reviewed segmentation precision.'],cases=cases)
    write(output/'manifest.json',report)
    save_image(output/'all-eight-regressions.png',np.vstack(tiles))
    print(json.dumps({k:v for k,v in report.items() if k!='cases'}))


if __name__=='__main__': main()
