"""Source-verified two-station app replay, using completed inference only."""
import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import shutil


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')


def prepare(results,out):
    import numpy as np
    from PIL import Image,ImageDraw
    out.mkdir(parents=True,exist_ok=False);assets=out/'assets';assets.mkdir()
    prior=results/'two-station-rehearsal-20261023';bundle=prior/'bundle-v1';sources=prior/'sources-v2'
    for name,digest in read(bundle/'manifest.json')['artifacts'].items():
        if sha(bundle/name)!=digest:raise ValueError('Frozen evidence bundle differs')
    source_index=read(sources/'source-assets.json')['assets']
    observations=[json.loads(s) for s in (bundle/'observations.jsonl').read_text().splitlines()]
    samroot=results/'synthetic-iterations-20261022/gb10-results/temporal-sam-refinement-v1'
    completion=read(samroot/'completed.json');samfile=samroot/'pass-1.jsonl'
    if sha(samfile)!=completion['artifacts']['pass-1.jsonl']:raise ValueError('Refinement evidence differs')
    sam={r['source_frame']:r for r in map(json.loads,samfile.read_text().splitlines())}

    def mask(encoded):
        result=np.zeros((480,640),dtype=bool)
        if encoded is None:return result
        x,y=encoded['origin'];w,h=encoded['size']
        if not(0<=x<x+w<=640 and 0<=y<y+h<=480):raise ValueError('Mask outside source')
        flat=np.zeros(w*h,dtype=bool)
        for start,length in encoded['runs']:
            if start<0 or length<=0 or start+length>len(flat):raise ValueError('Bad mask encoding')
            flat[start:start+length]=True
        result[y:y+h,x:x+w]=flat.reshape(h,w);return result

    fill=[]
    for observation in observations:
        if observation['station']!='vial_fill' or observation['observation_id'] not in source_index:continue
        ident=observation['observation_id'];entry=source_index[ident];original=sources/entry['image'];rgb=Image.open(original).convert('RGB')
        if sha(original)!=entry['encoded_file_sha256'] or hashlib.sha256(rgb.tobytes()).hexdigest()!=observation['source']['decoded_rgb_sha256']:raise ValueError('Source pixels differ')
        frame=observation['source']['frame_index'];refined=sam[frame]
        if refined['source_rgb_sha256']!=observation['source']['decoded_rgb_sha256'] or refined['pts']!=observation['source']['pts']:raise ValueError('SAM/source join differs')
        source_path='assets/'+ident+'-source.png';rgb.save(out/source_path);modes={}
        for name,encoded,measurement in [('direct',observation['proposal']['raw_mask'],observation['measurement']),('sam',refined['raw_union'],refined['measurement'])]:
            union=mask(encoded);array=np.asarray(rgb).copy();array[union]=(.60*array[union]+.40*np.array([45,255,160])).astype('uint8')
            overlay=Image.fromarray(array);draw=ImageDraw.Draw(overlay);box=observation['target']['box_xyxy']
            if box:draw.rectangle(box,outline='#58afff',width=2)
            overlay_path='assets/'+ident+'-'+name+'.png';overlay.save(out/overlay_path)
            mask_path='assets/'+ident+'-'+name+'-mask.png';Image.fromarray((union*255).astype('uint8')).save(out/mask_path)
            if name=='direct':qa=measurement['qa_pass'];flags=measurement['qa_reasons']
            else:qa=measurement['measurement_valid'];flags=measurement['quality_flags']
            modes[name]=dict(overlay=overlay_path,mask=mask_path,mask_pixels=int(union.sum()),qa_pass=qa,
                qa_reasons=flags,rgb=measurement['rgb'],lab=measurement['lab'])
        fill.append(dict(id=ident,station='vial_fill',domain='real',source=source_path,source_identity=observation['source'],
            context=observation['target']['context_id'],seconds=float(observation['source']['pts']*Fraction(observation['source']['time_base'])),
            frame_index=frame,models=observation['models'],modes=modes,qualification=observation['qualification'],quality_decision=None))
    if len(fill)!=72:raise ValueError('Expected72 exact source frames')

    caproot=results/'dataset1-cap-confirmation-20261023/gb10-results-v1';modelroot=results/'dataset1-cap-state-20261023/gb10-results-v1'
    capdone=read(caproot/'completed.json')
    if sha(caproot/'results.json')!=capdone['results_sha256']:raise ValueError('Cap completion differs')
    for name,digest in read(caproot/'artifact-manifest.json').items():
        if sha(caproot/name)!=digest:raise ValueError('Cap artifact differs')
    frozen=read(modelroot/'fitted-v1/frozen.json');predictions=read(caproot/'unchanged-predictions.json');classes=['seated_cap','missing_cap','crooked_cap','no_vial'];cap=[]
    for row in map(json.loads,(caproot/'rendered-v1/manifest.jsonl').read_text().splitlines()):
        for normal in (True,False):
            scene=row['id']+('-normal' if normal else '');key='golden' if normal else 'image';file=caproot/'rendered-v1'/row[key]
            if sha(file)!=row[key+'_sha256']:raise ValueError('Cap source differs')
            records=[p for p in predictions if p['scene']==scene]
            if len(records)!=60:raise ValueError('Incomplete cap scene')
            source='assets/'+scene+'-source.png';shutil.copy2(file,out/source)
            image=Image.open(file).convert('RGB');slots=[]
            for p in records:
                if p['source_sha256']!=row[key+'_sha256'] or p['box_xyxy']!=row['boxes'][p['slot']]:raise ValueError('Cap target/source join differs')
                i=p['prediction'];l,t,r,b=p['box_xyxy'];x,y=(l+r)/2,(t+b)/2
                slots.append(dict(id=p['slot'],box=p['box_xyxy'],center=[x,y],class_name=classes[i],probabilities=p['probabilities'],
                    cap_warning=(i!=3 and 1-p['probabilities'][0]>=frozen['thresholds']['0.99']),body_condition='unassessed',
                    anchor_kind='fixture_aperture',is_learned_cap_localization=False))
            overlay='assets/'+scene+'-overlay.png';image.save(out/overlay)
            cap.append(dict(id=scene,station='final_fixture',domain='synthetic',source=source,overlay=overlay,slots=slots,
                context='normal control' if normal else row['nuisance']['density']+' synthetic defects',source_sha256=row[key+'_sha256'],
                model_sha256=frozen['checkpoint_sha256'],warning_threshold=frozen['thresholds']['0.99'],quality_decision=None,
                mapping_qualified=False,body_condition='unassessed',authored_lying_S_coverage=False))
    per_class={}
    for i,name in enumerate(classes):
        selected=[p for p in predictions if p['truth']==i]
        per_class[name]=dict(targets=len(selected),correct=sum(p['prediction']==i for p in selected))
    gated_warning={}
    for i in (1,2):
        selected=[p for p in predictions if p['truth']==i]
        gated_warning[classes[i]]=dict(targets=len(selected),missed=sum(p['prediction']==3 or 1-p['probabilities'][0]<frozen['thresholds']['0.99'] for p in selected))
    data=dict(schema='full-app-evidence-demo/v1',mode='completed model-output replay',fill=fill,cap=cap,
        cap_display=dict(anchor_kind='fixed fixture aperture',learned_cap_localization=False,classification_crop_pixels=96,
            per_class=per_class,display_gated_warning=gated_warning,
            note='Class probabilities apply to a numbered fixture position. The crop center is not the detected center of a cap.'),
        counts=dict(fill_frames=len(fill),cap_states=len(cap),direct_qa=sum(r['modes']['direct']['qa_pass'] for r in fill),sam_qa=sum(r['modes']['sam']['qa_pass'] for r in fill)),
        limits=dict(live_inference=False,physical_vials_linked=False,recipe_acceptance_available=False,body_damage_assessed=False,hardware_commands=0),
        model_names=dict(vial='YOLO11n',liquid='Temporal U-Net',refinement='SAM2.1 Hiera Tiny',cap='Dual-branch reference CNN · cap state'),
        sources=dict(fill_bundle_sha256=sha(bundle/'manifest.json'),sam_pass_sha256=sha(samfile),cap_confirmation_manifest_sha256=sha(caproot/'artifact-manifest.json'),cap_model_sha256=frozen['checkpoint_sha256']))
    write(out/'data.json',data)
    write(out/'manifest.json',dict(schema=data['schema'],files={p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file()},
        training=False,new_inference=False,helper_sha256=sha(__file__)))
    print(json.dumps(data['counts']))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--results',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();prepare(a.results,a.out)
