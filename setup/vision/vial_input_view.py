"""Synthetic-only lower-vial crop pilot. Mapping always returns full-image boxes."""
import argparse
import copy
from pathlib import Path
import shutil

from target_synthetic_transfer import read,write,sha
from vial_basal_policy import active_vial

VIEW=dict(kind='lower_vial_letterbox_v1',start_height_fraction=.5,width_margin_fraction=.10,
          bottom_margin_fraction=.02,canvas=[640,480],fill_rgb=[128,128,128],
          selection='One active synthetic-trained YOLO vial, same station association as existing pipeline.',
          limitations='Lower-compartment visible body only; upper liquid hidden by the crop is outside this input view.')
VIAL_OPTIONS=dict(imgsz=640,rect=False,conf=.35,iou=.45,max_det=300,device='cuda:0',
                  quantize=None,augment=False,agnostic_nms=False,verbose=False,save=False)


def make_view(rgb,vials,view=VIEW):
    import numpy as np
    from PIL import Image
    vial=active_vial(vials)
    canvas=Image.new('RGB',(640,480),tuple(view['fill_rgb']))
    if vial is None:return np.asarray(canvas).copy(),None
    x,y,r,b=vial;w=r-x;h=b-y
    box=[max(0,int(x-w*view['width_margin_fraction'])),max(0,int(y+h*view['start_height_fraction'])),
         min(640,int(r+w*view['width_margin_fraction'])+1),min(480,int(b+h*view['bottom_margin_fraction'])+1)]
    a,c,d,e=box
    if d<=a or e<=c:return np.asarray(canvas).copy(),None
    scale=min(640/(d-a),480/(e-c))
    nw=max(1,round((d-a)*scale));nh=max(1,round((e-c)*scale));ox=(640-nw)//2;oy=(480-nh)//2
    canvas.paste(Image.fromarray(rgb).crop(box).resize((nw,nh),Image.Resampling.BILINEAR),(ox,oy))
    return np.asarray(canvas).copy(),dict(crop=box,sx=nw/(d-a),sy=nh/(e-c),ox=ox,oy=oy)


def map_box(box,affine,inverse=False):
    if affine is None:return None
    a,c,d,e=affine['crop'];sx=affine['sx'];sy=affine['sy'];ox=affine['ox'];oy=affine['oy']
    x,y,r,b=box
    if inverse:
        result=[max(a,(x-ox)/sx+a),max(c,(y-oy)/sy+c),min(d,(r-ox)/sx+a),min(e,(b-oy)/sy+c)]
    else:
        result=[(max(a,x)-a)*sx+ox,(max(c,y)-c)*sy+oy,(min(d,r)-a)*sx+ox,(min(e,b)-c)*sy+oy]
    return result if result[2]>result[0] and result[3]>result[1] else None


def main(a):
    import numpy as np
    from PIL import Image
    from ultralytics import YOLO
    from train_synthetic_liquid_pilot import validate
    original=validate(a.source);plan=copy.deepcopy(original)
    checkpoint=a.tools/'results/real-synthetic-detector-20261019/models/sim-only/checkpoint.pt'
    metadata=read(checkpoint.with_name('model.json'))
    if sha(checkpoint)!=metadata['checkpoint_sha256'] or metadata['train_runs'] or metadata['development_runs']:
        raise ValueError('Synthetic-only vial initialization differs')
    view=dict(VIEW,vial_checkpoint=str(checkpoint),vial_checkpoint_sha256=sha(checkpoint))
    a.out.mkdir(parents=True,exist_ok=False);(a.out/'helpers').mkdir();(a.out/'pretrained').mkdir()
    shutil.copy2(a.source/'pretrained/fasterrcnn-coco.pth',a.out/'pretrained/fasterrcnn-coco.pth')
    helpers=[Path(__file__),Path(__file__).with_name('vial_basal_policy.py'),Path(__file__).with_name('target_synthetic_transfer.py'),
             Path(__file__).with_name('train_synthetic_liquid_pilot.py'),Path(__file__).with_name('train_early_liquid.py'),
             Path(__file__).with_name('experiment_sim_real_cpu.py'),Path(__file__).with_name('experiment_sim_real_robustness.py')]
    for path in helpers:shutil.copy2(path,a.out/'helpers'/path.name)
    plan['helpers']={p.name:sha(p) for p in helpers};plan['input_view']=view
    plan['kind']='synthetic_only_lower_vial_crop_fasterrcnn_pilot'
    plan['source_plan_sha256']=sha(a.source/'experiment-plan.json')
    plan['samples']=[];model=YOLO(str(checkpoint));truncated=0
    for i,row in enumerate(original['samples']):
        rgb=np.asarray(Image.open(a.source/row['image']).convert('RGB')).copy()
        result=model.predict([rgb[:,:,::-1].copy()],**VIAL_OPTIONS)[0]
        vials=[dict(class_id=int(p[5]),box=p[:4].tolist(),confidence=float(p[4])) for p in result.boxes.data.cpu().numpy() if int(p[5])==0]
        pixels,affine=make_view(rgb,vials,view)
        if affine is None:raise ValueError('Missing synthetic training/development vial: '+row['source_id'])
        boxes=[]
        for line in (a.source/row['label']).read_text().splitlines():
            _,x,y,w,h=map(float,line.split());box=[(x-w/2)*640,(y-h/2)*480,(x+w/2)*640,(y+h/2)*480]
            new=map_box(box,affine)
            if new is None:raise ValueError('Synthetic body entirely outside lower-vial view')
            truncated+=int(box[1]<affine['crop'][1]);boxes.append(new)
        image_rel='data/images/'+row['source_id']+'.png';label_rel='data/labels/'+row['source_id']+'.txt'
        image=a.out/image_rel;label=a.out/label_rel
        image.parent.mkdir(parents=True,exist_ok=True);label.parent.mkdir(parents=True,exist_ok=True)
        Image.fromarray(pixels).save(image)
        label.write_text(''.join('0 '+' '.join(f'{v:.8f}' for v in [(b[0]+b[2])/1280,(b[1]+b[3])/960,(b[2]-b[0])/640,(b[3]-b[1])/480])+'\n' for b in boxes),encoding='utf-8')
        plan['samples'].append(dict(row,image=image_rel,label=label_rel,image_sha256=sha(image),label_sha256=sha(label),
            parent_image_sha256=row['image_sha256'],parent_label_sha256=row['label_sha256'],input_view_affine=affine))
        if (i+1)%100==0:print('PREPARED ROI',i+1,flush=True)
    train_hashes={r['image_sha256'] for r in plan['samples'] if r['split']=='train'}
    if train_hashes & {r['image_sha256'] for r in plan['samples'] if r['split']=='val'}:
        raise ValueError('Cropped image bytes cross the synthetic split')
    plan['target_scope']='Visible accumulated body intersected with lower-vial view; upper compartments excluded. Native geometry, not optical truth.'
    plan['limitations']+=['Input crop is an additional adaptive architecture/preprocessing hypothesis; no real-label fitting or threshold search.',
        'Real agreement is against unchanged full-frame teacher boxes; crop truncation can be penalized when their scope extends above the lower compartment.']
    write(a.out/'experiment-plan.json',plan)
    write(a.out/'preparation.json',dict(plan_sha256=sha(a.out/'experiment-plan.json'),helpers=plan['helpers'],
        source_plan_sha256=sha(a.source/'experiment-plan.json'),cropped_positive_boxes=truncated))
    print('PREPARED',a.out,'truncated synthetic boxes',truncated,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['source','out']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--tools',type=Path,default=Path('C:/Users/Austin/ai-club-tools'))
    main(p.parse_args())
