"""CPU-only independent checks and local media for the SAM-free DINO pilot."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image,ImageDraw,ImageFont


ARMS=['unet','dino_rgb','dino_paired']
LABELS=['Temporal U-Net','DINOv3: current RGB','DINOv3: current + reference']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def font(n):return ImageFont.truetype('/usr/share/fonts/truetype/lato/Lato-Regular.ttf',n)
def decode(encoded):
    mask=np.zeros((480,640),np.uint8)
    if encoded is None:return mask
    x,y=encoded['origin'];w,h=encoded['size'];assert 0<=x<x+w<=640 and 0<=y<y+h<=480
    flat=np.zeros(w*h,np.uint8)
    for start,length in encoded['runs']:
        assert start>=0 and length>0 and start+length<=len(flat);flat[start:start+length]=1
    mask[y:y+h,x:x+w]=flat.reshape(h,w);return mask
def paint(rgb,encoded):
    result=rgb.copy();mask=decode(encoded).astype(bool);result[mask]=(.55*result[mask]+.45*np.array([50,240,180])).astype(np.uint8)
    return Image.fromarray(result)
def movie(path,size,images):
    process=subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{size[0]}x{size[1]}',
        '-r','5','-i','pipe:0','-an','-c:v','libx264','-preset','fast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(path)],stdin=subprocess.PIPE)
    count=0
    try:
        for im in images:process.stdin.write(im.tobytes());count+=1
    finally:process.stdin.close()
    assert process.wait()==0
    result=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0','-show_entries','stream=nb_read_frames,r_frame_rate,width,height','-of','json',str(path)]))['streams'][0]
    assert int(result['nb_read_frames'])==count==240
    subprocess.run(['ffmpeg','-v','error','-xerror','-i',str(path),'-f','null','-'],check=True)
    return dict(frames=count,fps=5,width=size[0],height=size[1],sha256=sha(path))


def main(a):
    a.out.mkdir(parents=True,exist_ok=False)
    completed=read(a.evaluation/'completed.json');verified=0
    for name,digest in completed['artifacts'].items():assert sha(a.evaluation/name)==digest;verified+=1
    plan=read(a.evaluation/'plan.json');report=read(a.evaluation/'comparison.json')
    frozen=read(a.models/'all-models-frozen.json');assert sha(a.models/'all-models-frozen.json')==plan['all_models_frozen_sha256']
    synthetic={}
    for mode in ['rgb','paired']:
        meta=read(a.models/mode/'frozen.json');pred=read(a.models/mode/'development-predictions.json')
        assert meta==frozen['models'][mode] and sha(a.models/mode/'best.pt')==meta['checkpoint_sha256']
        metrics=meta['synthetic_development'];positive=[r for r in pred if r['positive']]
        assert len(pred)==48 and sum(r['presence_correct'] for r in pred)/48==metrics['presence_accuracy']
        assert abs(np.mean([r['geometric_iou'] for r in positive])-metrics['positive_mean_geometric_iou'])<1e-8
        synthetic[mode]=dict(selected_epoch=meta['selected_epoch'],checkpoint_sha256=meta['checkpoint_sha256'],development=metrics)
    passes=[[json.loads(s)['arms'] for s in (a.evaluation/f'pass-{i}.jsonl').read_text().splitlines()] for i in (1,2)]
    assert all(len(p)==264 for p in passes)
    primary='dino_'+plan['primary_dino_mode'];first=passes[0];derived={}
    for row in first:
        rgb=np.asarray(Image.open(a.evaluation/'frames'/f'f{row["unet"]["source_frame"]:06d}.png').convert('RGB'))
        assert hashlib.sha256(rgb.tobytes()).hexdigest()==row['unet']['source_rgb_sha256']
        assert row['unet']['vials']==row['dino_rgb']['vials']==row['dino_paired']['vials']
        for arm in ARMS:
            r=row[arm];ref=next(item['unet'] for item in first if item['unet']['source_frame']==r['reference_frame'])
            assert ref['source_rgb_sha256']==r['reference_rgb_sha256']
            assert r['pts']==row['unet']['pts'] and r['source_rgb_sha256']==row['unet']['source_rgb_sha256']
            assert bool(decode(r['gated_mask']).any())==r['proposal_available']
            assert int(decode(r['raw_mask']).sum())==r['mask_pixels']
    for arm in ARMS:
        records=[row[arm] for p in passes for row in p]
        qa=sum(row[arm]['measurement']['measurement_valid'] for row in first);proposals=sum(row[arm]['proposal_available'] for row in first)
        assert qa==report['arms'][arm]['qa_pass'] and proposals==report['arms'][arm]['proposal_frames']
        for key in ['gated_mask','component_mask','presence_score']:
            differences=sum(x[arm][key]!=y[arm][key] for x,y in zip(*passes));assert differences==report['repeat_differences'][arm][key]
        for stage in ['vial','liquid','qa','total']:
            values=[r['timing_ms'][stage] for r in records];expected=report['arms'][arm]['latency_ms'][stage]
            assert abs(np.median(values)-expected['median'])<1e-8 and abs(np.percentile(values,95)-expected['p95'])<1e-8
        derived[arm]=dict(qa_pass=qa,proposal_frames=proposals,latency_ms=report['arms'][arm]['latency_ms']['total'])
    assert report['baseline_reproduced_qa'] and derived['unet']['qa_pass']==105
    selected=[]
    for row in first:
        r=row['unet'];cycle=next(c for c in plan['cycles'] if c['case_id']==r['case_id'])
        if r['source_frame']<cycle['source_end_frame_exclusive'] and (r['source_frame']-cycle['source_start_frame'])%6==0:selected.append(row)
    assert len(selected)==240
    def panels(single=False):
        for row in selected:
            source=row['unet'];rgb=np.asarray(Image.open(a.evaluation/'frames'/f'f{source["source_frame"]:06d}.png').convert('RGB'))
            arms=[primary] if single else ARMS;canvas=Image.new('RGB',(640*len(arms),784),(19,24,32));draw=ImageDraw.Draw(canvas)
            for column,arm in enumerate(arms):
                r=row[arm];x=column*640;view=paint(rgb,r['gated_mask']);canvas.paste(view,(x,84))
                draw.text((x+14,10),LABELS[ARMS.index(arm)],font=font(26),fill='white')
                draw.text((x+14,46),f'p={r["presence_score"]:.3f} | sampling QA: {"available" if r["measurement"]["measurement_valid"] else "unavailable"}',font=font(20),fill=(190,220,230))
                crop=view.crop((256,320,368,432)).resize((196,196));canvas.paste(crop,(x+12,576))
                draw.text((x+222,580),f'{r["case_id"]} / native f{r["source_frame"]}',font=font(22),fill='white')
                draw.text((x+222,616),'Raw liquid proposal in green',font=font(20),fill=(90,235,180))
                draw.text((x+222,650),'No SAM | no approved real labels',font=font(19),fill=(205,205,210))
                draw.text((x+222,687),'5 fps sampled source playback',font=font(19),fill=(205,205,210))
                draw.text((x+222,725),'Zoom: bottom of fixed station ROI',font=font(18),fill=(205,205,210))
            yield canvas
    videos={}
    videos['sam-free-comparison.mp4']=movie(a.out/'sam-free-comparison.mp4',(1920,784),panels())
    videos['dino-synthetic-selected.mp4']=movie(a.out/'dino-synthetic-selected.mp4',(640,784),panels(True))
    board=Image.new('RGB',(960,1670),(19,24,32));draw=ImageDraw.Draw(board)
    draw.text((14,10),'Same real frames: raw source and SAM-free proposals',font=font(26),fill='white')
    draw.text((14,48),'Six inspected Run4 clips; no approved mask / presence / onset accuracy',font=font(19),fill=(205,205,215))
    for j,title in enumerate(['Source']+LABELS):draw.text((j*240+12,86),title,font=font(17),fill='white')
    crops=[]
    for cycle in plan['cycles']:
        for offset in (0,90):
            target=cycle['provisional_anchor_frame']+offset
            row=next(r for r in first if r['unet']['source_frame']==target);crops.append(row)
    for i,row in enumerate(crops):
        y=124+i*128;rgb=np.asarray(Image.open(a.evaluation/'frames'/f'f{row["unet"]["source_frame"]:06d}.png').convert('RGB'))
        for j,arm in enumerate([None]+ARMS):
            im=Image.fromarray(rgb) if arm is None else paint(rgb,row[arm]['gated_mask'])
            board.paste(im.crop((232,320,392,432)).resize((240,112)),(j*240,y))
            text=f'{row["unet"]["case_id"]} f{row["unet"]["source_frame"]}' if arm is None else f'p {row[arm]["presence_score"]:.3f} / QA {row[arm]["measurement"]["measurement_valid"]}'
            draw.text((j*240+4,y+112),text,font=font(13),fill='white')
    board.save(a.out/'early-later-comparison.png')
    summary=dict(metrics=derived,synthetic=synthetic,verified_remote_artifacts=verified,source_RGB_hash_checks=264,reference_checks=792,
        primary_mode=plan['primary_dino_mode'],primary_selected_only_on_synthetic=True,videos=videos,
        repeat_differences=report['repeat_differences'],versus_unet=report['versus_unet'],real_accuracy=None,
        run5_used=False,local_GPU_used=False,app_defaults_changed=False,SAM_used=False)
    write(a.out/'verification.json',summary)
    write(a.out/'completed.json',dict(artifacts={p.name:sha(p) for p in a.out.iterdir() if p.is_file()}))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['evaluation','models','out']:p.add_argument('--'+name,type=Path,required=True)
    main(p.parse_args())
