"""Single-panel actual cap-state proposals; reuse completed inference, CPU only."""
import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess

from dataset1_fiveclass import read,sha,write
from experiment_dataset1_cap_state import CAP_CLASSES


def export(root,cap,out):
    from PIL import Image,ImageDraw,ImageFont
    out.mkdir(exist_ok=False);frames=out/'frames';frames.mkdir()
    predictions=read(cap/'fitted-v1/test-predictions.json');frozen=read(cap/'fitted-v1/frozen.json')
    rows=[json.loads(s) for s in (root/'rendered-v1/manifest.jsonl').read_text().splitlines()]
    fontfile='/usr/share/fonts/truetype/lato/Lato-Regular.ttf';font=ImageFont.truetype(fontfile,23);small=ImageFont.truetype(fontfile,17)
    colors=['#4de1cf','#ffc247','#ff716c','#a7b9c8'];story=[];index=0
    for row in rows:
        for normal in (True,False):
            scene=row['id']+('-normal' if normal else '');records=[p for p in predictions if p['scene']==scene]
            if len(records)!=60:raise ValueError('Incomplete scene')
            key='golden' if normal else 'image';source=root/'rendered-v1'/row[key]
            if sha(source)!=row[key+'_sha256']:raise ValueError('Original pixels differ')
            image=Image.open(source).convert('RGB');draw=ImageDraw.Draw(image);counts=Counter(p['prediction'] for p in records)
            for p in records:
                l,t,r,b=p['box_xyxy'];x,y=(l+r)/2,(t+b)/2;color=colors[p['prediction']]
                draw.ellipse((x-4,y-4,x+4,y+4),outline=color,width=2)
            canvas=Image.new('RGB',(1280,1160),'#112539');draw=ImageDraw.Draw(canvas)
            draw.text((15,9),'SYNTHETIC CAP INSPECTION | actual frozen-model proposals',font=font,fill='white')
            kind='all-good current image' if normal else row['nuisance']['density']+' defects'
            draw.text((15,43),f'{scene} | {kind} | fixed60 slot targets',font=font,fill='#d7e8ee')
            for i,label in enumerate(('Nominal cap','Missing cap','Crooked cap','Empty')):
                draw.text((15+i*310,80),f'{label}: {counts[i]}',font=font,fill=colors[i])
            canvas.paste(image.resize((1280,960)),(0,120))
            draw.text((15,1090),'Body damage, seal quality and product acceptance remain UNASSESSED.',font=font,fill='#e8d2a4')
            draw.text((15,1127),'Matched synthetic reference | static images held1s | real reference/alignment qualification still required',font=small,fill='#b9cbd5')
            path=frames/f'frame-{index:03d}.png';canvas.save(path)
            story.append(dict(frame=index,scene=scene,source_sha256=row[key+'_sha256'],counts={CAP_CLASSES[i]:counts[i] for i in range(4)},png_sha256=sha(path)));index+=1
    video=out/'cap-state-predictions.mp4'
    subprocess.run(['ffmpeg','-v','error','-framerate','1','-i',str(frames/'frame-%03d.png'),'-vf','fps=24','-c:v','libx264','-preset','fast','-crf','18','-pix_fmt','yuv420p',str(video)],check=True)
    process=subprocess.run(['ffmpeg','-v','error','-i',str(video),'-f','framehash','-'],check=True,stdout=subprocess.PIPE,text=True)
    decoded=sum(bool(line.strip()) and not line.startswith('#') for line in process.stdout.splitlines())
    if decoded!=index*24:raise ValueError('Decoded video count differs')
    write(out/'completed.json',dict(decoded_frames=decoded,source_images=index,fps=24,video_sha256=sha(video),
        checkpoint_sha256=frozen['checkpoint_sha256'],predictions_sha256=sha(cap/'fitted-v1/test-predictions.json'),
        states=story,body_condition='unassessed',real_accuracy=None,helper_sha256=sha(__file__)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','cap','out'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();export(a.root,a.cap,a.out)
