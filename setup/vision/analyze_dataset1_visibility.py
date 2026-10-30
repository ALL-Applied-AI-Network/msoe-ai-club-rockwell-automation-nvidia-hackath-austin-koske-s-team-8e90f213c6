"""Compare isolated geometry effects with repeated-render noise; CPU only."""
import argparse
import json
from pathlib import Path

from dataset1_fiveclass import read,sha,write


def analyze(root,out):
    import numpy as np
    from PIL import Image,ImageDraw,ImageFont
    out.mkdir(exist_ok=False);render=root/'rendered-v1'
    rows=[json.loads(s) for s in (render/'manifest.jsonl').read_text().splitlines()]
    font=ImageFont.truetype('/usr/share/fonts/truetype/lato/Lato-Regular.ttf',16)
    small=ImageFont.truetype('/usr/share/fonts/truetype/lato/Lato-Regular.ttf',13)
    board=Image.new('RGB',(1120,72+len(rows)*180),'#132538');draw=ImageDraw.Draw(board)
    draw.text((14,12),'Isolated geometry changes | original golden / repeated golden / changed / absolute difference x8',font=font,fill='white')
    draw.text((14,38),'Only one target changes; rings added after rendering. Pixel change measures visibility, not physical defect accuracy.',font=small,fill='#c5d5df')
    records=[]
    for i,row in enumerate(rows):
        arrays={}
        for key in ('golden','repeat','image'):
            if sha(render/row[key])!=row[key+'_sha256']:raise ValueError('Rendered source differs')
            arrays[key]=np.array(Image.open(render/row[key]).convert('RGB'),dtype=np.float32)
        case=row['nuisance']['isolated_case'];slot=case['slot'];box=row['boxes'][slot];l,t,r,b=box
        golden=arrays['golden'][t:b,l:r];repeat=arrays['repeat'][t:b,l:r];changed=arrays['image'][t:b,l:r]
        delta=np.abs(changed-golden);noise=np.abs(repeat-golden)
        metrics={}
        for size in (32,64,96):
            a=(96-size)//2;d=delta[a:a+size,a:a+size];n=noise[a:a+size,a:a+size]
            threshold=max(2.,float(np.percentile(n,99.9)))
            metrics[str(size)]=dict(change_mae=float(d.mean()),repeat_noise_mae=float(n.mean()),
                mae_ratio=float(d.mean()/max(float(n.mean()),1e-8)),noise_channel_threshold=threshold,
                changed_pixels_above_noise=int((d.max(2)>threshold).sum()),repeat_pixels_above_noise=int((n.max(2)>threshold).sum()),
                pixels=size*size)
        y=72+i*180
        for col,array in enumerate((golden,repeat,changed,np.clip(delta*8,0,255))):
            tile=Image.fromarray(array.astype('uint8')).resize((160,160));d=ImageDraw.Draw(tile)
            d.ellipse((74,74,86,86),outline='#52edcc',width=1);board.paste(tile,(col*168,y))
        draw.text((680,y+10),f'{slot} {case["geometric_class"]}',font=font,fill='white')
        draw.text((680,y+38),f'lean {case["lean_deg"]} deg / {case["damage_kind"]} proxy',font=small,fill='#c5d5df')
        draw.text((680,y+65),f'center64 change MAE {metrics["64"]["change_mae"]:.3f}',font=small,fill='white')
        draw.text((680,y+88),f'repeat noise MAE {metrics["64"]["repeat_noise_mae"]:.3f}',font=small,fill='white')
        draw.text((680,y+111),f'above noise: {metrics["64"]["changed_pixels_above_noise"]}/4096 pixels',font=small,fill='white')
        records.append(dict(scene=row['id'],case=case,box_xyxy=box,metrics=metrics,source_sha256=row['image_sha256'],golden_sha256=row['golden_sha256'],repeat_sha256=row['repeat_sha256']))
    board.save(out/'isolated-defect-visibility.png')
    write(out/'report.json',dict(schema='dataset1-visibility/v1',cases=records,real_accuracy=None,training=False,
        interpretation='Counterfactual RGB signal versus repeated-render variation; threshold has a two-digital-level floor. Not a perceptual labeling standard or evidence that all damage is detectable.',
        physical_sensor_noise_modeled=False,unchanged_scene_render_may_reuse_accumulation=True,
        renderer_sha256=read(root/'experiment-plan.json')['renderer_sha256'],manifest_sha256=sha(render/'manifest.jsonl')))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();analyze(a.root,a.out)
