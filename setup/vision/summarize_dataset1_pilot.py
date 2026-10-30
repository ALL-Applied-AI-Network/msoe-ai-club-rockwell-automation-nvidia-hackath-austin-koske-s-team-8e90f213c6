"""CPU-only verification, plots and source-exact synthetic fixture demo."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess

from dataset1_pilot import CLASSES, read, sha, write
from train_dataset1_pilot import summarize, validate_records


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();root=args.root;out=args.out
    out.mkdir(exist_ok=False)
    fitted=root/'fitted';rendered=root/'rendered'
    completion=read(fitted/'completed.json')
    if completion['comparison_sha256']!=sha(fitted/'comparison.json'):
        raise ValueError('Comparison report differs')
    rows=[json.loads(l) for l in (rendered/'manifest.jsonl').read_text().splitlines()]
    counts=validate_records(rows)
    test=[r for r in rows if r['split']=='test']
    comparison=read(fitted/'comparison.json')
    predictions={}
    for mode in ('rgb','paired_reference'):
        arm=fitted/mode;frozen=read(arm/'frozen.json')
        if sha(arm/'best.pt')!=frozen['checkpoint_sha256']:
            raise ValueError('Checkpoint differs')
        records=read(arm/'test-predictions.json')
        expected=summarize(records,frozen['threshold'])
        for key,value in expected.items():
            if value!=comparison['arms'][mode][key]:
                raise ValueError('Recomputed metric differs: '+key)
        source={r['id']:r for r in test}
        assert len(records)==360 and len({(r['scene'],r['slot']) for r in records})==360
        for r in records:
            original=source[r['scene']]
            assert r['source_sha256']==original['image_sha256']
            assert r['box']==original['boxes'][r['slot']]
            assert r['truth']==CLASSES.index(original['labels'][r['slot']])
        predictions[mode]={(r['scene'],r['slot']):r for r in records}
    for row in rows:
        for key in ('image','golden'):
            assert sha(rendered/row[key])==row[key+'_sha256']
    plt.rcParams.update({'font.size':11})
    fig,axes=plt.subplots(1,2,figsize=(11,4.7),layout='constrained')
    for axis,mode in zip(axes,('rgb','paired_reference')):
        matrix=np.array(comparison['arms'][mode]['confusion_rows_truth_columns_prediction'])
        axis.imshow(matrix,cmap='Blues',vmin=0,vmax=120)
        axis.set(xticks=range(3),xticklabels=CLASSES,yticks=range(3),yticklabels=CLASSES,
                 xlabel='Predicted synthetic class',ylabel='Rendered geometry class',title=mode.replace('_',' '))
        for i in range(3):
            for j in range(3):
                axis.text(j,i,str(matrix[i,j]),ha='center',va='center',color='white' if matrix[i,j]>70 else '#142B40')
    fig.suptitle('Dataset 1 pilot / 12 held-out synthetic scenes / 360 targets\nSame CAD and positions; real accuracy unmeasured',fontsize=14)
    fig.savefig(out/'synthetic-test-confusion.png',dpi=170);plt.close(fig)
    fontpath='/usr/share/fonts/truetype/lato/Lato-Bold.ttf'
    font=ImageFont.truetype(fontpath,22);small=ImageFont.truetype(fontpath,16)
    colors={'pass':'#32BD87','missing_cap':'#FFAE42','no_vial':'#94B1CA'}
    frames=out/'demo-frames';frames.mkdir()
    observations=[]
    for index,row in enumerate(test):
        src=Image.open(rendered/row['image']).convert('RGB')
        board=Image.new('RGB',(1280,1120),'#142B40');board.paste(src.resize((1280,960)),(0,80))
        draw=ImageDraw.Draw(board)
        draw.text((22,15),'DATASET 1 / SYNTHETIC TEST / '+row['id'],font=font,fill='white')
        draw.text((22,45),'Paired-reference CNN / actual frozen predictions / independent scenes, not a live video',font=small,fill='white')
        scene_predictions=[]
        for slot in sorted(row['labels']):
            r=predictions['paired_reference'][(row['id'],slot)]
            label=CLASSES[r['prediction']];scene_predictions.append(label)
            x=(r['box'][0]+r['box'][2]);y=(r['box'][1]+r['box'][3])+80
            draw.ellipse((x-10,y-10,x+10,y+10),outline=colors[label],width=3)
            draw.text((x-9,y-27),slot,font=small,fill=colors[label])
            observations.append(dict(schema='experimental-station-observation/v1',data_domain='synthetic',station='final_fixture',
                fixture='G',target_slot=slot,source=dict(scene=row['id'],rgb_sha256=row['image_sha256'],
                    dimensions=[640,480],native_pts=None,native_time_base=None),box_xyxy=r['box'],
                crop_space='original source pixels; authored slot identity, not confirmed real mapping',
                model_sha256=read(fitted/'paired_reference/frozen.json')['checkpoint_sha256'],
                proposed_class=label,class_scores=dict(zip(CLASSES,r['probabilities'])),scores_calibrated=False,
                unavailable_classes=['crooked_cap','damaged_vial'],expected_occupied=None,
                placement_assessment='unavailable: no attributed placement/order context',
                physical_vial_association='unlinked',human_quality_approval=None,equipment_command=None))
        stats=Counter(scene_predictions)
        draw.text((22,1053),'Capped: '+str(stats['pass'])+'    Uncapped: '+str(stats['missing_cap'])+'    Empty: '+str(stats['no_vial']),font=font,fill='white')
        draw.text((22,1085),'Occupancy is separate from placement error. Real transfer and cross-station identity remain unmeasured.',font=small,fill='white')
        board.save(frames/f'{index:03d}.png')
        if index==0:board.save(out/'fixture-predictions.png')
    command=['ffmpeg','-v','error','-n','-framerate','1','-i',str(frames/'%03d.png'),'-c:v','libx264','-pix_fmt','yuv420p',str(out/'dataset1-synthetic-fixture-demo.mp4')]
    subprocess.run(command,check=True)
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0','-show_entries','stream=nb_read_frames,r_frame_rate','-of','json',str(out/'dataset1-synthetic-fixture-demo.mp4')]))
    assert probe['streams'][0]['nb_read_frames']=='12'
    subprocess.run(['ffmpeg','-v','error','-i',str(out/'dataset1-synthetic-fixture-demo.mp4'),'-f','null','-'],check=True)
    write(out/'sample-observations.json',observations[:30])
    write(out/'comparison.json',comparison)
    lines=['# Dataset 1 pilot results','','All figures below are synthetic. No real Dataset 1 images were fitted or evaluated.','',
           '| Model | Selected epoch | Test correct /360 | Missing cap correct /120 | Empty correct /120 | All30-correct scenes /12 |',
           '| --- | ---: | ---: | ---: | ---: | ---: |']
    for mode,metrics in comparison['arms'].items():
        frozen=read(fitted/mode/'frozen.json')
        correct=sum(metrics['confusion_rows_truth_columns_prediction'][i][i] for i in range(3))
        lines.append(f'| {mode} | {frozen["selected_epoch"]} | {correct} | {metrics["per_class"]["missing_cap"]["correct"]} | {metrics["per_class"]["no_vial"]["correct"]} | {metrics["all_30_correct_scenes"]} |')
    lines += ['', 'The demo shows 12 independent rendered test scenes at one scene per second; it is not a physical placement sequence.',
        'Both models freeze on synthetic development before test. The test uses the same CAD and fixed30 positions.',
        'A matching synthetic golden does not establish a reliable real reference procedure.',
        'No-vial is an occupancy class. Missing-placement assessment needs attributed progress/completion context.',
        'Crooked caps, damaged bodies, real transfer and physical vial-to-slot association remain outside this pilot.',
        'See sample-observations.json for the unlinked experimental station contract; no operator app was changed.']
    (out/'README.md').write_text('\n'.join(lines)+'\n')
    write(out/'completed.json',dict(split_counts=counts,verified_render_images=120,verified_test_predictions=720,
        video_decoded_frames=12,video_scene_rate='1 independent synthetic scene/s',human_real_accuracy_available=False,
        files={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file()}))
    print(json.dumps(dict(out=str(out),comparison=comparison)),flush=True)


if __name__=='__main__':
    main()
