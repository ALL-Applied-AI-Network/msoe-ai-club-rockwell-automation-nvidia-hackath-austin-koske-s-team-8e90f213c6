"""CPU-only source/metric verification and local media for the completed extension."""
import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess

from dataset1_fiveclass import CLASSES,SLOTS,read,sha,write
from train_dataset1_fiveclass import summarize


def summarize_run(root,out):
    from PIL import Image,ImageDraw,ImageFont
    out.mkdir(exist_ok=False)
    manifest=read(root/'artifact-manifest.json')
    for relative,digest in manifest.items():
        if sha(root/relative)!=digest:raise ValueError('Completed artifact differs: '+relative)
    rows=[json.loads(s) for s in (root/'rendered-v1/manifest.jsonl').read_text().splitlines()];by_id={r['id']:r for r in rows}
    completion=read(root/'fitted-v1/completed.json');comparison=read(root/'fitted-v1/comparison.json');models=completion['models'];verified=0
    for mode,meta in models.items():
        arm=root/'fitted-v1'/mode
        if sha(arm/'best.pt')!=meta['checkpoint_sha256']:raise ValueError('Selected model differs')
        predictions=read(arm/'test-predictions.json')
        for record in predictions:
            source=by_id[record['scene']]
            if (source['split']!='test' or record['source_sha256']!=source['image_sha256'] or record['box_xyxy']!=source['boxes'][record['slot']]
                or record['truth']!=CLASSES.index(source['labels'][record['slot']]) or record['prediction']!=max(range(5),key=lambda i:record['probabilities'][i])):
                raise ValueError('Test source/slot/label/prediction join differs')
            verified+=1
        recalculated=summarize(predictions,meta['threshold'])
        for key,value in recalculated.items():
            if comparison['arms'][mode][key]!=value:raise ValueError('Reported metric differs: '+key)
        val=read(arm/'development-predictions.json');threshold=min(1-r['probabilities'][0] for r in val if r['truth'] in (1,2,3))
        if threshold!=meta['threshold']:raise ValueError('Threshold differs from frozen development rule')
        curve=read(arm/'curve.json');best=max(r['validation_macro_recall'] for r in curve)
        if meta['selected_epoch']!=next(r['epoch'] for r in curve if r['validation_macro_recall']==best):raise ValueError('Checkpoint selection differs')
    real=read(root/'real-batch-v1/inference/completed.json');placement=read(root/'placement-inference-v1/completed.json')
    selected=real['selected_review_model'];predictions=read(root/'fitted-v1'/selected/'test-predictions.json')
    prediction_by_slot={(r['scene'],r['slot']):r for r in predictions}
    font_file='/usr/share/fonts/truetype/lato/Lato-Regular.ttf';bold_file='/usr/share/fonts/truetype/lato/Lato-Bold.ttf'
    font=ImageFont.truetype(font_file,17);small=ImageFont.truetype(font_file,13);title=ImageFont.truetype(bold_file,24)
    example=next(r for r in rows if r['split']=='test');source=Image.open(root/'rendered-v1'/example['image']).convert('RGB')
    board=Image.new('RGB',(1200,650),'#172b40');draw=ImageDraw.Draw(board)
    draw.text((12,8),'Five synthetic classes | fixed first held-out scene | target ring added after inference',font=font,fill='white')
    displayed=[]
    for col,label in enumerate(CLASSES):
        draw.text((col*240+8,38),label,font=font,fill='white')
        for row,prefix in enumerate(('G','S')):
            slot=next(s for s in SLOTS if s.startswith(prefix) and example['labels'][s]==label)
            crop=source.crop(example['boxes'][slot]).resize((240,240));d=ImageDraw.Draw(crop);d.ellipse((108,108,132,132),outline='#31e8ac',width=2)
            x,y=col*240,70+row*290;board.paste(crop,(x,y));pred=prediction_by_slot[(example['id'],slot)]
            draw.text((x+6,y+245),f'{slot}: model {CLASSES[pred["prediction"]]}',font=small,fill='white')
            displayed.append(dict(scene=example['id'],slot=slot,source_sha256=example['image_sha256'],truth=label,prediction=CLASSES[pred['prediction']]))
    board.save(out/'five-class-heldout-examples.png');write(out/'example-source-identities.json',displayed)
    # Actual predictions on six discrete rendered states; video does not imply motion.
    inference=read(root/'placement-inference-v1/assessments.json');scene_rows={r['id']:r for r in [json.loads(s) for s in (root/'placement-scenes-v1/rendered-v1/manifest.jsonl').read_text().splitlines()]}
    rawpred=read(root/'placement-inference-v1/predictions.json');frames=out/'placement-frames';frames.mkdir();story=[]
    for index,assessment in enumerate(inference):
        row=scene_rows[assessment['scene']];path=root/'placement-scenes-v1/rendered-v1'/row['image']
        if sha(path)!=row['image_sha256']:raise ValueError('Placement source differs')
        image=Image.open(path).convert('RGB');d=ImageDraw.Draw(image);warnings=[]
        for fixture,result in assessment['assessments'].items():
            for target in result['targets']:
                slot=target['slot'];box=row['boxes'][slot];x,y=(box[0]+box[2])/2,(box[1]+box[3])/2
                color='#ff7474' if target['review_requested'] else '#48d6d0' if target['proposed_class']!='no_vial' else '#afb7bf'
                d.ellipse((x-3,y-3,x+3,y+3),outline=color,width=2)
                if target['review_requested']:
                    d.text((x-9,y-15),slot,fill=color,font=small);warnings.append(slot+': '+target['finding'])
        canvas=Image.new('RGB',(1280,1220),'#101c2c');d=ImageDraw.Draw(canvas)
        d.text((18,12),'SYNTHETIC FINAL-FIXTURE INSPECTION | model predictions',font=title,fill='white')
        d.text((18,50),f"State {index+1}/6: {assessment['scene']} | expected G={row['nuisance']['placement_count']['G']}, S={row['nuisance']['placement_count']['S']}",font=font,fill='white')
        d.text((18,80),f"Review requests: {len(warnings)} | gray: empty proposal | cyan: occupied proposal | red: review",font=font,fill='#e7c27b')
        canvas.paste(image.resize((1280,960)),(0,120))
        for line,text in enumerate(warnings[:3]):d.text((18,1095+line*25),text,font=font,fill='#ffb3b3')
        d.text((18,1190),'Discrete rendered states held for viewing; no robot-motion, live-latency or real-accuracy claim.',font=font,fill='#a9bacb')
        output=frames/f'frame-{index:02d}.png';canvas.save(output)
        story.append(dict(scene=row['id'],source_sha256=row['image_sha256'],output_png_sha256=sha(output),review_requests=len(warnings),
            exact_class_correct=assessment['predictions_correct'],targets=60))
    video=out/'synthetic-placement-inspection.mp4'
    subprocess.run(['ffmpeg','-v','error','-framerate','1/2','-i',str(frames/'frame-%02d.png'),'-vf','fps=24','-c:v','libx264','-preset','fast','-crf','18','-pix_fmt','yuv420p',str(video)],check=True)
    # Decode every output frame, rather than trusting the container frame count.
    process=subprocess.run(['ffmpeg','-v','error','-i',str(video),'-f','framehash','-'],check=True,stdout=subprocess.PIPE,text=True)
    decoded=sum(bool(line.strip()) and not line.startswith('#') for line in process.stdout.splitlines())
    if decoded!=288:raise ValueError('Unexpected decoded video length')
    write(out/'placement-storyboard.json',dict(states=story,decoded_video_frames=decoded,output_fps=24,source_frames=6,state_hold_seconds=2,source_is_native_video=False))
    report=dict(schema='dataset1-fiveclass-summary/v1',verified_input_artifacts=len(manifest),verified_test_predictions=verified,
        groups=read(root/'experiment-plan.json')['groups'],classes=CLASSES,fixtures=['G','S'],models=models,synthetic_test=comparison,
        selected_review_model=selected,real_probe=real,placement_rehearsal=placement,real_accuracy=None,run5_used=False,
        training_real_runs=[],app_defaults_changed=False,hardware_commands=0,
        limits=['Chips/dents are geometric defect proxies, not validated coverage of real body damage.',
            'Model classes do not approve real target mapping/visibility; robot/structure occlusion needs review.',
            'Rendered placement states are not physical robot-motion or live feed validation.',
            'Real golden unavailable; reference-paired model tested on synthetic data only.',
            'Existing16 real stills are development evidence; no reviewed real defect accuracy is available.'])
    write(out/'report.json',report)
    lines=['# Five-class, two-fixture experiment results','',f'Source/metric verification passed for {verified} held-out synthetic predictions across three frozen models.','',
        '| Model | Selected epoch | Synthetic test accuracy | All60-correct scenes | Occupied defect false alarms |',
        '| --- | ---: | ---: | ---: | ---: |']
    for mode,metrics in comparison['arms'].items():
        diagnostic=metrics['occupied_defect_diagnostic'];lines.append(f'| {mode} | {models[mode]["selected_epoch"]} | {metrics["accuracy"]:.2%} | {metrics["all_60_correct_scenes"]}/{metrics["complete_scenes"]} | {diagnostic["false_alarms"]}/{diagnostic["acceptable_targets"]} |')
    lines+=['','Use report.json for per-class misses, confusion, thresholds and timing exclusions.',
        f'Real probe:960 target proposals per model on16 previously inspected stills; review arm {selected} selected using synthetic development only.',
        'Real accuracy is unavailable. New predictions are not labels; both fixture maps/visibility remain unreviewed.',
        'The MP4 contains actual frozen-model predictions on six rendered placement states, held two seconds per state. It is not native/live camera video.',
        'Green rings in the class contact sheet identify the target and were added after inference.',
        'See the independent review packet and commissioning notes before any physical test.']
    (out/'README.md').write_text('\n'.join(lines)+'\n')
    write(out/'completed.json',dict(helper_sha256=sha(__file__),source_manifest_sha256=sha(root/'artifact-manifest.json'),
        files={p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file()},real_accuracy=None))
    print(json.dumps(dict(out=str(out),verified_test_predictions=verified,decoded_video_frames=decoded,selected_model=selected)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();summarize_run(a.root,a.out)
