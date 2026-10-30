"""CPU source checks and unreviewed real-camera prediction media; no truth fitting."""
import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess

from dataset1_real_transfer import read,sha,validate_source_rows,write


def main():
    from PIL import Image,ImageDraw,ImageFont
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--batch',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();batch=a.batch;out=a.out;out.mkdir(exist_ok=False)
    samples=read(batch/'samples.json');validate_source_rows(samples)
    completion=read(batch/'inference/completed.json');plan=read(batch/'plan.json');registration=read(batch/'registration.json')
    if sha(batch/'inference/predictions.json')!=completion['predictions_sha256'] or completion['checkpoint_sha256']!=plan['checkpoint_sha256']:
        raise ValueError('Inference evidence differs')
    records=read(batch/'inference/predictions.json')
    assert len(records)==480
    by_target={(r['sample_id'],r['slot']):r for r in records};assert len(by_target)==480
    colors={'pass':'#32BD87','missing_cap':'#FFAE42','no_vial':'#94B1CA'}
    font=ImageFont.truetype('/usr/share/fonts/truetype/lato/Lato-Bold.ttf',22)
    small=ImageFont.truetype('/usr/share/fonts/truetype/lato/Lato-Bold.ttf',16)
    frames=out/'frames';frames.mkdir();by_run={};observations=[]
    for i,sample in enumerate(samples):
        if sha(batch/sample['image'])!=sample['source_sha256']:
            raise ValueError('Real source changed')
        image=Image.open(batch/sample['image']).convert('RGB');assert image.size==(640,480)
        board=Image.new('RGB',(1280,1120),'#142B40');board.paste(image.resize((1280,960)),(0,80));draw=ImageDraw.Draw(board)
        draw.text((20,12),f'REAL CAMERA / RUN {sample["run_id"]} / FROZEN SYNTHETIC RGB MODEL / UNREVIEWED',font=font,fill='white')
        draw.text((20,45),'Sampled stills, not live playback / G targets only / mapping and occlusion require human review',font=small,fill='white')
        counts=Counter()
        for slot,box in sorted(sample['boxes'].items()):
            r=by_target[(sample['sample_id'],slot)]
            assert r['source_sha256']==sample['source_sha256'] and r['box_xyxy']==box and r['truth'] is None
            assert set(r['class_scores'])==set(colors) and abs(sum(r['class_scores'].values())-1)<1e-5
            assert r['proposed_class']==max(r['class_scores'],key=r['class_scores'].get)
            x=box[0]+box[2];y=box[1]+box[3]+80;c=colors[r['proposed_class']]
            draw.ellipse((x-9,y-9,x+9,y+9),outline=c,width=3);draw.text((x-9,y-26),slot,fill=c,font=small)
            counts[r['proposed_class']]+=1
            observations.append(dict(schema='experimental-station-observation/v1',data_domain='real',source=sample,
                target_slot=slot,box_xyxy=box,model_sha256=plan['checkpoint_sha256'],proposed_class=r['proposed_class'],
                class_scores=r['class_scores'],scores_calibrated=False,mapping_approved=False,visibility_approved=False,
                unsupported_classes=['crooked_cap','damaged_vial'],expected_occupied=None,placement_assessment=None,
                human_quality_approval=None,physical_vial_association='unlinked',equipment_command=None))
        by_run.setdefault(str(sample['run_id']),Counter()).update(counts)
        draw.text((20,1052),f'Proposals: capped {counts["pass"]} / uncapped {counts["missing_cap"]} / empty {counts["no_vial"]}. No quality or placement decision.',font=font,fill='white')
        draw.text((20,1085),sample['capture_timestamp']+' / still-to-video frame association unmatched / S fixture not assessed',font=small,fill='white')
        board.save(frames/f'{i:03d}.png')
        if i==len(samples)-1:board.save(out/'real-fixture-predictions.png')
    video=out/'real-fixture-predictions.mp4'
    subprocess.run(['ffmpeg','-v','error','-n','-framerate','1','-i',str(frames/'%03d.png'),'-c:v','libx264','-pix_fmt','yuv420p',str(video)],check=True)
    stream=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0','-show_entries','stream=nb_read_frames,r_frame_rate','-of','json',str(video)]))
    assert stream['streams'][0]['nb_read_frames']=='16'
    subprocess.run(['ffmpeg','-v','error','-i',str(video),'-f','null','-'],check=True)
    report=dict(source_stills=16,slot_frame_predictions=480,per_run_prediction_counts=by_run,
        prediction_counts=Counter(r['proposed_class'] for r in records),model_sha256=plan['checkpoint_sha256'],
        geometric_registration=dict(anchor='first indexed Run4 cap_seal still',rmse_pixels=registration['rmse_pixels'],max_error_pixels=registration['max_error_pixels'],approved=False),
        real_accuracy=None,reviewed_labels=0,paired_reference_status=plan['paired_reference_status'],
        physical_vial_association=False,training=False,run5_used=False,video_decoded_frames=16,
        limitations=['Counts are provisional repeated slot-frame predictions, not unique vials or correctness.',
            'No reviewed defect labels or qualified occlusion/mapping. No missing-cap recall can be claimed from zero predictions.',
            'Only capped/uncapped/empty G targets are modeled; tilted caps/damaged bodies and S fixture are unsupported.',
            'Output MP4 is a one-still-per-second visualization, not original timing or a live feed.'])
    write(out/'report.json',report);write(out/'sample-observations.json',observations[-30:])
    (out/'README.md').write_text(
        '# Frozen synthetic model on real final-fixture images\n\n'
        '16 real cap_seal stills from Runs1–4, four deterministic image-index samples per run;480 attributed G slot-frame predictions.\n'
        'The frozen epoch5 RGB model predicts41 capped-looking and439 empty slots, with zero uncapped predictions.\n'
        'These are repeated slot-frame proposals, not unique vials, labels, or accuracy. Missing-cap recall is unavailable.\n\n'
        'The geometric Run4 anchor fit has0.659px RMS/1.414px maximum aperture reprojection error.\n'
        'The mapping is proposed, not approved. Other runs were visually checked but are not independently calibrated.\n'
        'Robot occlusion, unsupported crooked/damaged classes and ambiguous targets require human review.\n'
        'The paired-reference arm is unavailable without a qualified capped real golden; no substitute was invented.\n\n'
        'real-fixture-predictions.mp4 shows16 sampled stills at one per second, not live/native video playback.\n'
        'Open ../review-v1/review.html for raw-first review; start with96 priority targets or inspect all480.\n'
        'Save explicit mapping/visibility/class decisions under your own name and export JSON. No reviewed truth is supplied.\n'
        'All real data is evaluation-only; no fitting, checkpoint selection, threshold tuning or Run5 use.\n',encoding='utf-8')
    write(out/'completed.json',dict(report_sha256=sha(out/'report.json'),predictions_sha256=completion['predictions_sha256'],
        source_hashes_verified=16,source_target_records_verified=480,video_decoded_frames=16,
        files={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()}))
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
