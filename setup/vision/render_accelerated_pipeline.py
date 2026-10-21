"""Single-view local application video from the measured full TensorRT pipeline."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import textwrap
import av
import cv2
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from liquid_specialists_data import read,write,sha
from render_liquid_pipeline import pixel_hash
from replay.geometry import decode_mask


def text(canvas,value,x,y,size=.55,color=(225,233,240)):
    cv2.putText(canvas,value,(x,y),cv2.FONT_HERSHEY_SIMPLEX,size,color,1,cv2.LINE_AA)


def compose(rgb,row):
    bgr=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
    for encoded in row['raw_masks']:
        mask=decode_mask(encoded).astype(bool)
        bgr[mask]=(.65*bgr[mask]+.35*np.array([70,220,90])).astype(np.uint8)
    sampled=decode_mask(row['measurement']['geometry']['mask']).astype(np.uint8)
    contours,_=cv2.findContours(sampled,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(bgr,contours,-1,(245,230,55),1)
    for prediction in row['predictions']:
        if prediction['class_id']==0:
            x,y,r,b=[int(v) for v in prediction['box']]
            cv2.rectangle(bgr,(x,y),(r,b),(230,165,55),1)
    canvas=np.full((720,1280,3),(35,27,20),np.uint8)
    text(canvas,'LIQUID INSPECTION / ACCELERATED PIPELINE',24,42,.83)
    text(canvas,f'RECORDED RUN 4  |  Source time {row["seconds"]:.1f}s  |  5 Hz analyzed samples',24,78,.56)
    canvas[125:605,20:660]=bgr
    zoom=cv2.resize(bgr[220:450,250:365],(180,360),interpolation=cv2.INTER_LINEAR)
    canvas[160:520,700:880]=zoom
    text(canvas,'Liquid close-up',700,143,.5)
    m=row['measurement'];state=row['fill_detection'];g=m['geometry']
    valid=m['measurement_valid']
    color=(100,220,130) if valid else (160,180,205)
    lines=[('SAMPLE QA: '+('PASS' if valid else 'UNAVAILABLE'),color),
           ('Image phase: '+state['phase'].replace('_',' '),(225,233,240)),
           ('Visual track: '+str(g['track_id']),(225,233,240)),
           ('Sample pixels: '+str(g['mask_pixels']),(225,233,240)),
           ('Camera reference: '+('stable' if row['camera_feature'].get('healthy') else 'unavailable'),(225,233,240)),
           ('Recipe match: UNAVAILABLE',(130,200,245)),
           ('Intended recipe / acceptable reference missing.',(180,195,215)),
           ('Hardware command: NONE',(225,233,240)),
           (f'Recorded processing: {row["timing_ms"]["decode_plus_processing"]:.1f} ms',(180,195,215))]
    y=148
    for line,c in lines:
        for piece in textwrap.wrap(line,33):
            text(canvas,piece,925,y,.5,c);y+=24
        y+=10
    text(canvas,'Blue: vial proposal. Green: proposed liquid. Cyan: pixels used for color.',24,647,.58)
    text(canvas,'Geometry awaits human review. Sample QA does not establish recipe correctness or approve product.',24,684,.55,(130,200,245))
    return canvas


def main(a):
    if a.out.exists():raise ValueError('Use a fresh media directory')
    done=read(a.input/'completed.json');plan=read(a.input/'plan.json')
    if sha(a.source)!=plan['source_sha256']:raise ValueError('Wrong source')
    marker=read(a.input/'tensorrt-pass0/completed.json')
    if sha(a.input/'tensorrt-pass0/completed.json')!=done['runs']['tensorrt-pass0']:
        raise ValueError('TensorRT evidence changed')
    path=a.input/'tensorrt-pass0/frames.jsonl'
    if sha(path)!=marker['artifacts']['frames.jsonl']:raise ValueError('Frame evidence changed')
    rows={r['source_frame']:r for r in map(json.loads,path.read_text().splitlines())}
    request=read(a.request)
    a.out.mkdir(parents=True)
    shutil.copy2(__file__,a.out/Path(__file__).name)
    outputs=[('accelerated-pipeline-run4-60s.mp4',[(0,1800)]),
             ('accelerated-pipeline-six-early-fills.mp4',[(c['provisional_anchor_frame']-60,c['provisional_anchor_frame']+180) for c in request['cycles']])]
    reports=[]
    for name,windows in outputs:
        writer=cv2.VideoWriter(str(a.out/name),cv2.VideoWriter_fourcc(*'mp4v'),5.,(1280,720))
        if not writer.isOpened():raise RuntimeError('Cannot encode')
        trace=[]
        try:
            for start,end in windows:
                with av.open(str(a.source)) as c:
                    for index,frame in enumerate(c.decode(video=0)):
                        if index>=end:break
                        if index<start or index%6:continue
                        row=rows[index];rgb=frame.to_ndarray(format='rgb24')
                        if pixel_hash(rgb)!=row['source_rgb_sha256'] or frame.pts!=row['pts'] or str(frame.time_base)!=row['time_base']:
                            raise ValueError('Source sample or PTS differs')
                        canvas=compose(rgb,row);writer.write(canvas)
                        if index in (390,432,720,1740,2094):
                            cv2.imwrite(str(a.out/f'pipeline-frame{index:06d}.png'),canvas)
                        trace.append(dict(output_frame=len(trace),source_frame=index,pts=frame.pts,
                            time_base=str(frame.time_base),source_rgb_sha256=row['source_rgb_sha256']))
        finally:writer.release()
        with av.open(str(a.out/name)) as c:
            decoded=sum(1 for f in c.decode(video=0) if (f.width,f.height)==(1280,720))
        expected=sum((end-start)//6 for start,end in windows)
        if decoded!=expected or decoded!=len(trace):raise ValueError('Incomplete video decode')
        write(a.out/(Path(name).stem+'-source-trace.json'),trace)
        reports.append(dict(path=name,decoded_frames=decoded,seconds=decoded/5,sha256=sha(a.out/name)))
    comparison=read(a.comparison/'comparison.json')
    original_path=a.input/'pytorch-pass0/frames.jsonl'
    original_marker=read(a.input/'pytorch-pass0/completed.json')
    if sha(original_path)!=original_marker['artifacts']['frames.jsonl']:
        raise ValueError('PyTorch evidence changed')
    original_rows={r['source_frame']:r for r in map(json.loads,original_path.read_text().splitlines())}
    disagreements=comparison['backend_agreement']['pass0']['qa_disagreements']
    selected={r['source_frame'] for r in disagreements}
    if selected:
        with av.open(str(a.source)) as c:
            for index,frame in enumerate(c.decode(video=0)):
                if index not in selected:continue
                rgb=frame.to_ndarray(format='rgb24')
                if pixel_hash(rgb)!=rows[index]['source_rgb_sha256']:raise ValueError('Diagnostic source differs')
                left,right=compose(rgb,original_rows[index]),compose(rgb,rows[index])
                text(left,'PYTORCH / SAME SOURCE',24,108,.65)
                text(right,'TENSORRT FP16 / SAME SOURCE',24,108,.65)
                cv2.imwrite(str(a.out/f'qa-disagreement-frame{index:06d}.png'),np.concatenate([left,right],axis=1))
                selected.remove(index)
                if not selected:break
    write(a.out/'completed.json',dict(input_completion_sha256=sha(a.input/'completed.json'),videos=reports,
        run5_used=False,hardware_actions=0,cosmos_used=False,
        artifacts={p.name:sha(p) for p in a.out.iterdir() if p.is_file()}))
    print(json.dumps(reports,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','input','request','comparison','out'):p.add_argument('--'+name,type=Path,required=True)
    main(p.parse_args())
