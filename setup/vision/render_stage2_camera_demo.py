"""Single-panel real-pixel camera correction experiment, explicitly labeled."""
import argparse,json,textwrap
from pathlib import Path
import sys
import av,cv2,numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from stage2_camera_reference import PATCHES,correction,feature,anchor_health,perturb
from liquid_specialists_data import read,write,sha
from replay.geometry import decode_mask
from replay.measurement import delta_e
from render_liquid_pipeline import pixel_hash


def text(canvas,value,x,y,size=.59,color=(225,232,240)):
 cv2.putText(canvas,value,(x,y),cv2.FONT_HERSHEY_SIMPLEX,size,color,1,cv2.LINE_AA)


def main(a):
 if a.out.exists():raise ValueError('Use fresh output')
 a.out.mkdir(parents=True)
 camera=read(a.camera/'completed.json');cal=read(a.camera/'calibration.json');real=read(a.real/'plan.json')
 if sha(a.camera/'calibration.json')!=camera['artifacts']['calibration.json'] or sha(a.source)!=real['source_sha256']:
  raise ValueError('Camera/source identity changed')
 rows=[json.loads(s) for s in (a.real/'frames.jsonl').read_text().splitlines()]
 writer=cv2.VideoWriter(str(a.out/'camera-normalization-test.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),5.,(1280,720))
 if not writer.isOpened():raise RuntimeError('Cannot encode')
 reference=np.array(cal['reference_rgb']);logged=[]
 try:
  with av.open(str(a.source)) as c:
   for index,frame in enumerate(c.decode(video=0)):
    if index<1650 or index%6:continue
    if index>=2340:break
    row=rows[index//6];rgb=frame.to_ndarray(format='rgb24')
    if pixel_hash(rgb)!=row['source_rgb_sha256'] or frame.pts!=row['pts'] or str(frame.time_base)!=row['time_base']:
     raise ValueError('Source frame mismatch')
    anchors=[]
    for key in cal['selected']:
     x,y,r,b=PATCHES[key];anchors.append(np.median(rgb[y:b,x:r].reshape(-1,3),axis=0))
    anchors=np.array(anchors);injected=np.clip(np.rint(perturb(rgb,gain=[1,1.1,1],domain='srgb')),0,255).astype(np.uint8)
    changed=[]
    for key in cal['selected']:
     x,y,r,b=PATCHES[key];changed.append(np.median(injected[y:b,x:r].reshape(-1,3),axis=0))
    changed=np.array(changed);healthy,_,spread=anchor_health(changed,reference,cal['maximum_disagreement'])
    corrected,_,_=correction(injected.reshape(-1,3),changed,reference,'srgb_gain')
    corrected=corrected.reshape(rgb.shape);mask=decode_mask(row['measurement']['geometry']['mask'],rgb.shape[:2]).astype(bool)
    raw_error=corrected_error=None
    if row['measurement']['measurement_valid']:
     original_lab,_=feature(rgb[mask]);noisy_lab,_=feature(injected[mask])
     nominal,_,_=correction(rgb[mask],anchors,reference,'srgb_gain');nominal_lab,_=feature(nominal)
     corrected_lab,_=feature(corrected[mask]);raw_error=delta_e(original_lab,noisy_lab);corrected_error=delta_e(nominal_lab,corrected_lab)
    bgr=cv2.cvtColor(np.rint(corrected).astype(np.uint8),cv2.COLOR_RGB2BGR)
    for key in cal['selected']:
     x,y,r,b=PATCHES[key];cv2.rectangle(bgr,(x,y),(r,b),(60,210,220),1)
    if mask.any():
     contours,_=cv2.findContours(mask.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
     cv2.drawContours(bgr,contours,-1,(90,210,90),1)
    canvas=np.full((720,1280,3),(23,28,38),np.uint8)
    text(canvas,'STAGE 2 / CONTROLLED CAMERA-NORMALIZATION TEST',24,42,size=.82)
    text(canvas,f'Real Run4 | native PTS {row["seconds"]:.1f}s | injected +10% green | corrected view',24,77)
    canvas[112:592,20:660]=bgr
    lines=['Input: real video with an injected camera nuisance',
        'Background references: '+('healthy' if healthy else 'unavailable'),
        f'Patch disagreement: {spread*100:.2f}%',
        'Original mask/QA: reused; unreviewed',
        'Raw paired shift: '+('unavailable' if raw_error is None else f'{raw_error:.2f} dE2000'),
        'Corrected paired shift: '+('unavailable' if corrected_error is None else f'{corrected_error:.2f} dE2000'),
        'Recipe judgement: UNAVAILABLE','Hardware actions: ZERO']
    y=138
    for line in lines:
     for part in textwrap.wrap(line,50):text(canvas,part,690,y);y+=27
     y+=13
    text(canvas,'Yellow boxes: background anchors. Green outline: frozen experimental sampling mask.',24,633,size=.61)
    text(canvas,'Paired error uses the same unperturbed frame, not an acceptable-recipe standard.',24,669,size=.61,color=(120,190,245))
    writer.write(canvas)
    logged.append(dict(frame=index,pts=frame.pts,time_base=str(frame.time_base),source_rgb_sha256=pixel_hash(rgb),
        anchor_healthy=healthy,raw_paired_de2000=raw_error,corrected_paired_de2000=corrected_error))
    if index==2094:cv2.imwrite(str(a.out/'camera-test-poster.png'),canvas)
 finally:writer.release()
 decoded=0
 with av.open(str(a.out/'camera-normalization-test.mp4')) as c:
  for frame in c.decode(video=0):
   assert (frame.width,frame.height)==(1280,720)
   if decoded==65:cv2.imwrite(str(a.out/'decoded-camera-test.png'),frame.to_ndarray(format='bgr24'))
   decoded+=1
 if decoded!=len(logged) or decoded!=115:raise ValueError('Incomplete video')
 write(a.out/'frames.json',logged)
 result=dict(schema_version=1,source_frames_verified=115,fully_decoded_frames=decoded,
     kind='paired_camera_test_not_recipe_validation',camera_calibration_sha256=sha(a.camera/'calibration.json'),
     script_sha256=sha(Path(__file__)),run5_used=False,hardware_actions=0,
     limits=['All source pixels are real Run4; +10% green is an artificial camera perturbation.',
         'Original geometry/QA are frozen; detector and segmenter robustness are not evaluated.',
         'Paired errors do not establish absolute liquid color or product acceptance.'])
 result['artifacts']={p.name:sha(p) for p in a.out.iterdir() if p.is_file()};write(a.out/'completed.json',result)
 print(json.dumps(dict(fully_decoded_frames=decoded,source_frames_verified=len(logged)),indent=2))


if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__)
 for n in ('camera','real','source','out'):p.add_argument('--'+n,type=Path,required=True)
 main(p.parse_args())
