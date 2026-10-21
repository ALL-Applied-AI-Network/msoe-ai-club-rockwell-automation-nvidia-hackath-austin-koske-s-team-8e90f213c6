"""Local single-panel evidence video and explicitly simulated action-loop video."""
import argparse,json,textwrap
from pathlib import Path
import sys
import av,cv2,numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from liquid_specialists_data import read,write,sha
from replay.geometry import decode_mask
from render_liquid_pipeline import pixel_hash
from stage2_policy import ActionCoordinator


def text(canvas,value,x,y,color=(225,232,240),size=.6):
    cv2.putText(canvas,value,(x,y),cv2.FONT_HERSHEY_SIMPLEX,size,color,1,cv2.LINE_AA)


def box(canvas,lines,x=690,y=130,color=(225,232,240)):
    for line in lines:
        for piece in textwrap.wrap(str(line),55) or ['']:
            text(canvas,piece,x,y,color);y+=27
        y+=8


def verify(path,count):
    decoded=0;snapshot=None
    with av.open(str(path)) as c:
        for frame in c.decode(video=0):
            assert frame.width==1280 and frame.height==720
            if decoded==count//2:snapshot=frame.to_ndarray(format='bgr24')
            decoded+=1
    if decoded!=count:raise ValueError('Incomplete output video')
    return snapshot


def render_real(a):
    done=read(a.real/'completed.json');plan=read(a.real/'plan.json')
    if sha(a.source)!=plan['source_sha256']:raise ValueError('Wrong source')
    for name in ('frames.jsonl','observations.json'):
        if sha(a.real/name)!=done['artifacts'][name]:raise ValueError('Evidence changed')
    rows=[json.loads(s) for s in (a.real/'frames.jsonl').read_text().splitlines()]
    checkpoints=read(a.audit/'phase-checkpoints.json')
    writers=[cv2.VideoWriter(str(a.out/n),cv2.VideoWriter_fourcc(*'mp4v'),5.,(1280,720))
             for n in ('run4-evidence.mp4','run4-checkpoint-highlight.mp4')]
    if not all(w.isOpened() for w in writers):raise RuntimeError('Cannot encode')
    n=short=0
    try:
        with av.open(str(a.source)) as c:
            for index,frame in enumerate(c.decode(video=0)):
                if index%6:continue
                row=rows[n];rgb=frame.to_ndarray(format='rgb24')
                if row['source_rgb_sha256']!=pixel_hash(rgb) or frame.pts!=row['pts'] or str(frame.time_base)!=row['time_base']:
                    raise ValueError('Video source does not match frozen measurements')
                bgr=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR);m=row['measurement'];state=row['fill_detection']
                mask=decode_mask(m['geometry']['mask'],rgb.shape[:2]).astype(bool)
                bgr[mask]=(.55*bgr[mask]+.45*np.array([90,220,90])).astype(np.uint8)
                vial=m['geometry']['vial_box']
                if vial:
                    x,y,r,b=[int(v) for v in vial];cv2.rectangle(bgr,(x,y),(r,b),(230,170,60),1)
                canvas=np.full((720,1280,3),(23,28,38),np.uint8)
                text(canvas,'STAGE 2 / REAL RUN 4 / OFFLINE EVIDENCE',24,42,size=.85)
                text(canvas,f'Native PTS {row["seconds"]:.1f}s | decoded frame {index} | exact current-frame mask',24,76,size=.59)
                canvas[112:592,20:660]=bgr
                current=[p for p in checkpoints if p['detection_id']==state['detection_id'] and p['available_frame']<=index]
                checkpoint=current[-1] if current else None
                lines=[f'Fill candidate: {state["detection_id"] or "none"}',
                    f'Image phase: {state["phase"]}',
                    'Current sample QA: '+('PASSES' if m['measurement_valid'] else 'UNAVAILABLE'),
                    f'Current sampling pixels: {m["geometry"]["mask_pixels"]}',
                    'Phase checkpoint: '+(f'available at {checkpoint["available_s"]:.1f}s (past evidence)' if checkpoint else 'not yet available'),
                    'Intended recipe: UNKNOWN','Process-complete signal: UNAVAILABLE',
                    'Recipe decision: ABSTAIN','Hardware command: NONE']
                box(canvas,lines)
                text(canvas,'Green = experimental sampling mask. A phase checkpoint is not final recipe completion.',24,635,size=.64)
                text(canvas,'Perception and sample availability only; no reviewed defect or product-release decision.',24,672,color=(125,195,240),size=.64)
                writers[0].write(canvas);n+=1
                if 55<=row['seconds']<78:writers[1].write(canvas);short+=1
                if index in (720,2094,2196):cv2.imwrite(str(a.out/f'run4-frame-{index:05d}.png'),canvas)
    finally:
        for writer in writers:writer.release()
    if n!=len(rows):raise ValueError('Incomplete input inventory')
    verify(a.out/'run4-evidence.mp4',n);snapshot=verify(a.out/'run4-checkpoint-highlight.mp4',short)
    cv2.imwrite(str(a.out/'run4-highlight-decoded.png'),snapshot)
    return dict(source_pixels_and_native_pts_verified=n,full_decoded_frames=n,highlight_decoded_frames=short)


def render_simulation(a):
    sim=read(a.simulation/'completed.json')
    files=['r0-d1-h2.0-seed0-ideal-phase_reference_3_frames.json']*2+['r0-dnone-h0.0-seed0-ideal-phase_reference_3_frames.json']*2
    for name in set(files):
        if sha(a.simulation/name)!=sim['artifacts'][name]:raise ValueError('Simulation trace changed')
    loop=ActionCoordinator('demo-sim');offset=0.;count=0;derivation=[]
    path=a.out/'simulated-action-loop.mp4'
    writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),5.,(1280,720))
    if not writer.isOpened():raise RuntimeError('Cannot encode')
    try:
        for index,name in enumerate(files):
            case=read(a.simulation/name);vial='sim-vial'+str(index+1)
            trace=case['trace'];event=None
            for frame,row in enumerate(trace):
                if frame%2:continue # simulated time advances 0.4s per 0.2s playback
                result=row['result'];sample=row['sample'];t=sample['source_time_s'];score=result['delta_e2000']
                canvas=np.full((720,1280,3),(23,28,38),np.uint8)
                text(canvas,'SIMULATION / STAGE 2 DECISION AND ACTION CONTRACT',24,42,size=.81)
                text(canvas,f'{vial} | virtual time {offset+t:.1f}s | 2x playback | no hardware connected',24,78)
                text(canvas,'Observed toy camera color',30,137)
                text(canvas,'Expected recipe-prefix color',350,137)
                observed=np.array(case['observed_rgb'][frame][::-1],dtype=np.uint8)
                expected=np.array(case['expected_rgb'][frame][::-1],dtype=np.uint8)
                canvas[158:298,30:310]=observed;canvas[158:298,350:630]=expected
                # Plot only already available scores, never a future trajectory.
                history=trace[:frame+1];points=[]
                for j,h in enumerate(history):
                    de=h['result']['delta_e2000']
                    if de is not None:points.append((int(40+580*j/max(1,len(trace)-1)),int(550-min(60,de)*3.3)))
                cv2.rectangle(canvas,(30,335),(630,575),(65,75,90),1)
                cv2.line(canvas,(30,533),(630,533),(90,150,235),1)
                for left,right in zip(points,points[1:]):cv2.line(canvas,left,right,(100,215,220),2)
                text(canvas,'Past/current deviation only; orange line = 5 dE engineering threshold',30,605,size=.48)
                lines=[f'State: {result["status"]}',f'Ingredient phase: {sample["phase_key"]}',
                    'Deviation: '+('unavailable' if score is None else f'{score:.2f} dE2000'),
                    f'Consecutive evidence: {result.get("consecutive_deviations",0)} / 3',
                    'Requested disposition: '+('REVIEW THIS VIAL' if result['warning_time_s'] is not None else 'monitor'),
                    'Batch hold: '+('LATCHED' if loop.hold else 'not latched'),
                    'Flush receipt: '+('recorded; verifying recovery' if loop.flush_acknowledged else 'not recorded'),
                    'Physical actions issued: ZERO']
                box(canvas,lines)
                text(canvas,'Analytic optical toy. The scenario sets residue to zero after a simulated flush receipt.',24,656,size=.58,color=(120,190,245))
                text(canvas,'This tests software behavior; it does not measure real defects or physical flush effectiveness.',24,690,size=.58)
                writer.write(canvas);count+=1
            final=trace[-1]['result']['status']
            completed_at=offset+case['departure_s']
            event=loop.complete(vial,'sim-recipe',final,completed_at,session_id='demo-sim')
            derivation.append(dict(vial_id=vial,source_trace=name,source_trace_sha256=sha(a.simulation/name),
                                   assessment=final,action=event['action']))
            if index==1:
                loop.acknowledge_flush(completed_at+1.,'SIMULATED_OPERATOR','SIMULATED_COMPLETED_FLUSH_RECEIPT',session_id='demo-sim')
            canvas=np.full((720,1280,3),(23,28,38),np.uint8)
            text(canvas,'SIMULATION / COMPLETED VIAL ACTION',35,65,size=.9)
            box(canvas,[vial,event['action'],'Hold remains latched: '+str(loop.hold),
                'Flush acknowledgement never clears the hold.',
                'Two qualified completed normal vials are required for a resume recommendation.',
                'All actions advisory. Physical flush response is prescribed by this fixture.'],x=50,y=160)
            if index==1:text(canvas,'Operator records simulated flush completion before the next vial.',50,555,size=.65)
            if index==3:cv2.imwrite(str(a.out/'simulated-action-final.png'),canvas)
            for _ in range(10):writer.write(canvas);count+=1
            offset=completed_at+3.
    finally:writer.release()
    snapshot=verify(path,count);cv2.imwrite(str(a.out/'simulation-decoded.png'),snapshot)
    write(a.out/'derived-action-loop.json',dict(kind='simulation_from_logged_policy_outputs',source_traces=derivation,events=loop.events,
        prescribed_postflush_residue_ml=0.,physical_flush_effectiveness_measured=False,hardware_actions=0))
    return dict(fully_decoded_frames=count,derived_actions=[r['action'] for r in loop.events])


def main(a):
    if a.out.exists():raise ValueError('Use fresh output')
    a.out.mkdir(parents=True)
    real=render_real(a);simulation=render_simulation(a)
    write(a.out/'completed.json',dict(schema_version=1,real=real,simulation=simulation,
        script_sha256=sha(Path(__file__)),artifacts={p.name:sha(p) for p in a.out.iterdir() if p.is_file()},
        hardware_actions=0,run5_used=False))
    print(json.dumps(dict(real=real,simulation=simulation),indent=2))


if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__)
 for n in ('real','audit','simulation','source','out'):p.add_argument('--'+n,type=Path,required=True)
 main(p.parse_args())
