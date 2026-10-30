"""Pin a browser-compatible video and lossless samples decoded from that video."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main(a):
    import av
    import cv2
    if a.out.exists():raise ValueError('Use a fresh cache directory')
    done=json.loads((a.render/'render-completed.json').read_text())
    if not done['source_assets_unchanged'] or done['fps']!=30:raise ValueError('Incomplete 30 Hz render')
    rendered=[json.loads(s) for s in (a.render/'render-manifest.jsonl').read_text().splitlines()]
    a.out.mkdir(parents=True);(a.out/'samples').mkdir();shutil.copy2(a.video,a.out/'video.mp4')
    samples=[];decoded=0
    with av.open(str(a.out/'video.mp4')) as container:
        stream=container.streams.video[0]
        if stream.average_rate!=30:raise ValueError('Video must retain real-time 30 fps')
        for index,frame in enumerate(container.decode(video=0)):
            if (frame.width,frame.height)!=(640,480) or abs(float(frame.pts*frame.time_base)-index/30)>1e-8:
                raise ValueError('Decoded video dimensions/clock differ')
            source=rendered[index]
            if source['local_frame']%6==0:
                rgb=frame.to_ndarray(format='rgb24');path=Path('samples')/f'{index:06d}.png'
                if not cv2.imwrite(str(a.out/path),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)):raise RuntimeError('PNG export failed')
                samples.append(dict(seconds=float(frame.pts*frame.time_base),pts=frame.pts,time_base=str(frame.time_base),
                    video_frame=index,segment_id=source['segment_id'],telemetry=source['telemetry'],
                    rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),png=path.as_posix(),png_sha256=sha(a.out/path)))
            decoded+=1
    if decoded!=done['frames'] or decoded!=len(rendered):raise ValueError('Video is incomplete')
    manifest=dict(schema_version=1,kind='prerendered_isaac_input_only',video='video.mp4',video_sha256=sha(a.out/'video.mp4'),
        fps=30,frames=decoded,duration_s=decoded/30,width=640,height=480,samples=samples,
        cached_predictions=False,sample_pixels='decoded_from_this_encoded_video',render_root=str(a.render),
        render_manifest_sha256=sha(a.render/'render-manifest.jsonl'),render_completed_sha256=sha(a.render/'render-completed.json'))
    (a.out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({k:v for k,v in manifest.items() if k!='samples'},indent=2));print('CV input samples:',len(samples))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('render','video','out'):p.add_argument('--'+key,type=Path,required=True)
    main(p.parse_args())
