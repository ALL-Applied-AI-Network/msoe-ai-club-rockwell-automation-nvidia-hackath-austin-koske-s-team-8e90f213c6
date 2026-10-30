"""Materialize source-exact evidence; explicitly report unavailable frozen pixels."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())


def main():
    from PIL import Image
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('bundle','video','fixture-batch','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--frozen-probe',type=Path,help='Copy verified lossless probe frames; do not re-decode missing frames')
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    observations=[json.loads(s) for s in (a.bundle/'observations.jsonl').read_text().splitlines()]
    filling=[r for r in observations if r['station']=='vial_fill'];fixtures=[r for r in observations if r['station']=='final_fixture']
    video_hash=sha(a.video)
    if any(r['source']['source_id']!=video_hash for r in filling):raise ValueError('Original video differs from cached model source')
    wanted={r['source']['frame_index']:r for r in filling};assets={}
    (a.out/'fill').mkdir();(a.out/'fixture').mkdir()
    missing=[]
    if a.frozen_probe:
        tasks={r['frame_index']:r for r in read(a.frozen_probe/'review-drafts.json')['tasks']}
        for index,observation in wanted.items():
            source=observation['source'];task=tasks.get(index)
            if task is None:
                missing.append(dict(observation_id=observation['observation_id'],frame_index=index,
                    reason='Frozen RGB pixels unavailable in local lossless probe; no decoder substitution'))
                continue
            original=a.frozen_probe/task['image']
            with Image.open(original) as image:rgb_hash=hashlib.sha256(image.convert('RGB').tobytes()).hexdigest()
            if (sha(original)!=task['image_sha256'] or rgb_hash!=source['decoded_rgb_sha256']
                or task['source_video_sha256']!=video_hash or task['pts']!=source['pts'] or task['time_base']!=source['time_base']):
                raise ValueError('Frozen probe source identity differs')
            path=a.out/'fill'/f'f{index:06d}.png';shutil.copy2(original,path)
            assets[observation['observation_id']]=dict(image=path.relative_to(a.out).as_posix(),encoded_file_sha256=sha(path),
                decoded_rgb_sha256=rgb_hash,native_pts=task['pts'],time_base=task['time_base'],frame_index=index)
    else:
        import av
        with av.open(str(a.video)) as container:
            decode_original(container,wanted,a.out,assets)
        if len(assets)!=len(filling):raise ValueError('Not all frozen source frames decoded')
    fill_count=len(assets)
    samples={r['sample_id']:r for r in read(a.fixture_batch/'samples.json')};copied=set()
    for observation in fixtures:
        sample=samples[observation['lineage']['sample_id']];source=a.fixture_batch/sample['image']
        if sha(source)!=observation['source']['encoded_file_sha256']:raise ValueError('Original fixture JPEG differs')
        path=a.out/'fixture'/source.name
        if source.name not in copied:shutil.copy2(source,path);copied.add(source.name)
        assets[observation['observation_id']]=dict(image=path.relative_to(a.out).as_posix(),encoded_file_sha256=sha(path),
            decoded_rgb_sha256=None,native_pts=None,time_base=None,frame_index=None,capture_timestamp=sample['capture_timestamp'])
    report=dict(observations=len(observations),available_observation_images=len(assets),
        frozen_fill_frames=fill_count,missing_fill_frames=len(missing),original_fixture_jpegs=len(copied),source_video_sha256=video_hash,
        available_source_identities_verified=True,all_observation_pixels_available=not missing,
        materialization='frozen_lossless_probe' if a.frozen_probe else 'strict_original_decode',
        still_to_video_mapping='unmatched',training=False,local_gpu_used=False,
        assets=assets,missing=missing,helper_sha256=sha(__file__))
    (a.out/'source-assets.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('assets','missing')}),flush=True)


def decode_original(container,wanted,out,assets):
    from PIL import Image
    # Refuse even small RGB differences: cached inference must retain its pixels.
    stream=container.streams.video[0]
    for index,frame in enumerate(container.decode(stream)):
        if index in wanted:
            observation=wanted[index];source=observation['source']
            rgb=frame.to_ndarray(format='rgb24')
            if frame.pts!=source['pts'] or str(frame.time_base)!=source['time_base'] or hashlib.sha256(rgb.tobytes()).hexdigest()!=source['decoded_rgb_sha256']:
                raise ValueError(f'Original source clock/pixels differ at frame{index}; no substitution permitted')
            path=out/'fill'/f'f{index:06d}.png';Image.fromarray(rgb).save(path)
            assets[observation['observation_id']]=dict(image=path.relative_to(out).as_posix(),encoded_file_sha256=sha(path),
                decoded_rgb_sha256=source['decoded_rgb_sha256'],native_pts=frame.pts,time_base=str(frame.time_base),frame_index=index)
        if index>=max(wanted):break


if __name__=='__main__':main()
