"""Verify input-only cache pixels, live inference acknowledgements and browser evidence."""
import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys

import av
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from replay.geometry import decode_mask
from replay.measurement import rgb_to_lab
from replay.operator.prerender import PrerenderBundle


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text())


def main(a):
    a.out.mkdir(parents=True,exist_ok=False)
    bundle=PrerenderBundle(a.cache);manifest=bundle.manifest
    browser=read(a.browser/'verification.json')
    assert browser['status']=='passed' and all(browser['checks'].values())
    for name,expected in browser['artifacts'].items():assert sha(a.browser/name)==expected
    samples={s['video_frame']:s for s in manifest['samples']};decoded=0
    with av.open(str(bundle.video)) as container:
        for index,frame in enumerate(container.decode(video=0)):
            assert abs(float(frame.pts*frame.time_base)-index/30)<1e-9
            assert (frame.width,frame.height)==(640,480)
            if index in samples:
                sample=samples[index];rgb=frame.to_ndarray(format='rgb24')
                png=cv2.cvtColor(cv2.imread(str(a.cache/sample['png'])),cv2.COLOR_BGR2RGB)
                assert np.array_equal(rgb,png)
                assert hashlib.sha256(rgb.tobytes()).hexdigest()==sample['rgb_sha256']
            decoded+=1
    assert decoded==manifest['frames']
    # Capture complete-line prefixes of growing logs without modifying originals.
    snapshots={}
    def snapshot(path):
        raw=path.read_bytes();end=raw.rfind(b'\n')+1;prefix=raw[:end]
        relative=path.relative_to(a.run);target=a.out/'snapshots'/relative
        target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(prefix)
        snapshots[str(relative)]=dict(sha256=sha(target),bytes=len(prefix),
            excluded_incomplete_tail_bytes=len(raw)-end,source_log_may_continue=True)
        return [json.loads(line) for line in prefix.splitlines()]
    rows=snapshot(a.run/'frames.jsonl');lookup={};colors=0;cached=0
    for record in rows:
        source=record['source']
        if not source['stream_id'].startswith('prerender-'):continue
        cached+=1;sample=manifest['samples'][source['sequence']];output=record['model_output']
        assert source['rgb_sha256']==sample['rgb_sha256']==output['source_rgb_sha256']
        assert source['pts']==sample['pts'] and source['time_base']==sample['time_base']
        assert abs(float(source['pts']*Fraction(source['time_base']))-output['seconds'])<1e-9
        assert output['inference_ms']>0 and output['hardware_command']=='none'
        assert output['recipe_decision']['status']=='unavailable'
        measurement=output['measurement']
        if measurement['lab'] is not None:
            rgb=cv2.cvtColor(cv2.imread(str(a.cache/sample['png'])),cv2.COLOR_BGR2RGB)
            mask=decode_mask(measurement['geometry']['mask'],(480,640)).astype(bool)
            assert mask.any();median=np.median(rgb[mask],axis=0)
            assert np.allclose(median,measurement['rgb'],atol=1e-9)
            assert np.allclose(rgb_to_lab(median),measurement['lab'],atol=1e-8);colors+=1
        lookup[(source['stream_id'],source['sequence'])]=record
    acknowledgements=0;skips=0
    for path in sorted(a.run.glob('source-*-prerender/delivery.jsonl')):
        for delivered in snapshot(path):
            key=(delivered['stream_id'],delivered['sample_index'])
            # Delivery can advance beyond the previously frozen frame-log prefix.
            if key not in lookup:continue
            source=lookup[key]['source'];ack=delivered['ack']
            assert ack['accepted_sequence']==source['sequence']
            assert ack['rgb_sha256']==source['rgb_sha256']==delivered['rgb_sha256']
            assert ack['hardware_commands']==0 and delivered['publish_wall_s']>0
            acknowledgements+=1;skips+=delivered['skipped_samples']
    assert cached>0 and acknowledgements>0 and colors>0
    completed=read(a.render/'render-completed.json');assert completed['source_assets_unchanged']
    assert sha(a.render/'render-manifest.jsonl')==manifest['render_manifest_sha256']
    assert sha(a.render/'render-completed.json')==manifest['render_completed_sha256']
    assets=ROOT/'datasets/mixing_twin/assets/csi_digital_twin_v1'
    for name,expected in completed['assets_sha256'].items():
        assert sha(assets/Path(name.replace('\\','/')))==expected
    assert read(a.run/'source-001-isaac/stopped.json')['source_assets_unchanged']
    launch=read(a.run/'launch.json')
    for name,expected in launch['source_sha256'].items():assert sha(a.run/'helpers'/name)==expected
    result=dict(status='passed',decoded_video_frames=decoded,fps=manifest['fps'],
        duration_s=manifest['duration_s'],verified_video_input_samples=len(samples),
        cached_predictions=False,verified_live_CV_samples=cached,
        joined_inference_acknowledgements=acknowledgements,recomputed_sampling_colors=colors,
        skipped_input_samples_in_joined_deliveries=skips,source_assets_unchanged=True,
        native_source_toggle_stopped_cleanly=True,browser_checks=browser['checks'],
        browser_playback=browser['video_playback'],video_sha256=manifest['video_sha256'],
        cache_manifest_sha256=sha(a.cache/'manifest.json'),browser_report_sha256=sha(a.browser/'verification.json'),
        snapshots=snapshots,hardware_commands=0,operational_recipe_assessment='unavailable',
        limits=['30 fps playback is independent of sampled live CV throughput.',
            'Input is a fixed repeating development simulation; no real camera or model-accuracy result.',
            'Live log snapshots retain complete-line prefixes; later inference is outside this report.'])
    (a.out/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='snapshots'},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('cache','run','render','browser','out'):parser.add_argument('--'+key,type=Path,required=True)
    main(parser.parse_args())
