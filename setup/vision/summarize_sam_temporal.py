"""CPU plots and exact-frame review boards for the matched SAM/temporal trial."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())


def main(a):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image,ImageDraw
    for root in [a.comparison,a.refinement]:
        c=read(root/'completed.json')
        for group,sub in [('artifacts',''),('helpers','helpers'),('draft_masks','draft-masks')]:
            for name,digest in c.get(group,{}).items():
                if sha(root/sub/name)!=digest:raise ValueError('Completed artifact changed')
    plan=read(a.comparison/'plan.json');report=read(a.comparison/'comparison.json');ref=read(a.refinement/'comparison.json')
    a.out.mkdir(parents=True,exist_ok=False)
    tasks={t['frame_index']:t for t in read(a.comparison/'review-drafts.json')['tasks']}
    rows={r['source_frame']:r for r in [json.loads(s) for s in (a.refinement/'pass-1.jsonl').read_text().splitlines()]}
    names=['Legacy R-CNN + SAM','Synthetic R-CNN + SAM','Temporal direct'];arms=plan['arms']
    x=np.arange(3);fig,axes=plt.subplots(2,1,figsize=(11,8),constrained_layout=True)
    axes[0].bar(x-.18,[report['arms'][k]['latency_ms']['total']['median'] for k in arms],.36,label='Median')
    axes[0].bar(x+.18,[report['arms'][k]['latency_ms']['total']['p95'] for k in arms],.36,label='p95')
    axes[0].set_xticks(x,names);axes[0].set_ylabel('Milliseconds');axes[0].legend();axes[0].grid(axis='y',alpha=.2)
    axes[0].set_title('Matched GB10 cached CV processing — 528 samples per arm')
    axes[1].bar(x-.18,[report['arms'][k]['proposal_frames'] for k in arms],.36,label='Mask proposal available')
    axes[1].bar(x+.18,[report['arms'][k]['qa_pass'] for k in arms],.36,label='Color sample passes existing QA')
    axes[1].set_xticks(x,names);axes[1].set_ylabel('Frames / 264');axes[1].legend();axes[1].grid(axis='y',alpha=.2)
    axes[1].set_title('Availability counts — not mask or product accuracy')
    fig.savefig(a.out/'matched-comparison.png',dpi=160);plt.close(fig)
    def decode(e):
        m=np.zeros((480,640),np.uint8)
        if e:
            x,y=e['origin'];w,h=e['size'];flat=np.zeros(w*h,np.uint8)
            for start,length in e['runs']:flat[start:start+length]=1
            m[y:y+h,x:x+w]=flat.reshape(h,w)
        return m.astype(bool)
    # Same unreviewed early/later exact frames for every model; no absence labels inferred.
    board=Image.new('RGB',(1200,6*280+50),'#101925');d=ImageDraw.Draw(board)
    d.text((10,10),'Exact early / later frames, all6clips | green = raw proposal; no reviewed presence/mask truth',fill='white')
    for y,cycle in enumerate(plan['cycles']):
        for phase,offset in enumerate([-30,30]):
            task=tasks[cycle['provisional_anchor_frame']+offset];image=a.source_probe/task['image']
            assert sha(image)==task['image_sha256'];rgb=np.asarray(Image.open(image).convert('RGB')).copy()
            for x,arm in enumerate(arms):
                draft=next(v for v in task['drafts'] if v['mode']==arm);mask=np.asarray(Image.open(a.comparison/draft['mask']))>=128
                display=rgb.copy();display[mask]=(.6*display[mask]+.4*np.array([45,255,150])).astype(np.uint8)
                crop=Image.fromarray(display).crop((240,230,400,440));col=phase*3+x;top=y*280+50
                board.paste(crop,(col*200+20,top+45));d.text((col*200+5,top),names[x],fill='white')
                d.text((col*200+5,top+17),f'{cycle["case_id"]} f{task["frame_index"]}',fill='#b8cbd8')
                d.text((col*200+5,top+32),'Early review frame' if phase==0 else 'Later review frame',fill='#b8cbd8')
    board.save(a.out/'early-later-mask-review.png')
    board=Image.new('RGB',(3*260,6*260+45),'#101925');d=ImageDraw.Draw(board)
    d.text((10,10),'Same temporal proposal -> direct mask / SAM refinement / final sample | unreviewed',fill='white')
    for y,cycle in enumerate(plan['cycles']):
        frame=cycle['provisional_anchor_frame']+30;task=tasks[frame];r=rows[frame]
        rgb=np.asarray(Image.open(a.source_probe/task['image']).convert('RGB')).copy()
        draft=next(v for v in task['drafts'] if v['mode']=='temporal')
        masks=[np.asarray(Image.open(a.comparison/draft['mask']))>=128,decode(r['raw_union']),decode(r['measurement']['geometry']['mask'])]
        for x,(title,mask) in enumerate(zip(['Temporal raw','SAM refined raw','Final actual color sample'],masks)):
            display=rgb.copy();display[mask]=(.6*display[mask]+.4*np.array([45,255,150])).astype(np.uint8)
            board.paste(Image.fromarray(display).crop((240,230,400,440)),(x*260+50,y*260+85))
            d.text((x*260+5,y*260+48),title,fill='white');d.text((x*260+5,y*260+65),f'{cycle["case_id"]} f{frame}',fill='#b8cbd8')
    board.save(a.out/'temporal-sam-refinement-review.png')
    speedup={k:report['arms'][k]['latency_ms']['total']['median']/report['arms']['temporal']['latency_ms']['total']['median'] for k in arms[:2]}
    result=dict(matched=report,refinement=ref,median_speed_ratio_to_temporal=speedup,
        comparison_plan_sha256=sha(a.comparison/'plan.json'),refinement_plan_sha256=sha(a.refinement/'plan.json'),
        source_video_sha256=plan['source_video_sha256'],human_accuracy=None,app_changed=False,run5_used=False,
        helper_sha256=sha(__file__))
    (a.out/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# Matched SAM2 / temporal comparison','',
        'Same264 native Run4 source frames across six clips, same synthetic vial detector, same QA, same GB10. Full264-frame warm-up per arm and two measured passes; model state resets per clip.','',
        '| System | CV median / p95 | Mask proposals / 264 | Color QA passes / 264 |','| --- | ---: | ---: | ---: |']
    for arm,name in zip(arms,names):
        r=report['arms'][arm];t=r['latency_ms']['total'];lines.append(f'| {name} | {t["median"]:.2f} / {t["p95"]:.2f} ms | {r["proposal_frames"]} | {r["qa_pass"]} |')
    lines+=['',f'Temporal-prompted SAM refinement produced **{ref["qa_pass"]}/264** qualified color samples versus **{ref["original_temporal_qa"]}/264** direct: **{ref["qa_gains"]} gains, {ref["qa_losses"]} losses**.',
        f'Actual SAM inference was invoked on {ref["sam_invoked_frames"]} frames. Cached vial/temporal outputs were reused; incremental prompt/SAM/QA median was {ref["incremental_latency_ms"]["incremental_total"]["median"]:.2f} ms. This is not a full hybrid pipeline latency benchmark.',
        '', '## What the evidence supports','',
        '- Temporal direct is about6x faster in this matched cached-input test.',
        '- Guided SAM produces more QA-eligible samples. This does not establish more accurate masks or presence.',
        '- The temporal model can supply SAM prompts without R-CNN. The exploratory hybrid increases QA availability while retaining the temporal proposal gate.',
        '- All compared methods and refinement have identical raw/final masks, colors and QA across repeated passes.',
        '- Presence, visible-pool mask accuracy and first-visible delay remain unavailable without human review.',
        '', '## View the evidence','',
        '- Matched3-arm video: ../gb10-results/sam-comparison-verified-v2/sam-vs-temporal.mp4',
        '- Actual temporal-to-SAM video: ../gb10-results/temporal-sam-refinement-v1/temporal-vs-sam-refinement.mp4',
        '- early-later-mask-review.png: same early/later frame overlays; early does not mean confirmed empty.',
        '- temporal-sam-refinement-review.png: direct/refined masks and final sampling interiors.',
        '', '## Limits','',
        'Legacy R-CNN was trained on realCSI Runs1-3 and selected on Run4; it is a comparator. Synthetic R-CNN and temporal weights use syntheticCSI fine-tuning only; generic COCO/SAM pretraining is disclosed. Training sizes, initializations and selection criteria differ, so this is a system comparison, not an architecture-only ablation.',
        'All clips are inspected development data. Timing excludes decode, rendering, network, serialization and browser. QA availability and tall-mask counts are not accuracy. Later/bad references remain a temporal failure mode. No app defaults or industrial connections changed.',
        '', 'The first run finished inference/video but failed its manifest when encountering __pycache__. Its files/logs remain intact. A fresh verified copy independently redecoded all inputs/video and recomputed timing/counts without repeatingGPU inference. The original October20 prompt/bridge implementation was replayed exactly on all528SAM policy inputs; sources are past-only.',
        '', 'Use the existing72-frame human review packet; annotation_review.score_sam_temporal_review can score all arms and the optional refinement from the same explicit source-matched, non-demonstration export.']
    (a.out/'README.md').write_text('\n'.join(lines)+'\n')
    (a.out/'completed.json').write_text(json.dumps(dict(artifacts={p.name:sha(p) for p in a.out.iterdir() if p.is_file()}),indent=2)+'\n')
    print(json.dumps(dict(out=str(a.out),speedup=speedup,qa_gains=ref['qa_gains'],qa_losses=ref['qa_losses'])))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['comparison','source-probe','refinement','out']:p.add_argument('--'+name,type=Path,required=True)
    main(p.parse_args())
