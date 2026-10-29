"""Local-only RGB comparison sheets after native render validation. No models."""
import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from box_annotations import sha256, write_json
from review_box_batch import review


def font(size):
    try: return ImageFont.truetype('C:/Windows/Fonts/arial.ttf',size)
    except OSError: return ImageFont.truetype('DejaVuSans.ttf',size)


def crop(path):
    with Image.open(path) as im:
        return im.convert('RGB').crop((255,45,365,435)).resize((220,780))


def sheet(path,title,subtitle,panels,footer):
    width=max(1150,len(panels)*244+24)
    canvas=Image.new('RGB',(width,950),'#191d24');draw=ImageDraw.Draw(canvas)
    draw.text((24,12),title,font=font(25),fill='white')
    draw.text((24,48),subtitle,font=font(17),fill='white')
    for n,(caption,path_or_image) in enumerate(panels):
        x=24+n*244
        draw.text((x,87),caption,font=font(17),fill='white')
        image=path_or_image if isinstance(path_or_image,Image.Image) else crop(path_or_image)
        canvas.paste(image,(x,115))
    draw.text((24,912),footer,font=font(16),fill='white')
    canvas.save(path)


def main(args):
    batch=args.batch.resolve(); prepared=args.prepared.resolve();old=args.legacy.resolve()
    verified=review(batch)
    if not verified['passed']: raise ValueError('Native validation failed')
    run=json.loads((batch/'run.json').read_text())
    refs=json.loads((prepared/'references.json').read_text())['references']
    if run['renderer_profile']!='meniscus-transmission-v2' or any(r['source_run'] not in (1,2,3) for r in refs):
        raise ValueError('Unexpected renderer/reference boundary')
    for ref in refs:
        if sha256(prepared/ref['file'])!=ref['image_sha256']:raise ValueError('Reference RGB changed')
    by_id={r['source_id']:prepared/r['file'] for r in refs}
    rgb=lambda n:batch/'rgb'/f'{n}.png'
    stages=['v2-00-empty','v2-01-pendant-drop','v2-02-falling-jet','v2-03-impact-no-pool',
            'v2-04-first-wet-patch','v2-05-small-pale-pool','v2-07-low-pale']
    captions=['Empty','Pendant drop','Falling jet','Jet reaches base','First wet patch','Small pool','Low pale body']
    for folder,suffix in [('rgb','rgb'),('overlays','boxes')]:
        sheet(batch/f'early-pour-{suffix}.png','Early-pour review: body and stream remain separate',
            'Procedural static states, not validated chronology. Original camera; display crops enlarged only.',
            [(caption,batch/folder/f'{identity}.png') for identity,caption in zip(stages,captions)],
            'Boxes: body orange / stream cyan / vial green. Geometric labels, not verified liquid measurement pixels.')
    sheet(batch/'pale-before-after.png','Pale liquid: real references, old proxy, revised rendering',
        'Qualitative RGB comparison; volumes and hues are NOT physically paired or calibrated. References: Run 3 only here.',
        [('Real: low pale',by_id['vial_fill_20260923_184720_822117']),
         ('Old: opacity proxy',old/'rgb/03-tiny-pale.png'),
         ('New: small pale pool',rgb('v2-05-small-pale-pool')),
         ('Real: higher pale',by_id['vial_fill_20260923_184726_838430']),
         ('New: mid pale body',rgb('v2-08-mid-pale'))],
        'Revised: transmitting volume + curved surface. Bubbles, wall films, camera noise and measured optical calibration remain incomplete.')
    sheet(batch/'clear-geometry-review.png','Clear liquid: keep RGB and geometric masks distinct',
        'Clear remains a difficult observability case; a geometric box does not certify a visible RGB liquid boundary.',
        [('Old clear proxy',old/'rgb/11-high-clear-proxy-stopped.png'),
         ('New clear / RGB',rgb('v2-11-mid-clear')),
         ('New clear / boxes',batch/'overlays/v2-11-mid-clear.png'),
         ('New clear / body mask',batch/'masks/v2-11-mid-clear-201.png')],
        'Old/new heights differ: this sheet illustrates material and mask policy, not a matched material-only ablation.')
    # Runs 1 and 2 constrain colored jets/low pools; Run 3 constrains pale RGB.
    sheet(batch/'runs123-reference-cues.png','Appearance references: RGB from Runs 1-3 only',
        'No teacher boxes, reviewed timing labels, recipes, Run 4 or Run 5 input. Qualitative visual guidance only.',
        [('Run 1: blue early',by_id['vial_fill_20260922_170544_697177']),
         ('Run 1: blue pool',by_id['vial_fill_20260922_170600_743982']),
         ('Run 2: thin jet',by_id['vial_fill_20260922_192633_265636']),
         ('Run 2: blue pool',by_id['vial_fill_20260922_192639_295513']),
         ('Run 3: pale low',by_id['vial_fill_20260923_184720_822117']),
         ('Run 3: pale higher',by_id['vial_fill_20260923_184726_838430'])],
        'Reference images keep source identities and SHA256 in references.json. Selected frames are not a new accuracy benchmark.')
    rows=[json.loads(line) for line in (batch/'manifest.jsonl').read_text().splitlines()]
    write_json(batch/'appearance-review.json',dict(schema_version=1, complete=True,
        renderer_profile=run['renderer_profile'], training_eligible=False, human_appearance_approved=False,
        no_training=True, reference_runs=[1,2,3], references=refs,
        native_verification_sha256=sha256(batch/'verification.json'),
        cases=[dict(sample_id=r['sample_id'],rgb=r['image_file'],image_sha256=r['image_sha256'],
            body=next(o for o in r['annotations'] if o['instance_id']==201),
            stream=next(o for o in r['annotations'] if o['instance_id']==301),
            review_status='pending_visual_approval_not_training_admitted') for r in rows],
        review_questions=['Is the pale liquid recognizably transmitting rather than milky?',
            'Does the early-pool footprint/meniscus resemble training references?',
            'Are body and stream separately localized and absent states empty?',
            'Which RGB boundaries are unobservable despite known geometric presence?'],
        limits=['Procedural snapshots, not CFD or calibrated optical/temporal truth.',
            'Opaque annotation passes remain geometric, not refracted RGB segmentation.',
            'No transfer improvement or color-QA claim without subsequent reviewed evaluation.']))
    print(f'Local comparison PNGs and appearance-review.json: {batch}')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch',type=Path,required=True)
    parser.add_argument('--prepared',type=Path,required=True)
    parser.add_argument('--legacy',type=Path,required=True)
    main(parser.parse_args())
