"""Prepare a CPU-only report, source-bound evidence board and blank capture sheets."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import shutil


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(root,out):
    from PIL import Image,ImageDraw,ImageFont
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,PageBreak,Image as PDFImage

    out.mkdir(parents=True,exist_ok=False)
    report=read(root/'rehearsal-v2/report.json');sources=read(root/'sources-v2/source-assets.json')
    observations=[json.loads(s) for s in (root/'bundle-v1/observations.jsonl').read_text().splitlines()]
    refs=read(root/'bundle-v1/reference-candidates.json');geometry=read(root/'geometry-v1/geometry-checks.json')
    # Select an available, QA-passing cached frame only for presentation, not fitting.
    filling=next(r for r in observations if r['station']=='vial_fill' and r['measurement']['qa_pass'] and r['observation_id'] in sources['assets'])
    capped=Counter(r['target']['context_id'] for r in observations if r['station']=='final_fixture' and r['proposal']['proposed_class']=='pass')
    fixture=[r for r in observations if r['station']=='final_fixture' and r['target']['context_id']==capped.most_common(1)[0][0]]
    chosen=[filling,fixture[0]];panels=[]
    for observation in chosen:
        asset=sources['assets'][observation['observation_id']];path=root/'sources-v2'/asset['image']
        if sha(path)!=asset['encoded_file_sha256']:raise ValueError('Source pixels changed')
        panels.append(Image.open(path).convert('RGB'))
    overlay=Image.new('RGBA',(640,480));draw=ImageDraw.Draw(overlay)
    mask=filling['proposal']['raw_mask']
    if mask:
        x,y=mask['origin'];w,h=mask['size']
        for start,length in mask['runs']:
            for pos in range(start,start+length):draw.point((x+pos%w,y+pos//w),fill=(40,230,150,100))
    panels[0]=Image.alpha_composite(panels[0].convert('RGBA'),overlay).convert('RGB')
    box=filling['target']['box_xyxy']
    if box:ImageDraw.Draw(panels[0]).rectangle(box,outline='#42b9f5',width=2)
    font_path='/usr/share/fonts/truetype/lato/Lato-Regular.ttf'
    font=ImageFont.truetype(font_path,18);small=ImageFont.truetype(font_path,14)
    draw=ImageDraw.Draw(panels[1])
    for row in fixture:
        color='#38d1bd' if row['proposal']['proposed_class']=='pass' else '#edb95d'
        draw.rectangle(row['target']['box_xyxy'],outline=color,width=1)
        draw.text(tuple(row['target']['box_xyxy'][:2]),row['target']['id'],fill=color,font=small)
    board=Image.new('RGB',(1328,608),'#101c2c');d=ImageDraw.Draw(board)
    d.text((16,12),'Filling camera | liquid proposal + unchanged QA',font=font,fill='white')
    d.text((688,12),'Final fixture | capped / empty model proposals',font=font,fill='white')
    board.paste(panels[0],(16,48));board.paste(panels[1],(688,48))
    d.text((16,540),f"Run4 native frame {filling['source']['frame_index']} | source-exact RGB",font=small,fill='white')
    d.text((688,540),f"{fixture[0]['target']['context_id']} | original source JPEG",font=small,fill='white')
    d.text((16,578),'Independent development examples. No cross-camera time join or physical-vial association. Predictions remain unreviewed.',font=small,fill='#f7ce80')
    board.save(out/'two-station-evidence.png')
    identity=dict(filling_observation_id=filling['observation_id'],fixture_observation_ids=[r['observation_id'] for r in fixture],
        filling_source_asset=sources['assets'][filling['observation_id']],fixture_source_asset=sources['assets'][fixture[0]['observation_id']],
        displayed_masks='original cached raw proposal; no new inference',physical_vials_linked=False,source_clocks_joined=False)
    (out/'evidence-board.json').write_text(json.dumps(identity,indent=2)+'\n')

    fields=['reference_id','purpose','camera_id','source_id','stream_id','profile_id','context_id','observation_id',
        'frame_index','pts','time_base','decoded_rgb_sha256','required_condition','confirmed_condition',
        'alignment_status','reviewer','evidence_id','status','notes']
    with (out/'reference-capture-worksheet.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for ref in refs:
            o=ref['observation'];s=o['source']
            writer.writerow(dict(reference_id=ref['reference_id'],purpose=ref['purpose'],camera_id=s['camera_id'],
                source_id=s['source_id'],stream_id=s['stream_id'],profile_id=o['profile_id'],context_id=o['target']['context_id'],
                observation_id=o['observation_id'],frame_index=s['frame_index'],pts=s['pts'],time_base=s['time_base'],
                decoded_rgb_sha256=s['decoded_rgb_sha256'],required_condition='empty',status='unreviewed',
                notes='Candidate only; do not infer emptiness from model absence. Capture a new reference if not confirmable.'))
        for purpose,condition in [('capped_golden','capped'),('recipe_endpoint','accepted_recipe_endpoint')]:
            writer.writerow(dict(purpose=purpose,required_condition=condition,status='not_captured',notes='New source capture, independent condition evidence and reviewed mapping required. V1 gate requires comparable native clocks.'))
    with (out/'physical-association-worksheet.csv').open('w',newline='') as stream:
        csv.writer(stream).writerow(['physical_vial_id','fill_observation_id','fixture_observation_id','reviewer','evidence_id','basis','notes'])
    # Keep blank decision fields: this is a capture plan, not a review import.
    shutil.copy2(root/'rehearsal-v2/report.json',out/'rehearsal-report.json')
    shutil.copy2(root/'evaluation-v1/evaluation-status.json',out/'evaluation-status.json')
    shutil.copy2(root/'geometry-v1/geometry-checks.json',out/'geometry-checks.json')
    protocol=Path(__file__).resolve().parents[2]/'replay/operator/COMMISSIONING.md'
    shutil.copy2(protocol,out/'COMMISSIONING.md')

    pdfmetrics.registerFont(TTFont('Lato',font_path))
    pdfmetrics.registerFont(TTFont('LatoBold','/usr/share/fonts/truetype/lato/Lato-Bold.ttf'))
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name='BodyL',fontName='Lato',fontSize=10.5,leading=15,spaceAfter=9,textColor=colors.HexColor('#253747')))
    styles.add(ParagraphStyle(name='TitleL',fontName='LatoBold',fontSize=25,leading=30,spaceAfter=14,textColor=colors.HexColor('#12344b')))
    styles.add(ParagraphStyle(name='HeadingL',fontName='LatoBold',fontSize=15,leading=19,spaceAfter=10,textColor=colors.HexColor('#12344b')))
    story=[]
    def p(text,style='BodyL'):story.append(Paragraph(text,styles[style]))
    def table(rows,widths):
        data=[[Paragraph(str(v),styles['BodyL']) for v in row] for row in rows]
        t=Table(data,colWidths=widths,hAlign='LEFT');t.setStyle(TableStyle([
            ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e7f0f5')),('VALIGN',(0,0),(-1,-1),'TOP'),
            ('LINEBELOW',(0,0),(-1,-1),.5,colors.HexColor('#d4dfe5')),('LEFTPADDING',(0,0),(-1,-1),8),
            ('RIGHTPADDING',(0,0),(-1,-1),8),('TOPPADDING',(0,0),(-1,-1),8)]));story.append(t);story.append(Spacer(1,12))
    p('Two stations. One evidence pipeline.','TitleL')
    p('Integration and commissioning rehearsal | October 23, 2026')
    p('The story: train visual components on synthetic scenes, inspect independent real camera evidence at filling and final placement, and give an operator traceable observations plus explicit uncertainty. The two views are not yet linked to the same physical vials.')
    table([['Stage','Candidate path'],['Filling camera','Synthetic-trained vial detector → temporal liquid mask → unchanged color-sampling QA'],
        ['Final fixture camera','Twin-derived target geometry → synthetic-trained capped / uncapped / empty CNN'],
        ['Common evidence','Source identity + clock + model/profile + proposal + QA + qualification reasons'],
        ['Operator','View evidence → request review → acknowledge → record response → retain follow-up']], [112,388])
    p(f"Executed: {report['observations']} cached observations across two source streams. The rehearsal is CPU-only integration verification, not live inference, camera-to-dashboard validation or new model accuracy.")
    p('This is a candidate contract for Srijan and Josiah/Hayden. Existing application defaults remain unchanged. No physical vial association, reference approval or equipment command was invented.')
    story.append(PageBreak())
    p('Evidence we can show today','TitleL')
    story.append(PDFImage(str(out/'two-station-evidence.png'),width=500,height=500*608/1328));story.append(Spacer(1,12))
    table([['Executed check','Result'],['Liquid evidence','189/264 mask proposals; 105/264 unchanged-QA color samples'],
        ['Fixture evidence','480 proposals: 41 capped-looking, 439 empty; no reviewed truth'],
        ['Reference and identity software guards',f"{report['guard_checks_passed']}/{report['guard_checks']} engineering cases passed"],
        ['Grid diagnostic','0/16 sampled stills flagged; both digitally shifted 8-pixel controls flagged'],
        ['Audit rehearsal','757 hash-chained demonstration events; two source-pinned requests/responses']], [210,290])
    p('These are evidence-availability counts and software checks. They do not establish recipe correctness, mask accuracy, missing-cap recall, physical calibration or product release.')
    p('Portable source coverage: 72 exact filling PNGs + 16 original fixture JPEGs provide pixels for 552/744 observations. Another 192 filling observations are metadata-only here. Strict local video extraction detected an RGB-byte mismatch and refused substitution; the cause is unresolved.')
    story.append(PageBreak())
    p('What finishes the actual test','TitleL')
    table([['Owner','Concrete next deliverable'],['Srijan','Wire model adapters into the candidate contract. Capture/reset temporal references on target or stream changes. Benchmark the complete live path.'],
        ['Josiah / Hayden','Separate evidence from qualification. Pin the requested source image while new frames arrive. Preserve response and follow-up history.'],
        ['Human reviewers','Review 72 liquid frames and 96 priority fixture targets (480 total). Confirm geometry/visibility; include actual positives and uncertain cases.'],
        ['Austin + CSI','Confirm approved feed/session, empty and capped references, acceptable intended-recipe endpoints, completion context and physical vial-to-slot evidence.']], [115,385])
    p('Included worksheets list six unreviewed empty-reference candidates plus uncaptured golden/recipe references. Human decision fields and the physical-association sheet are blank. They are planning sheets, not automatically importable approvals.')
    p('Evaluation is ready: supply actual exports to annotation_review.evaluation_bundle using --slot-review and/or --pool-review, with a fresh --out. It reuses source-matched scorers and rejects demonstration reviews. Current reviewed real accuracy remains unavailable.')
    p('Rehearsal reports, source limitations, reproduction commands and interface details are in COMMISSIONING.md. Existing demonstration videos remain independent development streams, not a synchronized production demonstration.')
    def footer(canvas,doc):
        canvas.setFont('Lato',8);canvas.setFillColor(colors.HexColor('#607080'))
        canvas.drawString(48,25,'AI Club | synthetic-trained vision → source-bound operator evidence | engineering rehearsal')
        canvas.drawRightString(562,25,str(doc.page))
    SimpleDocTemplate(str(out/'two-station-commissioning.pdf'),pagesize=(612,792),leftMargin=50,rightMargin=50,topMargin=44,bottomMargin=44).build(story,onFirstPage=footer,onLaterPages=footer)
    (out/'README.md').write_text('# Two-station commissioning handoff\n\nRead two-station-commissioning.pdf or COMMISSIONING.md.\n'
        'The PNG shows independent, source-exact development examples with frozen model overlays; it is not a same-vial join.\n'
        'Capture worksheet decisions are blank; association worksheet is header-only. These are planning sheets, not approvals.\n'
        'All real accuracy remains unavailable pending human exports. See evaluation-status.json.\n')
    manifest=dict(schema='commissioning-handoff/v1',report_source_sha256=sha(root/'rehearsal-v2/report.json'),
        displayed_source_identities=identity,real_accuracy=None,helper_sha256=sha(__file__),
        files={p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file()})
    (out/'completed.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(out=str(out),files=len(manifest['files']),guard_checks=report['guard_checks_passed'],real_accuracy=None)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();prepare(a.root,a.out)
