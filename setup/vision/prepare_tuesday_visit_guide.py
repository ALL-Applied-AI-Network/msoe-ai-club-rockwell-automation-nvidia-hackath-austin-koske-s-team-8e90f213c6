"""Render Tuesday's visit guide and blank field sheets; no deployment or booking."""
import argparse
import csv
import hashlib
import html
import json
from pathlib import Path
import re
import shutil
import zipfile


FORMS={
    'session':'session_id,date,start_local,end_local,timezone,booking_status,CSI_supervisor,assigned_host,approved_feed_method,approval_reference,camera_id,owner,notes',
    'host-and-launch':'session_id,host,OS,CPU_architecture,GPU_name,GPU_memory,driver_version,driver_reported_CUDA,python_executable,python_version,torch_version,torch_CUDA_runtime,ultralytics_version,decoder_version,source_manifest_sha256,model_manifest_sha256,profile_sha256,start_command,stop_command,rollback_procedure,owner,smoke_result,notes',
    'camera-handoff':'session_id,camera_id,priority,CSI_provider,approved_method,source_descriptor_no_secrets,recording_permission,original_width,original_height,orientation,source_clock_kind,native_rate_metadata,clock_mapping_id,adapter_owner,host,profile_id,raw_recording_path,notes',
    'references':'session_id,reference_id,purpose,source_file,source_sha256,frame_index,pts,time_base,decoded_RGB_sha256,camera_id,stream_id,physical_vial_id,profile_id,confirmed_condition,reviewer,evidence_id,qualification_status,notes',
    'trials':'session_id,trial_id,role,source_file,source_sha256,physical_vial_id,previous_recipe,intended_recipe,actual_tank_contents,completion_evidence,source_start_pts,source_end_pts,time_base,profile_id,model_manifest_sha256,observations_file,operator_record_file,independent_outcome_id,notes',
    'operator-responses':'session_id,request_id,source_id,frame_index,pts,time_base,physical_vial_id,intended_recipe,request_time,acknowledgement_time,response_time,clock_id,operator,approved_procedure_id,actual_action,independent_outcome_id,evidence_file,notes',
    'latency':'session_id,trial_id,source_id,frame_index,native_pts,native_time_base,receive_monotonic_ns,inference_start_monotonic_ns,inference_end_monotonic_ns,publish_monotonic_ns,display_ack_received_monotonic_ns,host_clock_id,clock_mapping_id,measurement_scope,source_to_screen_ms_or_blank,frame_dropped,source_age_ms_or_blank,notes',
    'independent-outcomes':'session_id,outcome_id,physical_vial_id,intended_recipe,acceptance_criterion_id,method,independent_result,reviewer,evidence_file,evidence_sha256,uncertainty,decision_time,clock_id,notes',
    'issues':'session_id,trial_id,source_id,time,clock_id,issue_type,description,affected_artifacts,owner,resolution,notes',
}


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def render(markdown,target):
    from reportlab.lib import colors
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,PageBreak
    fontdir=Path('/usr/share/fonts/truetype/lato')
    pdfmetrics.registerFont(TTFont('VisitLato',str(fontdir/'Lato-Regular.ttf')))
    pdfmetrics.registerFont(TTFont('VisitLatoBold',str(fontdir/'Lato-Bold.ttf')))
    pdfmetrics.registerFontFamily('VisitLato',normal='VisitLato',bold='VisitLatoBold',italic='VisitLato',boldItalic='VisitLatoBold')
    navy='#142B40';teal='#087F83'
    body=ParagraphStyle('VisitBody',fontName='VisitLato',fontSize=10.3,leading=13.4,spaceAfter=8,textColor=colors.HexColor(navy))
    cell=ParagraphStyle('VisitCell',parent=body,fontSize=9.2,leading=11.8,spaceAfter=0)
    title=ParagraphStyle('VisitTitle',parent=body,fontName='VisitLatoBold',fontSize=22,leading=26,spaceAfter=11)
    heading=ParagraphStyle('VisitHeading',parent=body,fontName='VisitLatoBold',fontSize=15,leading=19,spaceAfter=11,textColor=colors.HexColor(teal))
    def inline(text):
        links=[]
        def replace(m):
            links.append(m.groups());return f'VISITLINKTOKEN{len(links)-1}TOKENEND'
        text=re.sub(r'\[([^\]]+)\]\(([^)]+)\)',replace,text)
        text=html.escape(text);text=re.sub(r'\*\*(.*?)\*\*',r'<b>\1</b>',text);text=text.replace('`','')
        for i,(label,url) in enumerate(links):
            label=html.escape(label)
            replacement=f'<link href="{html.escape(url,quote=True)}" color="{teal}">{label}</link>' if '://' in url else label
            text=text.replace(f'VISITLINKTOKEN{i}TOKENEND',replacement)
        return text
    lines=markdown.splitlines();story=[];i=0
    while i<len(lines):
        line=lines[i];i+=1
        if not line.strip():continue
        if line=='<!-- PAGEBREAK -->':story.append(PageBreak());continue
        if line.startswith('# '):story.append(Paragraph(inline(line[2:]),title));continue
        if line.startswith('## '):story.append(Paragraph(inline(line[3:]),heading));continue
        if line.startswith('```'):
            block=[]
            while i<len(lines) and not lines[i].startswith('```'):block.append(lines[i]);i+=1
            i+=1
            story.append(Paragraph('<br/>'.join(inline(s).replace(' ','&#160;') for s in block),cell));story.append(Spacer(1,8));continue
        if line.startswith('|'):
            raw=[line]
            while i<len(lines) and lines[i].startswith('|'):raw.append(lines[i]);i+=1
            rows=[[Paragraph(inline(c.strip()),cell) for c in row.strip('|').split('|')] for row in raw if not re.fullmatch(r'[| :\-]+',row)]
            columns=len(rows[0]);widths={2:[120,396],3:[66,225,225]}[columns]
            if 'Ask CSI before' in raw[0]:widths=[225,291]
            table=Table(rows,colWidths=widths,repeatRows=1,hAlign='LEFT')
            table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#E8F1F3')),('VALIGN',(0,0),(-1,-1),'TOP'),
                ('LINEBELOW',(0,0),(-1,-1),.35,colors.HexColor('#D5E0E5')),('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),
                ('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6)]))
            story.extend([table,Spacer(1,9)]);continue
        paragraph=line
        while i<len(lines) and lines[i].strip() and not lines[i].startswith(('#','|','```','<!-- PAGEBREAK')):
            paragraph+=' '+lines[i].strip();i+=1
        story.append(Paragraph(inline(paragraph),body))
    def footer(canvas,document):
        canvas.setStrokeColor(colors.HexColor('#D5E0E5'));canvas.line(48,39,564,39)
        canvas.setFont('VisitLato',7.5);canvas.setFillColor(colors.HexColor('#566676'))
        canvas.drawString(48,27,'CSI / Oct 27 target visit / time, host and camera arrangement pending confirmation')
        canvas.drawRightString(564,27,str(document.page))
    SimpleDocTemplate(str(target),pagesize=(612,792),leftMargin=48,rightMargin=48,topMargin=38,bottomMargin=49,
        title='CSI Tuesday Visit Guide — October 27, 2026',author='Austin Koske team').build(story,onFirstPage=footer,onLaterPages=footer)


def prepare(out):
    import pymupdf
    from PIL import Image,ImageDraw
    repo=Path(__file__).resolve().parents[2];guide=repo/'team_handoff/integration/CSI-VISIT-20261027.md'
    out=out.resolve();archive=out.with_suffix('.zip')
    if out.exists() or archive.exists():raise ValueError('Preserve existing guide output; use a fresh directory')
    out.mkdir(parents=True);shutil.copy2(guide,out/'CSI-Visit-Guide.md')
    pdf=out/'CSI-Tuesday-Visit-Guide.pdf';render(guide.read_text(),pdf)
    forms=out/'forms';forms.mkdir()
    for name,fields in FORMS.items():
        with (forms/(name+'.csv')).open('w',newline='',encoding='utf-8') as stream:csv.writer(stream).writerow(fields.split(','))
    checklist=['Supervisor/time/feed confirmed','Host and saved-clip smoke recorded','Original pilot recording decoded/clock checked',
        'Reference condition/alignment reviewed or explicitly unavailable','Prospective model/profile/settings frozen',
        'Failed-QA cannot display normal recipe status','Every attempted trial/issue retained','Operator request/response source-pinned',
        'Actual latency scope and clock assumptions recorded','Raw/mask/log joins spot-checked','Approved backup copy hash-verified','Live/replay/cached demo mode labeled']
    with (forms/'exit-checklist.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.writer(stream);writer.writerow(['check','completed_by','status','evidence_file','notes']);writer.writerows([item,'','','',''] for item in checklist)
    manifest=dict(schema='csi-tuesday-guide/v1',prepared='2026-10-24',target_date='2026-10-27',booking_status='unconfirmed',assigned_host_status='unconfirmed',
        published_host='Windows11 / RTX PRO5000 Blackwell; confirm assigned unit',feed_status='unconfirmed',
        guide_only=True,runtime_included=False,weights_included=False,camera_connector_included=False,CSI_host_tested=False,
        model_identities=dict(liquid='6714fe32191d8eba230aa87246f38a25c2906c619d60d7ba30c9e60205352eef',vial='0bfac07a256eaf190416b632ad1f094a6b534309df1bb11a8c175e9d811b56bc'),
        runtime_launch_status='Integrated U-Net start/stop procedure remains a Monday deliverable; older launchers/cached demo are distinct',
        source_guide_sha256=sha(guide),training_started=False,hardware_connections=0)
    (out/'visit-status.json').write_text(json.dumps(manifest,indent=2)+'\n')
    previews=out/'previews';previews.mkdir();doc=pymupdf.open(pdf);thumbs=[];pagechecks=[]
    for i,page in enumerate(doc):
        pix=page.get_pixmap(matrix=pymupdf.Matrix(1.4,1.4),alpha=False);pix.save(previews/f'page-{i+1}.png')
        im=Image.frombytes('RGB',(pix.width,pix.height),pix.samples);im.thumbnail((408,528));thumbs.append(im)
        text=page.get_text();pagechecks.append(dict(page=i+1,characters=len(text),first_lines=text.splitlines()[:2]))
    board=Image.new('RGB',(3*428,((len(thumbs)+2)//3)*558),'#DAE3E8');draw=ImageDraw.Draw(board)
    for i,im in enumerate(thumbs):
        x=(i%3)*428+10;y=(i//3)*558+20;board.paste(im,(x,y));draw.text((x,y-15),f'Page{i+1}',fill='black')
    board.save(previews/'contact-sheet.png')
    (out/'render-checks.json').write_text(json.dumps(dict(pages=len(doc),pages_detail=pagechecks),indent=2)+'\n')
    # Include the portable guide/forms only; previews and build dependencies stay local.
    files=[p for p in sorted(out.rglob('*')) if p.is_file() and not p.is_relative_to(previews)]
    (out/'completed.json').write_text(json.dumps(dict(artifacts={str(p.relative_to(out)):sha(p) for p in files},PDF_pages=len(doc),
        source_unchanged=sha(guide)==manifest['source_guide_sha256'],guide_only=True,hardware_connections=0),indent=2)+'\n')
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as bundle:
        for p in files+[out/'completed.json']:bundle.write(p,p.relative_to(out))
    print(json.dumps(dict(out=str(out),PDF=str(pdf),pages=len(doc),forms=len(FORMS)+1,archive=str(archive),archive_sha256=sha(archive)),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True);prepare(parser.parse_args().out)
