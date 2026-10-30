"""Build the local CSI visit PDF/forms/ZIP; does not send or book anything.

PDF generation requires ReportLab in a separate document-building environment.
The included capture inspector uses the existing PyAV data environment.
"""
import argparse
import csv
import hashlib
import html
import json
from pathlib import Path
import re
import shutil
import zipfile

from inspect_csi_capture import sha


FORMS = {
    'session': 'session_id,confirmed_date,confirmed_start_central,duration_minutes,booking_status,attendees,csi_operator,camera_id,camera_settings_file,approved_feed_or_export_arrangement,acceptance_criterion_file,notes',
    'vials': 'session_id,sequence_id,physical_vial_id,sequence_role,intended_recipe_id,prior_recipe_id,conditioning_evidence_file,operator,notes',
    'process': 'process_time,clock_id,physical_vial_id,intended_recipe_id,phase_id,tank_1_cumulative_ml,tank_2_cumulative_ml,tank_3_cumulative_ml,dispense_complete,meter_method,meter_uncertainty_ml,raw_log_file,notes',
    'clock-anchors': 'anchor_id,video_file,video_sha256,decoded_frame,video_pts,video_time_base,process_time,process_clock_id,landmark_description,evidence_file,mapping_uncertainty_ms,reviewer,review_status,notes',
    'outcomes': 'physical_vial_id,intended_recipe_id,independent_outcome,acceptance_criterion_id,criterion_version,measurement_method,measurement_uncertainty,evidence_file,evidence_sha256,reviewer,decision_time,decision_clock_id,notes',
    'inspection-routing': 'physical_vial_id,inspection_color_result,inspection_weight_result,inspection_clock_id,inspection_time,physical_disposition,routing_deadline,routing_clock_id,identity_link_status,raw_evidence_file,notes',
    'interventions': 'sequence_id,procedure_id,approved_by,operator,request_time,ack_time,completion_time,clock_id,actual_settings,settings_units,conditioning_evidence_file,receipt_file,receipt_sha256,notes',
    'issues': 'session_id,sequence_id,physical_vial_id,time,clock_id,issue_type,description,affected_evidence,operator_or_reviewer,resolution,notes',
}


def pdf(markdown, target):
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Preformatted
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='Text', fontName='Helvetica', fontSize=9.5, leading=13, spaceAfter=7))
    styles.add(ParagraphStyle(name='Cell', fontName='Helvetica', fontSize=8, leading=10, spaceAfter=2))
    styles['Title'].fontSize = 20; styles['Title'].leading = 24
    styles['Heading2'].fontSize = 13; styles['Heading2'].spaceBefore = 14
    def inline(s):
        s = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', s)
        s = html.escape(s)
        s = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', s)
        return s.replace('`', '')
    story = []; lines = markdown.splitlines(); i = 0
    while i < len(lines):
        line = lines[i]; i += 1
        if not line.strip():
            continue
        if line.startswith('# '):
            story.append(Paragraph(inline(line[2:]), styles['Title']))
        elif line.startswith('## '):
            story.append(Paragraph(inline(line[3:]), styles['Heading2']))
        elif line.startswith('|'):
            table_lines = [line]
            while i < len(lines) and lines[i].startswith('|'):
                table_lines.append(lines[i]); i += 1
            rows = [[Paragraph(inline(cell.strip()), styles['Cell']) for cell in ln.strip('|').split('|')]
                    for ln in table_lines if not re.fullmatch(r'[| :\-]+', ln)]
            columns = len(rows[0]); width = 468
            widths = [80, 175, 213] if columns == 3 else [140, 328]
            table = Table(rows, colWidths=widths, repeatRows=1, hAlign='LEFT')
            table.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e4edf5')),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'), ('GRID', (0, 0), (-1, -1), .3, colors.HexColor('#c5d2df')),
                ('LEFTPADDING', (0, 0), (-1, -1), 6), ('RIGHTPADDING', (0, 0), (-1, -1), 6),
                ('TOPPADDING', (0, 0), (-1, -1), 5), ('BOTTOMPADDING', (0, 0), (-1, -1), 5)]))
            story.extend([table, Spacer(1, 8)])
        elif line.startswith('```'):
            block = []
            while i < len(lines) and not lines[i].startswith('```'):
                # Long local paths wrap in prose to keep them within page margins.
                block.append(lines[i]); i += 1
            i += 1
            story.append(Paragraph('<br/>'.join(inline(s) for s in block), styles['Cell']))
        else:
            paragraph = line
            while i < len(lines) and lines[i].strip() and not re.match(r'^(#|\||```|- |\d+\. )', lines[i]):
                paragraph += ' '+lines[i].strip(); i += 1
            story.append(Paragraph(inline(paragraph), styles['Text']))
    def footer(canvas, doc):
        canvas.setFont('Helvetica', 8)
        canvas.drawString(72, 30, 'CSI visit preparation | Oct 21, 2026 | Draft: booking and access unconfirmed')
        canvas.drawRightString(540, 30, str(doc.page))
    SimpleDocTemplate(str(target), pagesize=(612, 792), rightMargin=72, leftMargin=72,
        topMargin=44, bottomMargin=50, title='CSI Visit Preparation', author='Austin Koske team').build(
            story, onFirstPage=footer, onLaterPages=footer)


def main(a):
    if a.out.exists() or a.out.with_suffix('.zip').exists():
        raise ValueError('Preserve existing packets; use a fresh path')
    repo = Path(__file__).resolve().parents[2]
    a.out.mkdir(parents=True); (a.out/'forms').mkdir()
    packet = repo/'setup/vision/CSI-SESSION-PACKET.md'
    for source in [packet, repo/'setup/vision/CSI-VALIDATION-PLAN.md',
                   repo/'docs/csi-session-request-2026-10-26.md',
                   repo/'annotation_review/EARLY-FILL-HANDOFF.md',
                   Path(__file__).with_name('inspect_csi_capture.py'), Path(__file__)]:
        shutil.copy2(source, a.out/source.name)
    for name, columns in FORMS.items():
        with (a.out/'forms'/f'{name}.csv').open('w', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow(columns.split(','))
    pdf(packet.read_text(encoding='utf-8'), a.out/'CSI-Visit-Preparation.pdf')
    baseline = a.tools/'results/pipeline-acceleration-20261020/full-run4-v1'
    plan = json.loads((baseline/'plan.json').read_text())
    metadata = json.loads((baseline/'completed.json').read_text())
    frozen = {k: plan[k] for k in ['frozen_models', 'qa_config', 'qa_code_sha256', 'vial_options', 'prompt_policy', 'bridge_policy']}
    frozen.update(software={k: metadata[k] for k in ['torch', 'ultralytics', 'tensorrt']},
                  baseline_backend='pytorch', baseline_completion_sha256=sha(baseline/'completed.json'),
                  camera_correction_promoted=False, tensorrt_promoted=False, hardware_commands=0)
    (a.out/'freeze-manifest.json').write_text(json.dumps(frozen, indent=2)+'\n')
    review = a.tools/'results/pipeline-acceleration-20261020/early-fill-review-v1.zip'
    if sha(review) != '4cfba5c723943b62edd64d92aa9486c4bef52dd5e202f41da17bd6c06951767d':
        raise ValueError('Original early-fill review archive changed')
    info = dict(schema='csi-visit-preparation/v1', prepared='2026-10-21', proposed_window='2026-10-26 through 2026-10-28',
                booking_status='unconfirmed', request_sent=False, hardware_connections=0,
                early_fill_review_archive=dict(path=str(review), sha256=sha(review), approved_onsets=0, approved_masks=0),
                artifacts={p.relative_to(a.out).as_posix(): sha(p) for p in a.out.rglob('*') if p.is_file()})
    (a.out/'package-hashes.json').write_text(json.dumps(info, indent=2)+'\n')
    with zipfile.ZipFile(a.out.with_suffix('.zip'), 'x', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(a.out.rglob('*')):
            if path.is_file():
                archive.write(path, path.relative_to(a.out))
    print(json.dumps(dict(packet=str(a.out), zip_sha256=sha(a.out.with_suffix('.zip')), files=len(info['artifacts'])+1), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tools', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    main(parser.parse_args())
