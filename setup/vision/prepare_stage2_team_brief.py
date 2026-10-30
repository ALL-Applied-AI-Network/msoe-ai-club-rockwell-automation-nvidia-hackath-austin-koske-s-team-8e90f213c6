"""Build a code-free Part 2 architecture handoff matching the Part 1 PDF style.

Document-building dependencies: ReportLab and PyMuPDF. No model inference,
dataset writes, network publication or equipment access.
"""
import argparse
import hashlib
import html
import json
from pathlib import Path
import re
import shutil
import zipfile

import pymupdf
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak


NAVY = '#142B40'
TEAL = '#087F83'
GRAY = '#566676'
PALE = '#EFF5F7'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def architecture(out):
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="1900" height="1370" viewBox="0 0 1900 1370">',
             '<rect width="1900" height="1370" fill="white"/>',
             '<defs><marker id="arrow" markerWidth="12" markerHeight="12" refX="9" refY="4" orient="auto"><path d="M0,0 L0,8 L10,4 z" fill="#5B7181"/></marker></defs>']
    def text(x, y, value, size=30, bold=False, color=GRAY):
        parts.append(f'<text x="{x}" y="{y}" font-family="Lato,Arial,sans-serif" font-size="{size}" font-weight="{700 if bold else 400}" fill="{color}">{html.escape(value)}</text>')
    def box(x, y, w, h, title, lines, tone='normal'):
        fill = '#EAF4F4' if tone == 'input' else '#FFF5E9' if tone == 'operator' else PALE
        border = TEAL if tone == 'input' else '#9B713B' if tone == 'operator' else '#526C87'
        parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="18" fill="{fill}" stroke="{border}" stroke-width="3"/>')
        text(x+22, y+45, title, 34, True, NAVY)
        for i, line in enumerate(lines):
            text(x+22, y+91+i*37, line)
    def arrow(points):
        parts.append(f'<polyline points="{points}" fill="none" stroke="#5B7181" stroke-width="4"/>')
        # Explicit polygons survive SVG rasterizers that omit marker-end.
        coords = [tuple(map(float, point.split(','))) for point in points.split()]
        (ax, ay), (x, y) = coords[-2:]
        if x == ax:
            back = y-18 if y > ay else y+18
            triangle = f'{x},{y} {x-10},{back} {x+10},{back}'
        else:
            back = x-18 if x > ax else x+18
            triangle = f'{x},{y} {back},{y-10} {back},{y+10}'
        parts.append(f'<polygon points="{triangle}" fill="#5B7181"/>')
    box(40, 30, 840, 180, 'PART 1 / GPU INFERENCE + CPU QA',
        ['Exact-frame vial, liquid and mask proposals', 'Sample feature + usable/unavailable + source ID'], 'input')
    box(1020, 30, 840, 180, 'CONTEXT / OPERATOR OR TEST SOURCE',
        ['Physical vial + recipe + human-confirmed completion', 'Phase / meters: simulation or qualified log exports'], 'input')
    box(40, 300, 840, 180, 'OBSERVED APPEARANCE / CPU',
        ['Compatible sampling / camera measurement profile', 'Raw baseline; guarded normalization is a candidate'])
    box(1020, 300, 840, 180, 'EXPECTED TRAJECTORY / CPU',
        ['Endpoint reference for confirmed completed fills', 'Phase / progress trajectory when evidence is available'])
    arrow('460,210 460,300'); arrow('1440,210 1440,300')
    box(500, 570, 900, 180, 'EVIDENCE ELIGIBILITY / CPU',
        ['Current clocks + physical identity + geometry / sample QA', 'Reference compatibility + camera health + provenance'])
    arrow('460,480 460,525 730,525 730,570')
    arrow('1440,480 1440,525 1170,525 1170,570')
    box(40, 880, 840, 155, 'UNAVAILABLE / REASON RECORDED',
        ['Missing, stale, ambiguous or incompatible evidence', 'A missing assessment cannot clear a latched review'])
    box(1020, 880, 840, 155, 'SEQUENTIAL COMPARISON / CPU',
        ['Same-vial / phase qualified evidence + persistence', 'Within band / collecting / review required'])
    arrow('730,750 730,810 460,810 460,880')
    arrow('1170,750 1170,810 1440,810 1440,880')
    text(50, 852, 'Missing or incompatible', 29)
    text(1470, 852, 'Qualified current evidence', 29)
    box(40, 1130, 840, 180, 'AUDIT & INDEPENDENT EVALUATION',
        ['Source / reference / event IDs + coverage and misses', 'Private truth and independent outcomes stay here'])
    box(1020, 1130, 840, 180, 'OPERATOR VIEW & RECOVERY / CPU',
        ['Review → human response → recorded outcome', 'No line I/O or database updates during hackathon'], 'operator')
    arrow('460,1035 460,1130'); arrow('1440,1035 1440,1130')
    arrow('880,955 970,955 970,1180 1020,1180')
    arrow('1020,1220 880,1220')
    text(40, 1352, 'Twin: metered trajectories + simulated actions. Physical mode: human response; no direct line I/O.', 29)
    parts.append('</svg>')
    svg = out/'stage2-architecture.svg'
    svg.write_text('\n'.join(parts)+'\n', encoding='utf-8')
    with pymupdf.open(svg) as doc:
        doc[0].get_pixmap(matrix=pymupdf.Matrix(1.2, 1.2)).save(out/'stage2-architecture.png')


def build_pdf(markdown, target, image):
    fonts = Path('/usr/share/fonts/truetype/lato')
    pdfmetrics.registerFont(TTFont('Lato', str(fonts/'Lato-Regular.ttf')))
    pdfmetrics.registerFont(TTFont('LatoBold', str(fonts/'Lato-Bold.ttf')))
    pdfmetrics.registerFontFamily('Lato', normal='Lato', bold='LatoBold', italic='Lato', boldItalic='LatoBold')
    styles = {
        'p': ParagraphStyle('p', fontName='Lato', fontSize=9.8, leading=13.1, textColor=colors.HexColor(NAVY), spaceAfter=8),
        'cell': ParagraphStyle('cell', fontName='Lato', fontSize=9.1, leading=11.7, textColor=colors.HexColor(NAVY)),
        'head': ParagraphStyle('head', fontName='LatoBold', fontSize=9.2, leading=12, textColor=colors.white),
        'h': ParagraphStyle('h', fontName='LatoBold', fontSize=12, leading=15.2, textColor=colors.HexColor(TEAL), spaceBefore=7, spaceAfter=7),
        'title': ParagraphStyle('title', fontName='LatoBold', fontSize=26, leading=29.5, textColor=colors.HexColor(NAVY), spaceAfter=8),
        'sub': ParagraphStyle('sub', fontName='Lato', fontSize=11.1, leading=14.3, textColor=colors.HexColor(GRAY), spaceAfter=12),
        'kicker': ParagraphStyle('kicker', fontName='LatoBold', fontSize=8.7, leading=12, textColor=colors.HexColor(TEAL), spaceAfter=9),
        'small': ParagraphStyle('small', fontName='Lato', fontSize=8.8, leading=11.7, textColor=colors.HexColor(GRAY), spaceAfter=7),
    }
    titles = [
        ('TEAM IMPLEMENTATION BRIEF / PART 2', 'Make inspection recipe-aware', 'From usable images to qualified operator advice'),
        ('SHARED INTERFACES', 'Connect observation and intent', 'Make clocks, identities and evidence mean the same thing'),
        ('EXPERIMENTS → DESIGN CHOICES', 'What we learned and selected', 'Measured development evidence, with its limits'),
        ('THREE IMPLEMENTATION WORKSTREAMS', 'Divide the work and integrate', 'Proposed workstreams; existing Part 1 assignments continue'),
        ('INTEGRATION & EVALUATION', 'Test the complete behavior', 'Digital twin first; independently documented physical evidence next'),
    ]
    def inline(s):
        s = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', s)
        s = html.escape(s)
        return re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', s).replace('`', '')
    def paragraph(s, style='p'):
        return Paragraph(inline(s), styles[style])
    story = []
    sections = re.split(r'(?m)^## ', markdown)[1:]
    if len(sections) != 5:
        raise ValueError('Expected the five reviewed architecture sections')
    for page, section in enumerate(sections):
        if page:
            story.append(PageBreak())
        story.extend([paragraph(titles[page][0], 'kicker'), paragraph(titles[page][1], 'title'), paragraph(titles[page][2], 'sub')])
        lines = section.splitlines()[1:]; i = 0
        while i < len(lines):
            line = lines[i].strip(); i += 1
            if not line:
                continue
            if line.startswith('### '):
                story.append(paragraph(line[4:], 'h'))
            elif line.startswith('!['):
                with pymupdf.open(image) as doc:
                    width, height = doc[0].rect.width, doc[0].rect.height
                display_width = 496
                story.extend([Image(str(image), width=display_width, height=display_width*height/width), Spacer(1, 9)])
            elif line.startswith('|'):
                group = [line]
                while i < len(lines) and lines[i].strip().startswith('|'):
                    group.append(lines[i].strip()); i += 1
                data = [[cell.strip() for cell in ln.strip('|').split('|')]
                        for ln in group if not re.fullmatch(r'[| \-:]+', ln)]
                data = [[paragraph(cell, 'head' if row == 0 else 'cell') for cell in cells]
                        for row, cells in enumerate(data)]
                widths = [310, 202] if page == 2 else [113, 399] if page == 1 else [125, 387]
                table = Table(data, colWidths=widths, repeatRows=1)
                table.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), colors.HexColor(NAVY)),
                    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor(PALE)]),
                    ('VALIGN', (0, 0), (-1, -1), 'TOP'), ('LEFTPADDING', (0, 0), (-1, -1), 8),
                    ('RIGHTPADDING', (0, 0), (-1, -1), 8), ('TOPPADDING', (0, 0), (-1, -1), 6),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 6)]))
                story.extend([table, Spacer(1, 10)])
            else:
                content = line
                while i < len(lines) and lines[i].strip() and not re.match(r'^(### |\||!\[)', lines[i].strip()):
                    content += ' '+lines[i].strip(); i += 1
                story.append(paragraph(content, 'small' if content.startswith('**Evidence navigation:') else 'p'))
    def footer(canvas, doc):
        canvas.setStrokeColor(colors.HexColor('#D4E0E7')); canvas.line(50, 39, 562, 39)
        canvas.setFont('Lato', 8); canvas.setFillColor(colors.HexColor(GRAY))
        canvas.drawString(50, 26, 'Austin Koske / Innovation Lab / Part 2 architecture / October 21, 2026')
        canvas.drawRightString(562, 26, str(doc.page))
    SimpleDocTemplate(str(target), pagesize=letter, leftMargin=50, rightMargin=50,
        topMargin=44, bottomMargin=51, title='Part 2 — Recipe-Aware Inspection / Team Implementation Brief',
        author='Austin Koske').build(story, onFirstPage=footer, onLaterPages=footer)


def main(a):
    if a.out.exists() or a.out.with_suffix('.zip').exists():
        raise ValueError('Use a fresh handoff output; preserve prior distributed versions')
    repo = Path(__file__).resolve().parents[2]
    source = repo/'team_handoff/part2/README.md'
    a.out.mkdir(parents=True)
    architecture(a.out)
    text = source.read_text(encoding='utf-8')
    shutil.copy2(source, a.out/'Part-2-Architecture-Implementation-Brief.md')
    clarification = source.parent/'OPERATOR-CLARIFICATION.md'
    if clarification.exists():
        shutil.copy2(clarification, a.out/clarification.name)
    build_pdf(text, a.out/'Part-2-Architecture-Implementation-Brief.pdf', a.out/'stage2-architecture.png')
    # Deterministic repository-native diagram; no large model/data files are copied.
    for name in ['stage2-architecture.svg', 'stage2-architecture.png']:
        target = source.parent/name
        if target.exists() and not a.update_diagram:
            raise ValueError('Preserve prior diagram: '+str(target))
        shutil.copy2(a.out/name, target)
    (a.out/'READ-ME.txt').write_text(
        'PART 2 / RECIPE-AWARE INSPECTION\n'
        'Start with Part-2-Architecture-Implementation-Brief.pdf.\n'
        'Read OPERATOR-CLARIFICATION.md for the confirmed hackathon access boundary.\n'
        'This code-free package contains architecture, module contracts, experimental\n'
        'evidence and implementation workstreams. It includes no implementation source,\n'
        'weights, training data, approved labels or solution traces.\n'
        'The Markdown copy is editable. Its repository-relative evidence links resolve\n'
        'from team_handoff/part2/ in the development checkout; they are not ZIP-local files.\n'
        'Existing Part 1 assignments continue; Part 2 owners are to be agreed.\n'
        'Real recipe-defect accuracy and production acceptance limits are not established.\n'
        'No direct existing line sensor/actuator access or Teledyne database updates.\n'
        'Use approved video/files and attributed operator context. PLC integration\n'
        'is a post-hackathon possibility. Missing meter/phase evidence stays unavailable.\n'
        'The proposed Oct 26–28 CSI session and video arrangement remain unconfirmed.\n'
        'No GitHub publication or industrial connection is performed by this package.\n', encoding='utf-8')
    proof = dict(schema='stage2-team-architecture-handoff/v1', prepared='2026-10-21',
        source_sha256=sha(source), builder_sha256=sha(Path(__file__)), code_free=True,
        published=False, training=False, hardware_connections=0,
        evidence_report_sha256={name: sha(repo/'setup/verification'/name) for name in [
            'stage2-refinement-20261020.json', 'twin-experiments-20261021.json',
            'pipeline-acceleration-20261020.json', 'cosmos-transfer-20261021.json',
            'csi-preparation-20261021.json']},
        artifacts={p.name: sha(p) for p in a.out.iterdir() if p.is_file()})
    (a.out/'package-hashes.json').write_text(json.dumps(proof, indent=2)+'\n', encoding='utf-8')
    with zipfile.ZipFile(a.out.with_suffix('.zip'), 'x', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(a.out.iterdir()):
            if path.is_file():
                archive.write(path, path.name)
    print(json.dumps(dict(pdf=str(a.out/'Part-2-Architecture-Implementation-Brief.pdf'),
        zip_sha256=sha(a.out.with_suffix('.zip')), files=len(proof['artifacts'])+1), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--update-diagram', action='store_true', help='Replace generated repository diagram; prior share packages remain intact')
    main(parser.parse_args())
