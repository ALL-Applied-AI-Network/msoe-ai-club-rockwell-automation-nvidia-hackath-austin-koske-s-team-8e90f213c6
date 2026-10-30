"""Build the code-free integration brief in the original student handoff style."""
from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path
import zipfile

import pymupdf
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

NAVY = '#142B40'
TEAL = '#087F83'
GRAY = '#566676'
PALE = '#EFF5F7'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def diagram(out):
    pieces = ['<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="875" viewBox="0 0 1600 875">', '<rect width="1600" height="875" fill="white"/>']

    def text(x, y, value, size=25, bold=False):
        pieces.append(f'<text x="{x}" y="{y}" font-family="Lato,Arial,sans-serif" font-size="{size}" font-weight="{700 if bold else 400}" fill="{NAVY if bold else GRAY}">{html.escape(value)}</text>')

    def box(x, y, w, h, title, lines, gpu=False):
        pieces.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="15" fill="{"#EAF4F4" if gpu else PALE}" stroke="{TEAL if gpu else "#526C87"}" stroke-width="2.5"/>')
        text(x+18, y+35, title, 28, True)
        for i, line in enumerate(lines):
            text(x+18, y+74+31*i, line)

    def arrow(coords):
        points = ' '.join(f'{x},{y}' for x, y in coords)
        pieces.append(f'<polyline points="{points}" fill="none" stroke="#5B7181" stroke-width="3.5"/>')
        (ax, ay), (x, y) = coords[-2:]
        if x == ax:
            back = y-14 if y > ay else y+14
            triangle = f'{x},{y} {x-8},{back} {x+8},{back}'
        else:
            back = x-14 if x > ax else x+14
            triangle = f'{x},{y} {back},{y-8} {back},{y+8}'
        pieces.append(f'<polygon points="{triangle}" fill="#5B7181"/>')

    box(40, 15, 1520, 100, 'SHARED FRAME / CPU', ['Approved camera or original file; Isaac RGB for testing; source ID + native PTS'])
    box(40, 170, 720, 130, 'TEAM 1 / YOLO11n / GPU', ['Active-vial geometry', 'Frame ID + box + uncertainty'], True)
    box(840, 170, 720, 130, 'TEAM 2 / FASTER R-CNN / GPU', ['Visible accumulated-liquid localization', 'Frame ID + regions + scores'], True)
    arrow([(800, 115), (800, 140), (400, 140), (400, 170)])
    arrow([(800, 140), (1200, 140), (1200, 170)])
    box(40, 355, 1520, 130, 'TEAM 3 + INTEGRATION / CPU ASSOCIATION + GPU SAM2.1', ['Same-frame proposals + prompt provenance -> full-image pool-mask proposal', 'Missing or ambiguous evidence remains explicit'])
    arrow([(400, 300), (400, 327), (800, 327), (800, 355)])
    arrow([(1200, 300), (1200, 327), (800, 327)])
    box(40, 540, 720, 130, 'SAMPLING & EVIDENCE / CPU', ['QA -> usable feature or unavailable reason', 'Exact-frame image, mask and source identity'])
    box(840, 540, 720, 130, 'CONTEXT & REFERENCES / CPU', ['Physical vial + intended recipe + completion', 'Compatible acceptable reference or unavailable'])
    arrow([(800, 485), (800, 512), (400, 512), (400, 540)])
    box(40, 725, 1520, 125, 'PART 2 / OPERATOR RESPONSE & AUDIT / CPU', ['Qualified comparison -> evidence/review state -> attributed human response', 'Record outcomes; existing line sensors, actuators and Teledyne remain outside the interface'])
    arrow([(400, 670), (400, 697), (800, 697), (800, 725)])
    arrow([(1200, 670), (1200, 697), (800, 697)])
    pieces.append('</svg>')
    svg = out/'team-integration-architecture.svg'
    svg.write_text('\n'.join(pieces)+'\n', encoding='utf-8')
    with pymupdf.open(svg) as doc:
        doc[0].get_pixmap(matrix=pymupdf.Matrix(1.3, 1.3)).save(out/'team-integration-architecture.png')


def build(source, out):
    spec = json.loads(source.read_text(encoding='utf-8'))
    fonts = Path('/usr/share/fonts/truetype/lato')
    pdfmetrics.registerFont(TTFont('Lato', str(fonts/'Lato-Regular.ttf')))
    pdfmetrics.registerFont(TTFont('LatoBold', str(fonts/'Lato-Bold.ttf')))
    styles = {
        'p': ParagraphStyle('body', fontName='Lato', fontSize=10, leading=13.5, textColor=colors.HexColor(NAVY), spaceAfter=8),
        'small': ParagraphStyle('small', fontName='Lato', fontSize=8.6, leading=11.4, textColor=colors.HexColor(GRAY), spaceAfter=7),
        'h': ParagraphStyle('heading', fontName='LatoBold', fontSize=12.3, leading=15, textColor=colors.HexColor(TEAL), spaceBefore=6, spaceAfter=7),
        'cell': ParagraphStyle('cell', fontName='Lato', fontSize=9.1, leading=12, textColor=colors.HexColor(NAVY)),
        'head': ParagraphStyle('tablehead', fontName='LatoBold', fontSize=9.1, leading=12, textColor=colors.white),
        'title': ParagraphStyle('title', fontName='LatoBold', fontSize=26, leading=30, textColor=colors.HexColor(NAVY), spaceAfter=9),
        'sub': ParagraphStyle('subtitle', fontName='Lato', fontSize=11.3, leading=15, textColor=colors.HexColor(GRAY), spaceAfter=12),
        'kicker': ParagraphStyle('kicker', fontName='LatoBold', fontSize=8.7, leading=12, textColor=colors.HexColor(TEAL), spaceAfter=9),
    }

    def p(value, style='p'):
        return Paragraph(html.escape(str(value)), styles[style])

    story = []
    plain = [spec['title'], 'Prepared October 22, 2026; readiness target Monday, October 26.', '']
    for i, page in enumerate(spec['pages']):
        if i:
            story.append(PageBreak())
        story.extend([p(page['kicker'], 'kicker'), p(page['title'], 'title'), p(page['sub'], 'sub')])
        plain.extend([f"PAGE {i+1}: {page['title']}", page['sub'], ''])
        for block in page['blocks']:
            if block[0] in ('p', 'small', 'h'):
                story.append(p(block[1], block[0]))
                plain.extend([block[1], ''])
            elif block[0] == 'diagram':
                image = out/'team-integration-architecture.png'
                with pymupdf.open(image) as doc:
                    width, height = doc[0].rect.width, doc[0].rect.height
                story.extend([Image(str(image), width=512, height=512*height/width), Spacer(1, 8)])
                plain.extend(['FRAME -> YOLO vial + Faster R-CNN liquid -> SAM2.1 mask -> sample QA -> qualified recipe/reference comparison -> operator evidence and response.', ''])
            elif block[0] == 'table':
                _, heads, rows, widths = block
                table = Table([[p(v, 'head') for v in heads]]+[[p(v, 'cell') for v in row] for row in rows], colWidths=widths, repeatRows=1)
                table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor(NAVY)),
                    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor(PALE)]),
                    ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                    ('LEFTPADDING', (0, 0), (-1, -1), 8), ('RIGHTPADDING', (0, 0), (-1, -1), 8),
                    ('TOPPADDING', (0, 0), (-1, -1), 6), ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                ]))
                story.extend([table, Spacer(1, 9)])
                plain.extend([' | '.join(heads), *[' | '.join(row) for row in rows], ''])

    def footer(canvas, doc):
        canvas.setStrokeColor(colors.HexColor('#D4E0E7'))
        canvas.line(50, 39, 562, 39)
        canvas.setFont('Lato', 8)
        canvas.setFillColor(colors.HexColor(GRAY))
        canvas.drawString(50, 26, 'Austin Koske / Innovation Lab / Integration brief / October 22, 2026')
        canvas.drawRightString(562, 26, str(doc.page))

    pdf = out/'Team-Integration-and-Physical-Test-Brief.pdf'
    SimpleDocTemplate(str(pdf), pagesize=letter, leftMargin=50, rightMargin=50, topMargin=44, bottomMargin=51, title=spec['title'], author='Austin Koske').build(story, onFirstPage=footer, onLaterPages=footer)
    (out/'Team-Integration-and-Physical-Test-Brief.txt').write_text('\n'.join(plain), encoding='utf-8')
    with pymupdf.open(pdf) as doc:
        if len(doc) != len(spec['pages']):
            raise ValueError(f'Layout overflow: expected {len(spec["pages"])} pages, found {len(doc)}')
        for i, page in enumerate(doc):
            if spec['pages'][i]['title'] not in page.get_text():
                raise ValueError(f'Unexpected page break at page {i+1}')
    return pdf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    archive = out.with_suffix('.zip')
    if out.exists() or archive.exists():
        raise ValueError('Use a fresh output directory; preserve distributed editions')
    repo = Path(__file__).resolve().parents[2]
    source = repo/'team_handoff/integration/brief-20261022.json'
    out.mkdir(parents=True)
    diagram(out)
    pdf = build(source, out)
    (out/'READ-ME.txt').write_text(
        'Start with Team-Integration-and-Physical-Test-Brief.pdf.\n'
        'This code-free supplement follows the original October 20 architecture brief.\n'
        'It includes module deliverables, interfaces, evaluation, operator workflow and\n'
        'readiness checkpoints for Monday, October 26. CSI booking/feed remain unconfirmed.\n'
        'The text and diagram copies are included for accessibility and whiteboard use.\n'
        'No source code, model weights, datasets, reviewed labels or implementation recipe\n'
        'is included. No new real-data training or equipment operation is authorized.\n'
        'Prepared locally October 22; this package has not been published by its builder.\n', encoding='utf-8')
    files = {p.name: {'bytes': p.stat().st_size, 'sha256': digest(p)} for p in sorted(out.iterdir()) if p.is_file()}
    manifest = {'schema': 'team-integration-brief/v1', 'prepared': '2026-10-22', 'code_free': True, 'files': files}
    (out/'package-hashes.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(out.iterdir()):
            bundle.write(path, arcname=path.name)
    proof = {'source': str(source), 'source_sha256': digest(source), 'builder_sha256': digest(Path(__file__)), 'pdf': str(pdf), 'pdf_sha256': digest(pdf), 'zip': str(archive), 'zip_sha256': digest(archive), 'pdf_pages': 5, 'code_free': True, 'published': False}
    (out.parent/(out.name+'-build.json')).write_text(json.dumps(proof, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(proof, indent=2))


if __name__ == '__main__':
    main()
