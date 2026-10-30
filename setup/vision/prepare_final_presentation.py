"""Render the dated presentation brief and a portable, code-free sharing bundle."""
from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path
import shutil
import zipfile

import pymupdf
from PIL import Image as PILImage, ImageDraw, ImageFont
from reportlab.graphics import renderSVG
from reportlab.graphics.shapes import Drawing, Line, Polygon, Rect, String
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

NAVY, TEAL, GRAY, PALE = '#142B40', '#087F83', '#566676', '#EFF5F7'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def architecture(out):
    drawing = Drawing(1024, 440)

    def box(x, y, w, h, title, lines):
        drawing.add(Rect(x, y, w, h, rx=10, fillColor=colors.HexColor(PALE), strokeColor=colors.HexColor(TEAL)))
        drawing.add(String(x+13, y+h-25, title, fontName='LatoBold', fontSize=17, fillColor=colors.HexColor(NAVY)))
        for i, value in enumerate(lines):
            drawing.add(String(x+13, y+h-49-20*i, value, fontName='Lato', fontSize=15, fillColor=colors.HexColor(GRAY)))

    def arrow(points, dashed=False):
        for a, b in zip(points, points[1:]):
            drawing.add(Line(*a, *b, strokeColor=colors.HexColor(GRAY), strokeWidth=2, strokeDashArray=[6, 4] if dashed else None))
        (ax, ay), (x, y) = points[-2:]
        if x == ax:
            back = y+9 if y < ay else y-9
            triangle = [x, y, x-5, back, x+5, back]
        else:
            back = x+9 if x < ax else x-9
            triangle = [x, y, back, y-5, back, y+5]
        drawing.add(Polygon(triangle, fillColor=colors.HexColor(GRAY), strokeColor=None))

    box(15, 340, 225, 90, 'Original RGB / CPU', ['Frame + source identity', '640 x 480'])
    box(320, 340, 250, 90, 'YOLO11n / GPU', ['Same-frame vial geometry'])
    box(15, 200, 225, 90, 'Earlier reference', ['Identity + aligned camera', 'Qualification still required'])
    box(320, 200, 250, 90, 'Reference features', ['Fixed liquid ROI + gain', 'Signed / absolute RGB diff'])
    box(650, 200, 350, 90, 'Small U-Net / GPU', ['Presence + pool mask', 'Synthetic-only, random init'])
    box(320, 20, 250, 105, 'Association + QA / CPU', ['Raw-RGB sample or unavailable', 'Operator evidence + response'])
    box(650, 20, 350, 105, 'Optional SAM2.1 / GPU', ['Refine temporal proposal', 'Cached-upstream experiment'])
    arrow([(240, 385), (320, 385)])
    arrow([(260, 385), (280, 385), (280, 310), (445, 310), (445, 290)])
    arrow([(240, 245), (320, 245)])
    arrow([(570, 245), (650, 245)])
    arrow([(570, 385), (605, 385), (605, 155), (445, 155), (445, 125)])
    arrow([(700, 200), (700, 175), (475, 175), (475, 125)])
    arrow([(825, 200), (825, 125)], True)
    arrow([(650, 72), (570, 72)], True)
    renderSVG.drawToFile(drawing, str(out/'architecture.svg'))
    with pymupdf.open(out/'architecture.svg') as doc:
        doc[0].get_pixmap(matrix=pymupdf.Matrix(1.5, 1.5)).save(out/'architecture.png')


def training_board(results, out):
    root = results/'targeted-synthetic-20261022/paired-render-v1'
    rows = [json.loads(line) for line in (root/'manifest.jsonl').read_text().splitlines()]
    examples = [row for row in rows if row['group'] == 0]
    if len(examples) != 4:
        raise ValueError('Expected one complete four-state family')
    board = PILImage.new('RGB', (1280, 650), 'white')
    draw = ImageDraw.Draw(board)
    font = ImageFont.truetype('/usr/share/fonts/truetype/lato/Lato-Bold.ttf', 25)
    for i, row in enumerate(examples):
        source = root/row['image']
        if sha(source) != row['image_sha256']:
            raise ValueError(f'Render hash mismatch: {source}')
        with PILImage.open(source) as image:
            crop = image.crop((256, 240, 368, 432)).resize((280, 480))
        board.paste(crop, (i*320+20, 65))
        draw.text((i*320+20, 20), row['state'].replace('-', ' ').title(), fill=NAVY, font=font)
        label = 'Body target: yes' if row['mask_pixels'] else 'Body target: no'
        draw.text((i*320+20, 570), label, fill=TEAL, font=font)
    board.save(out/'synthetic-states.png')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    source = repo/'team_handoff/presentation/brief-20261022.json'
    spec = json.loads(source.read_text())
    out = args.out.resolve()
    archive = out.with_suffix('.zip')
    if out.exists() or archive.exists():
        raise ValueError('Use a fresh output directory to preserve distributed editions')
    out.mkdir(parents=True)
    fontroot = Path('/usr/share/fonts/truetype/lato')
    for name, filename in [('Lato', 'Lato-Regular.ttf'), ('LatoBold', 'Lato-Bold.ttf')]:
        pdfmetrics.registerFont(TTFont(name, str(fontroot/filename)))
    architecture(out)
    training_board(args.results, out)

    # Prevent source prose from drifting away from the actual report values.
    evidence = args.results/'synthetic-iterations-20261022/sam-temporal-summary-v1/comparison.json'
    data = json.loads(evidence.read_text())
    if [data['matched']['arms'][k]['qa_pass'] for k in ('legacy_sam', 'synthetic_sam', 'temporal')] != [185, 210, 105]:
        raise ValueError('Comparison evidence differs from presentation edition')
    if (data['refinement']['qa_gains'], data['refinement']['qa_losses']) != (80, 0):
        raise ValueError('Refinement evidence differs from presentation edition')
    styles = {
        'p': ParagraphStyle('body', fontName='Lato', fontSize=10, leading=13.4, textColor=colors.HexColor(NAVY), spaceAfter=9),
        'lead': ParagraphStyle('lead', fontName='LatoBold', fontSize=12.5, leading=17, textColor=colors.HexColor(TEAL), spaceAfter=13),
        'h': ParagraphStyle('heading', fontName='LatoBold', fontSize=12, leading=15, textColor=colors.HexColor(TEAL), spaceBefore=5, spaceAfter=7),
        'small': ParagraphStyle('small', fontName='Lato', fontSize=8.5, leading=11.3, textColor=colors.HexColor(GRAY), spaceAfter=8),
        'callout': ParagraphStyle('callout', fontName='LatoBold', fontSize=10, leading=13.4, textColor=colors.HexColor(NAVY), backColor=colors.HexColor('#E9F4F2'), borderPadding=9, spaceBefore=6, spaceAfter=16),
        'cell': ParagraphStyle('cell', fontName='Lato', fontSize=9.2, leading=12, textColor=colors.HexColor(NAVY)),
        'head': ParagraphStyle('head', fontName='LatoBold', fontSize=9.2, leading=12, textColor=colors.white),
        'title': ParagraphStyle('title', fontName='LatoBold', fontSize=25, leading=29, textColor=colors.HexColor(NAVY), spaceAfter=9),
        'sub': ParagraphStyle('sub', fontName='Lato', fontSize=11, leading=14.5, textColor=colors.HexColor(GRAY), spaceAfter=16)
    }

    def para(value, style='p'):
        return Paragraph(html.escape(str(value)), styles[style])

    story, markdown, notes = [], [f'# {spec["title"]}', '', spec['edition'], ''], ['# Presenter notes', '', spec['edition'], '']
    for index, page in enumerate(spec['pages'], 1):
        if index > 1:
            story.append(PageBreak())
        story += [para(f'PRESENTATION BRIEF / {index:02d}', 'h'), para(page['title'], 'title'), para(page['subtitle'], 'sub')]
        markdown += [f'## {index}. {page["title"]}', '', page['subtitle'], '']
        notes += [f'## {index}. {page["title"]}', '', page['speaker'], '']
        for block in page['blocks']:
            kind = block[0]
            if kind == 'image':
                with PILImage.open(out/block[1]) as image:
                    width, height = image.size
                story += [Image(str(out/block[1]), width=512, height=512*height/width), Spacer(1, 7), para(block[2], 'small')]
                markdown += [f'![{block[2]}]({block[1]})', '']
            elif kind == 'table':
                _, heads, rows, widths = block
                table = Table([[para(v, 'head') for v in heads]]+[[para(v, 'cell') for v in row] for row in rows], colWidths=widths)
                table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor(NAVY)),
                    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor(PALE)]),
                    ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                    ('LEFTPADDING', (0, 0), (-1, -1), 8), ('RIGHTPADDING', (0, 0), (-1, -1), 8),
                    ('TOPPADDING', (0, 0), (-1, -1), 6), ('BOTTOMPADDING', (0, 0), (-1, -1), 6)
                ]))
                story += [table, Spacer(1, 12)]
                markdown += ['| '+' | '.join(heads)+' |', '| '+' | '.join(['---']*len(heads))+' |']
                markdown += ['| '+' | '.join(row)+' |' for row in rows]+['']
            else:
                story.append(para(block[1], kind))
                markdown += [('### ' if kind == 'h' else '')+block[1], '']

    def footer(canvas, doc):
        canvas.setStrokeColor(colors.HexColor('#D4E0E7'))
        canvas.line(50, 39, 562, 39)
        canvas.setFillColor(colors.HexColor(GRAY))
        canvas.setFont('Lato', 8)
        canvas.drawString(50, 26, 'Austin Koske / Innovation Lab / October 22, 2026 / Development evidence')
        canvas.drawRightString(562, 26, str(doc.page))

    pdf = out/'Synthetic-Training-and-Digital-Twin-Presentation.pdf'
    SimpleDocTemplate(str(pdf), pagesize=letter, leftMargin=50, rightMargin=50, topMargin=39, bottomMargin=54, title=spec['title'], author='Austin Koske').build(story, onFirstPage=footer, onLaterPages=footer)
    with pymupdf.open(pdf) as doc:
        if len(doc) != len(spec['pages']):
            raise ValueError(f'Layout overflow: {len(doc)} pages, expected {len(spec["pages"])}')
        for i, page in enumerate(doc):
            if spec['pages'][i]['title'] not in page.get_text():
                raise ValueError(f'Unexpected pagination at page {i+1}')
            page.get_pixmap(matrix=pymupdf.Matrix(0.9, 0.9)).save(out/f'page-{i+1:02d}.png')
    (out/'Presentation-Brief.md').write_text('\n'.join(markdown), encoding='utf-8')
    (out/'Presenter-Notes.md').write_text('\n'.join(notes), encoding='utf-8')
    shutil.copy2(source, out/source.name)
    shutil.copy2(repo/'team_handoff/presentation/README.md', out/'README.md')
    shutil.copy2(evidence, out/'measured-comparison.json')
    video = args.results/'synthetic-iterations-20261022/gb10-results/temporal-preview-v1/temporal-pool-candidate.mp4'
    shutil.copy2(video, out/video.name)
    model = args.results/'synthetic-iterations-20261022/gb10-results/masks-v1/paired_diff_gain/frozen-model.json'
    shutil.copy2(model, out/'liquid-model-provenance.json')
    verification = {
        'edition': spec['edition'], 'pages': len(spec['pages']), 'source_sha256': sha(source),
        'evidence_sha256': sha(evidence), 'model_sha256': json.loads(model.read_text())['checkpoint_sha256'],
        'dataset1_implemented': False, 'new_experiments_run': False, 'human_accuracy_available': False,
        'files': {str(p.relative_to(out)): sha(p) for p in sorted(out.rglob('*')) if p.is_file()}
    }
    (out/'completed.json').write_text(json.dumps(verification, indent=2)+'\n')
    with zipfile.ZipFile(archive, 'x', zipfile.ZIP_DEFLATED) as z:
        for path in sorted(out.rglob('*')):
            if path.is_file() and not path.name.startswith('page-'):
                z.write(path, f'presentation-20261022/{path.relative_to(out)}')
    print(json.dumps({'pdf': str(pdf), 'zip': str(archive), 'pages': len(spec['pages'])}))


if __name__ == '__main__':
    main()
