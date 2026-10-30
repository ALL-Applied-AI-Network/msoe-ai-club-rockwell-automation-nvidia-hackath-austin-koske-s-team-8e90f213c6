"""Code-free research brief from verified local experiment results."""
import argparse
import json
from pathlib import Path


def build(report,out):
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,PageBreak
    r=json.loads(report.read_text());out.parent.mkdir(exist_ok=True,parents=True)
    if out.exists():raise ValueError('Preserve existing brief; choose a new path')
    pdfmetrics.registerFont(TTFont('Lato','/usr/share/fonts/truetype/lato/Lato-Regular.ttf'))
    pdfmetrics.registerFont(TTFont('LatoBold','/usr/share/fonts/truetype/lato/Lato-Bold.ttf'))
    styles=getSampleStyleSheet()
    for key in ('Normal','BodyText','Heading1','Heading2','Title'):styles[key].fontName='LatoBold' if key in ('Title','Heading1','Heading2') else 'Lato'
    styles['BodyText'].fontSize=9.5;styles['BodyText'].leading=13
    styles.add(ParagraphStyle(name='TinyLato',fontName='Lato',fontSize=8,leading=10))
    flow=[]
    def p(text,style='BodyText'):flow.append(Paragraph(text,styles[style]));flow.append(Spacer(1,5))
    def table(rows,widths=None):
        cells=[[Paragraph(str(c),styles['TinyLato']) for c in row] for row in rows]
        t=Table(cells,colWidths=widths,repeatRows=1,hAlign='LEFT');t.setStyle(TableStyle([
            ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#dcebf0')),('VALIGN',(0,0),(-1,-1),'TOP'),
            ('BOTTOMPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),6),
            ('LINEBELOW',(0,0),(-1,0),.7,colors.HexColor('#416979')),('LINEBELOW',(0,1),(-1,-1),.25,colors.HexColor('#cbd7dc'))]))
        flow.append(t);flow.append(Spacer(1,12))
    p('Digital-twin inspection: what the experiments establish','Title')
    p('Team research update | October 23, 2026 | Synthetic training; real accuracy still unmeasured')
    p('The reference input improves cap inspection. It does not establish reliable whole-vial inspection. Damage visibility, normal-state false alarms and real evaluation remain separate gates.','Heading2')
    p('All new fitting ran on the GB10, using synthetic CSI scenes and random initialization. Real images and Run 5 were excluded from fitting and selection. Checkpoints and warning rules were frozen using synthetic development data. None of these experiments changed the deployed pipeline or approved a product.')
    p('Matched architecture test','Heading2')
    rows=[['Model','Exact five-class accuracy','Crooked-cap recall','Damaged-body recall']]
    for name,m in r['detail_fresh_test']['arms'].items():rows.append([name,f'{m["accuracy"]:.1%}',f'{m["per_class"]["crooked_cap"]["recall"]:.1%}',f'{m["per_class"]["damaged_vial"]["recall"]:.1%}'])
    table(rows,[155,112,112,112])
    p('Same 720 synthetic targets, 12 new scene families, 144 targets per class. The detail model uses a central crop and a context crop. The paired version receives current RGB, matched golden RGB, absolute difference and target position. This is a synthetic architecture comparison, not real transfer accuracy.')
    p('Why “caught every defect” was misleading','Heading2')
    p('The original warning threshold was set to catch every development defect. On the fresh test it warned on essentially every acceptable target. More relaxed development thresholds still show a substantial false-alarm/miss trade-off. Report both; class accuracy alone does not establish an operator-ready warning system.')
    flow.append(PageBreak())
    p('What changed in the data and what remains weak','Title')
    p('Normal scenes are part of the task','Heading2')
    p('The initial inspection images were all densely defective. The follow-up reused training golden images as normal current examples, retaining whole-family splits. It then scored six fresh dense families, six fresh sparse families, and the twelve all-good golden images as current inputs: 1,440 correlated target observations in total.')
    p('The all-good examples reuse identical current/reference RGB before joint augmentation. Treat those results as optimistic normal controls; separate-camera-capture variation requires its own test.')
    rows=[['Model','Five-class macro recall','Dense exact','Sparse exact','All-good exact']]
    for name in ('previous_detail_rgb','previous_detail_paired','detail_rgb','detail_paired'):
        m=r['normal_coverage_fresh_test']['arms'][name];s=r['normal_strata'][name]
        rows.append([name,f'{m["macro_recall"]:.1%}',f'{s["dense"]["accuracy"]:.1%}',f'{s["sparse"]["accuracy"]:.1%}',f'{s["normal"]["accuracy"]:.1%}'])
    table(rows,[155,100,78,78,80])
    p('“Previous” models are frozen detail models without normal-current training. New detail models include normal-current examples. Architecture and seed are shared, but this is a single-seed exploratory ablation, not a statistically established generalization gain.')
    a=r['matched_normal_ablation'];p('At each model’s threshold frozen for95% pooled development defect recall, acceptable-target false alarms fall from '+str(a['previous_without_normal_current']['false_alarms'])+'/1140 to '+str(a['with_normal_current']['false_alarms'])+'/1140. Damaged-target misses rise from8/76 to15/76; neither arm misses a missing/crooked cap. Pooled recall does not guarantee recall for each defect class.')
    p('The damage label is often much stronger than its visible pixels','Heading2')
    o=r['optical_identifiability']
    p(f'In 12 isolated body-damage changes, only {o["damage_pixels_above_noise_range"][0]}–{o["damage_pixels_above_noise_range"][1]} of 4,096 central-crop pixels changed above the chosen digital threshold. Four cap controls changed {o["cap_pixels_above_noise_range"][0]}–{o["cap_pixels_above_noise_range"][1]} pixels. Body change mean absolute difference was {o["damage_change_mae_range"][0]:.3f}–{o["damage_change_mae_range"][1]:.3f} digital levels, versus {o["cap_change_mae_range"][0]:.2f}–{o["cap_change_mae_range"][1]:.2f} for cap changes.')
    p('These controls isolate one mesh change while keeping neighbors and nuisance parameters fixed. The threshold has a two-digital-level floor. The unchanged renderer can reuse accumulation; physical sensor noise was not simulated. This demonstrates weak image evidence for these proxies, not that all real damage is invisible.')
    p('A damage mesh beneath an opaque cap is not automatically a useful damaged-vial training example. Keep geometric labels, optical visibility and real human labels separate. A larger model cannot recover detail that the camera does not expose.')
    p('A second gap is explicit in the CAD: the five authored S vials lie roughly81–84 degrees from vertical, while the generated canonical G/S bodies cover0–30 degrees. Rebuild S pose/target coverage from the authored geometry before treating synthetic S scores as station coverage. The source stage remains unchanged.')
    flow.append(PageBreak())
    p('A useful pipeline boundary for the team','Title')
    p('Camera image → alignment and target visibility → slot crop + qualified golden → cap state and occupancy → placement context → operator evidence','Heading2')
    p('Cap state means nominal/seated-looking cap, missing cap, crooked cap or empty position. Body condition stays unassessed. An empty position becomes a missed-placement candidate only when attributed progress says it should be occupied. Operator acknowledgement is recorded; no direct line control or product release is implied.')
    c=r['cap_state'];m=c['metrics']
    p('Dedicated cap-state experiment','Heading2')
    table([['Scope','Synthetic evidence'],['Exact cap-state accuracy',f'{m["accuracy"]:.2%} on {m["targets"]} targets'],
        ['Per-class recall','; '.join(f'{name}: {v["recall"]:.1%}' for name,v in m['per_class'].items())],
        ['Entire scenes correct',f'{m["all_60_correct_scenes"]}/{m["complete_scenes"]}'],
        ['Body damage / real accuracy','Unassessed / unavailable']], [185,306])
    policy=c['policies']['0.99'];p('At the threshold frozen for99% development cap-defect recall: '+str(policy['false_alarms'])+'/'+str(policy['seated_targets'])+' nominal-cap false warnings; '+', '.join(f'{name}: {v["missed"]}/{v["targets"]} missed' for name,v in policy['per_class'].items())+'. Empty positions require progress context. This threshold does not assess body quality.')
    stress=r['cap_reference_stress']['cases'];p('Reference stress on720 all-good synthetic targets, same frozen 95% policy: '+', '.join(f'{name}: {stress[name]["normal_policies"]["0.95"]["false_alarms"]}/720 warnings' for name in ('unchanged','reference_gain_0.9','reference_shift_1px'))+'. These digital controls do not establish tolerance of real camera drift.')
    p('A label-free global reference gain correction repairs the tested brightness changes without fitting. A subsequent six-family confirmation adds fresh scenes, reference gains0.83/1.17, independent digital RGB noise and JPEG90 crop compression. These are engineering perturbations, not a measured physical camera-noise model. An automatic registration-validity detector is not established here; misaligned references must remain unavailable.')
    confirm=r['cap_fresh_confirmation']['cases'];p('Fresh confirmation exact cap accuracy: '+', '.join(f'{name}: {confirm[name]["metrics"]["accuracy"]:.1%}' for name in ('unchanged','reference_noise_1dl','reference_jpeg90'))+'. Full condition/threshold tables are retained in the verified report.')
    p('Next physical test','Heading2')
    p('Confirm both fixture maps, visibility and image resolution. Capture a qualified nominal reference and define reset/recapture conditions. Use CSI-approved known cap/empty examples, with human labels recorded independently of predictions. Preserve timestamps and source hashes; score per-class misses, normal false alarms, complete scenes and end-to-end operator latency. Inspect body damage only where the view supports it; request another view or manual review otherwise.')
    p('Srijan owns live model adapters; Josiah/Hayden own operator UI/backend. Existing Dataset 2 filling evidence remains its own subsystem. Dataset 1 can add a standalone final-station check; joining it to the same physical filling vial requires an attributed identity link. The supplied manual tool offers 120 unresolved native-image positions, alongside our 960-target provisional two-fixture review packet.')
    def footer(canvas,doc):
        canvas.setFont('Lato',8);canvas.setFillColor(colors.HexColor('#526875'));canvas.drawString(42,25,'Synthetic research evidence | no reviewed real defect accuracy | operator-mediated only');canvas.drawRightString(570,25,str(doc.page))
    SimpleDocTemplate(str(out),pagesize=(612,792),rightMargin=42,leftMargin=42,topMargin=35,bottomMargin=43).build(flow,onFirstPage=footer,onLaterPages=footer)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--report',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();build(a.report,a.out)
