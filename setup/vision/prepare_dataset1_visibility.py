"""Prepare isolated geometry changes and a repeated-golden rendering noise control."""
import argparse
import copy
from pathlib import Path

from dataset1_fiveclass import SLOTS,read,sha,write


def prepare(parent,out):
    out.mkdir(exist_ok=False)
    original=parent/'dataset1_fiveclass.py';text=original.read_text()
    replacements={
        "for variant in ('golden','inspection'):":"for variant in ('golden','repeat','inspection'):",
        "label='pass' if variant=='golden' else family['labels'][slot]":"label='pass' if variant!='inspection' else family['labels'][slot]",
        "else:rows.append(dict(id=family['id']":"elif variant=='repeat':repeat_name=path.name;repeat_hash=sha(path)\n                else:rows.append(dict(repeat=repeat_name,repeat_sha256=repeat_hash,id=family['id']",
        'images=2*len(rows)':'images=3*len(rows)',
    }
    for old,new in replacements.items():
        if text.count(old)!=1:raise ValueError('Frozen renderer template changed: '+old)
        text=text.replace(old,new)
    (out/'visibility_renderer.py').write_text(text)
    (out/'input').symlink_to((parent/'input').resolve(),target_is_directory=True)
    plan=copy.deepcopy(read(parent/'experiment-plan.json'));base=copy.deepcopy(plan['families'][0]);rows=[]
    cases=[(fixture+'15','damaged_vial',lean,kind) for fixture in ('G','S') for lean in (0,18,30) for kind in ('chip','dent')]
    cases += [(fixture+'15',label,0,'chip') for fixture in ('G','S') for label in ('missing_cap','crooked_cap')]
    for i,(slot,label,lean,kind) in enumerate(cases):
        family=copy.deepcopy(base);family.update(id=f'isolated-{i:02d}',group=20000+i,split='test',camera_dx_px=0,camera_dy_px=0,
            light_scale=1,cap_color=[.65,.65,.65],labels={s:'pass' for s in SLOTS})
        for params in family['parameters'].values():params['body_lean_deg']=0
        family['labels'][slot]=label
        family['parameters'][slot].update(body_lean_deg=lean,body_azimuth_deg=0,damage_kind=kind,damage_azimuth_deg=0,
            damage_strength=.52,cap_tilt_deg=32,cap_tilt_azimuth_deg=0,cap_lift_m=.004)
        family['isolated_case']=dict(slot=slot,geometric_class=label,lean_deg=lean,damage_kind=kind,
            strength=.52,all_other_targets='pass',repeat_golden_noise_control=True)
        rows.append(family)
    plan.update(families=rows,groups=dict(train=0,val=0,test=len(rows)),renderer_sha256=sha(out/'visibility_renderer.py'),
        purpose='Optical identifiability diagnosis, not a classifier holdout, physical damage validation or real accuracy',
        original_renderer_sha256=sha(original),preparation_helper_sha256=sha(__file__))
    write(out/'experiment-plan.json',plan)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--parent',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();prepare(a.parent.resolve(),a.out.resolve())
