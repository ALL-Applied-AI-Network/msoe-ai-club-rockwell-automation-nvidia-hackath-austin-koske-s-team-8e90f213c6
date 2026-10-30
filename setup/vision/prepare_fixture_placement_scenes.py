"""Prepare six isolated synthetic placement states, never used for model fitting."""
import argparse
import copy
from pathlib import Path
from dataset1_fiveclass import SLOTS,families,read,sha,write


def prepare(parent,out):
    out.mkdir(exist_ok=False);(out/'input').symlink_to(parent/'input',target_is_directory=True)
    plan=copy.deepcopy(read(parent/'experiment-plan.json'))
    definitions=[('empty',0,0,{}),('first_good',1,0,{}),('missed_G03',6,0,{'G03':'no_vial'}),
        ('crooked_G07_missing_S02',7,3,{'G07':'crooked_cap','S02':'missing_cap'}),
        ('damaged_G16',30,5,{'G16':'damaged_vial'}),('complete_good',30,5,{})]
    scenes=families(seed=88172,counts=(6,0,0))
    for scene,(name,g_count,s_count,overrides) in zip(scenes,definitions):
        scene.update(id=name,split='integration_rehearsal',placement_count={'G':g_count,'S':s_count},order_complete={'G':g_count==30,'S':False})
        scene['labels']={slot:'pass' if int(slot[1:])<=({'G':g_count,'S':s_count}[slot[0]]) else 'no_vial' for slot in SLOTS}
        scene['labels'].update(overrides)
        # Hold poses fixed upright to make a readable placement-state demonstration.
        for params in scene['parameters'].values():params['body_lean_deg']=0
    plan.update(families=scenes,groups={'integration_rehearsal':6},training_eligible=False,parent_plan_sha256=sha(parent/'experiment-plan.json'),
        scenario_plan_helper_sha256=sha(__file__),purpose='Post-freeze rendered placement-state rehearsal, not robot-motion or independent physical validation')
    write(out/'experiment-plan.json',plan)
    # Retain the exact existing renderer; this script never modifies its bytes.
    import shutil
    shutil.copy2(parent/'dataset1_fiveclass.py',out/'dataset1_fiveclass.py')
    print('Prepared six synthetic-only placement states',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--parent',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();prepare(a.parent.resolve(),a.out.resolve())
