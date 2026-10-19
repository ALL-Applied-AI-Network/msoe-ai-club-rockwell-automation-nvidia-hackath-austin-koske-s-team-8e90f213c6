"""Prepare 18 review-only cases and hashed RGB references from Runs 1-3 only."""
import argparse
import json
from pathlib import Path
import shutil

from box_annotations import sha256, validate_fixture, write_json


REFERENCE_IDS = {
    1: ['vial_fill_20260922_170544_697177', 'vial_fill_20260922_170548_711253', 'vial_fill_20260922_170600_743982'],
    2: ['vial_fill_20260922_192633_265636', 'vial_fill_20260922_192639_295513', 'vial_fill_20260922_192651_356125'],
    3: ['vial_fill_20260923_184720_822117', 'vial_fill_20260923_184726_838430', 'vial_fill_20260923_184809_074191']}


def make(references):
    cases = []
    def case(name, level, stream, radius=.997, color=None, **extra):
        row=dict(id=name, level=level, stream=stream, rgb=[.65,.84,.88], opacity=1.,
            transmission_color=color or [.80,.86,.52], absorption_depth_m=.025,
            pool_radius_fraction=radius, meniscus_m=.0012, ripple_m=.0005,
            stream_radius_m=.001, jet_length_fraction=.25, jet_wobble_m=.00035,
            roughness=.035, phase=len(cases)*.7, light_scale=1.)
        row.update(extra); cases.append(row)
    case('v2-00-empty',0,'off')
    case('v2-01-pendant-drop',0,'drop')
    case('v2-02-falling-jet',0,'short')
    case('v2-03-impact-no-pool',0,'continuous')
    case('v2-04-first-wet-patch',.002,'continuous',.38)
    case('v2-05-small-pale-pool',.004,'continuous',.55)
    case('v2-06-growing-pale-pool',.01,'continuous',.72)
    case('v2-07-low-pale',.03,'continuous')
    case('v2-08-mid-pale',.30,'continuous')
    case('v2-09-high-pale',.78,'off')
    case('v2-10-low-clear',.02,'continuous',color=[.98,.995,.97],absorption_depth_m=.06)
    case('v2-11-mid-clear',.30,'off',color=[.98,.995,.97],absorption_depth_m=.06)
    case('v2-12-low-blue',.03,'continuous',color=[.12,.55,.82],absorption_depth_m=.02)
    case('v2-13-mid-blue',.30,'off',color=[.12,.55,.82],absorption_depth_m=.02)
    case('v2-14-impact-ripples',.004,'continuous',.55,ripple_m=.0012,jet_wobble_m=.0008)
    case('v2-15-rippled-pale',.10,'continuous',ripple_m=.0012)
    case('v2-16-stopped-pale',.30,'off')
    case('v2-17-weaker-pale-pool',.004,'continuous',.55,absorption_depth_m=.07)
    result=dict(schema_version=1, station_frame=686, renderer_profile='meniscus-transmission-v2',
        purpose='Small RGB review only; no automatic training eligibility',
        camera='Supplied FillCam unchanged', require_visible_present_objects=True,
        appearance_references=references, real_label_inputs=[],
        reference_policy='Qualitative RGB review from Runs 1-3; no numerical fit or recipe inference; no Runs 4/5 read',
        material_policy='MDL volume transmission and absorption, no diffuse opacity proxy; parameters provisional',
        geometry_policy='Procedural static snapshots; no CFD, measured volume, bubbles or temporal validation',
        cases=cases)
    validate_fixture(result)
    return result


def main(args):
    root=args.real.resolve(); provenance=json.loads((root/'provenance.json').read_text())
    rows={s['source_id']:s for s in provenance['samples'] if not s.get('parent_id') and s['source_run'] in (1,2,3)}
    args.out.mkdir(parents=True,exist_ok=False); (args.out/'references').mkdir()
    refs=[]
    for run, ids in REFERENCE_IDS.items():
        for identity in ids:
            s=rows[identity]; path=root/s['image']
            if s['source_run']!=run or s['split']!='train' or sha256(path)!=s['image_sha256']:
                raise ValueError('Reference identity/split/hash mismatch')
            target=args.out/'references'/(identity+'.jpg'); shutil.copyfile(path,target)
            refs.append(dict(source_id=identity, source_run=run, image_sha256=sha256(path),
                file='references/'+target.name, use='Qualitative RGB only; no teacher boxes or timing labels'))
    write_json(args.out/'fixture.json',make(refs))
    write_json(args.out/'references.json',dict(schema_version=1, source_inventory_sha256=sha256(root/'provenance.json'),
        selected_rgb_only=True, excluded_runs=[4,5], references=refs))
    print(f'Prepared 18 review cases and {len(refs)} RGB references: {args.out}')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--real',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    main(parser.parse_args())
