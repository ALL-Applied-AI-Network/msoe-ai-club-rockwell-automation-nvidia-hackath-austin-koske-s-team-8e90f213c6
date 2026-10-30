"""Paired synthetic hard-negative experiment; no real fine-tuning or asset saves.

Render edits live exclusively in the USD session layer. Geometry labels describe
the accumulated body, never a stream or the whole vial. They are not optical
segmentation ground truth. See the frozen experiment plan for selection policy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def prepare_render(a):
    a.out.mkdir(parents=True, exist_ok=False)
    rng = random.Random(2719)
    groups = list(range(a.groups)); rng.shuffle(groups)
    val = set(groups[:max(1, round(a.groups*.25))])
    cases = []
    colors = [[.12,.55,.82], [.8,.86,.52], [.55,.15,.25], [.16,.65,.30]]
    for group in range(a.groups):
        nuisance = dict(light_scale=rng.uniform(.7,1.3), glass_roughness=rng.uniform(.025,.16),
            label_brightness=rng.uniform(.65,1.25), camera_dx_px=rng.uniform(-8,8),
            camera_dy_px=rng.uniform(-6,6), color=colors[group % len(colors)],
            absorption_depth_m=rng.uniform(.012,.035), phase=rng.uniform(0,6.28))
        early = [.001,.003,.008,.02,.04,.07][group % 6]
        for state, level, stream in [('empty',0,'off'), ('stream-only',0,'continuous'),
                                      ('early',early,'continuous' if group%2 else 'off'),
                                      ('established',rng.uniform(.15,.75),'off')]:
            cases.append(dict(id=f'pair-{group:03d}-{state}', group=group,
                split='val' if group in val else 'train', state=state, level=level, stream=stream,
                pool_radius_fraction=min(.997,.32+level*14) if state=='early' else .997,
                nuisance=nuisance))
    helpers = [Path(__file__), HERE/'live_twin_isaac.py', HERE/'twin_contract.py',
               REPO/'setup/isaac/liquid_geometry.py', REPO/'setup/isaac/box_annotations.py']
    (a.out/'helpers').mkdir()
    for p in helpers: shutil.copy2(p, a.out/'helpers'/p.name)
    plan = dict(schema_version=1, seed=2719, cases=cases,
        purpose='Counterfactual empty/stream-only/body scenes with shared nuisance settings within each group.',
        training_real_runs=[], selection_real_runs=[], run5_used=False,
        camera_policy='Supplied camera plus recorded image-plane pose translations; actual matrix retained per group.',
        material_policy='Session-layer MDL transmission glass, IOR1.47/depth0.6m, roughness variation; paper diffuse brightness variation; procedural transmission body.',
        target_scope='Accumulated liquid body only; empty and stream-only are genuine synthetic negatives.',
        split_policy='Whole counterfactual group split, 25% development, seed2719; never select using real scores.',
        fit_policy='Replace original synthetic training samples with eligible new training samples, total614 per epoch; same 3 epochs and 462 optimizer steps as baseline. Fixed epoch3 checkpoint.',
        synthetic_visibility_screen='Positive case eligible only if mean RGB absolute difference vs paired empty over a 5px-expanded geometric mask exceeds3 and at least10 pixels differ by more than12 in some channel. Exploratory signal screen, not optical truth; excludes invisible positives, never converts them into negatives.',
        limitations=['Appearance parameters are exploratory, not calibrated glass or CFD.',
                    'Geometric body boxes may not match refracted RGB boundaries.',
                    'Clear liquid omitted from this pilot because geometric existence does not establish visibility.',
                    'Real labels remain provisional; development agreement is not human annotation accuracy.'],
        assets=str(a.assets), helpers={p.name:sha(p) for p in helpers}, spp=a.spp, subframes=a.subframes)
    write(a.out/'render-plan.json', plan)
    write(a.out/'preparation.json', dict(plan_sha256=sha(a.out/'render-plan.json'), helpers=plan['helpers']))
    print(json.dumps(dict(prepared=str(a.out), images=len(cases), groups=a.groups)), flush=True)


def launch(a):
    if (a.out/'launch.json').exists(): raise ValueError('Inspect existing launch; use a fresh output')
    cmd = ['cmd.exe','/d','/c','C:/Apps/IsaacSim/isaacsim/_build/windows-x86_64/release/python.bat',
           '-u', str(a.out/'helpers'/Path(__file__).name), 'render', '--out',str(a.out),'--no-ros-env']
    with (a.out/'isaac.log').open('w') as log:
        child = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=REPO)
        write(a.out/'launch.json', dict(pid=child.pid, command=cmd, timeout_s=1800))
        try:
            code = child.wait(timeout=1800)
            if code or not (a.out/'completed-render.json').exists():
                raise RuntimeError(f'Render failed, exit={code}; inspect isaac.log')
        except BaseException:
            if child.poll() is None: subprocess.run(['taskkill','/PID',str(child.pid),'/T','/F'],capture_output=True)
            raise


def prepare_fit(a):
    import train_synthetic_liquid_pilot as pilot
    if not (a.renders/'completed-render.json').exists():
        raise ValueError('Completed render batch required')
    completion=read(a.renders/'completed-render.json')
    if sha(a.renders/'manifest.jsonl')!=completion['manifest_sha256']:
        raise ValueError('Render inventory differs')
    render_plan=read(a.renders/'render-plan.json')
    if completion['plan_sha256']!=sha(a.renders/'render-plan.json'):
        raise ValueError('Render plan differs')
    a.augment=False
    pilot.prepare(a)
    plan=read(a.out/'experiment-plan.json')
    rows=[json.loads(line) for line in (a.renders/'manifest.jsonl').read_text().splitlines()]
    new=[]
    for row in rows:
        if not row.get('training_eligible',True):continue
        image=a.renders/row['image'];label=a.renders/row['label']
        if sha(image)!=row['image_sha256'] or sha(label)!=row['label_sha256']:
            raise ValueError('Rendered source changed')
        image_rel='data/targeted-images/'+image.name;label_rel='data/targeted-labels/'+label.name
        for source,target in [(image,a.out/image_rel),(label,a.out/label_rel)]:
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
        new.append(dict(source_id=row['id'],origin='synthetic',source_run=None,split=row['split'],
            counterfactual_group=row['group'],state=row['state'],image=image_rel,label=label_rel,
            image_sha256=sha(image),label_sha256=sha(label),liquid_targets=int(row['box'] is not None)))
    train_new=[s for s in new if s['split']=='train']
    original=[s for s in plan['samples'] if s['split']=='train']
    random.Random(2719).shuffle(original)
    kept=original[:614-len(train_new)]
    if len(kept)+len(train_new)!=614:raise ValueError('Training exposure budget changed')
    plan['samples']=kept+[s for s in plan['samples'] if s['split']=='val']+new
    train_hashes={s['image_sha256'] for s in plan['samples'] if s['split']=='train'}
    if train_hashes & {s['image_sha256'] for s in plan['samples'] if s['split']=='val'}:
        raise ValueError('Identical image bytes cross the synthetic split')
    train_groups={s['counterfactual_group'] for s in new if s['split']=='train'}
    if train_groups & {s['counterfactual_group'] for s in new if s['split']=='val'}:
        raise ValueError('Counterfactual families cross the synthetic split')
    plan['kind']='synthetic_only_paired_negatives_fasterrcnn_pilot'
    plan['augmentation']='none'
    plan['counts']={split:dict(images=sum(s['split']==split for s in plan['samples']),
        positive=sum(s['split']==split and s['liquid_targets']>0 for s in plan['samples']),
        targets=sum(s['liquid_targets'] for s in plan['samples'] if s['split']==split)) for split in ['train','val']}
    plan['synthetic_split']='Original image-hash split retained; new counterfactual groups split as indivisible families before rendering.'
    plan['targeted_render_plan']=render_plan
    plan['render_source']=dict(path=str(a.renders),manifest_sha256=sha(a.renders/'manifest.jsonl'),
        completed_sha256=sha(a.renders/'completed-render.json'))
    plan['matched_budget']=dict(baseline_training_images=614,candidate_training_images=614,
        epochs=3,optimizer_steps_per_epoch=154,total_steps=462,
        kept_original_training_images=len(kept),new_targeted_training_images=len(train_new))
    plan['limitations']=render_plan['limitations']+[
        'Adaptive development experiment motivated by inspected real Runs1-4, not a blind test.',
        'Paired negatives, body optics, small pools, camera/lighting/material variation and source mix change together; this is not a single-factor causal ablation.',
        'Original synthetic development and new grouped development are reported separately for comparison.']
    shutil.copy2(Path(__file__),a.out/'helpers'/Path(__file__).name)
    plan['helpers'][Path(__file__).name]=sha(Path(__file__))
    write(a.out/'experiment-plan.json',plan)
    prep=read(a.out/'preparation.json');prep['plan_sha256']=sha(a.out/'experiment-plan.json')
    prep['helpers']=plan['helpers']
    write(a.out/'preparation.json',prep)
    print(json.dumps(dict(prepared=str(a.out),counts=plan['counts'],budget=plan['matched_budget'])),flush=True)


def render(a):
    plan = read(a.out/'render-plan.json')
    if sha(a.out/'render-plan.json') != read(a.out/'preparation.json')['plan_sha256']:
        raise ValueError('Render plan changed')
    for name, expected in plan['helpers'].items():
        if sha(a.out/'helpers'/name) != expected: raise ValueError('Renderer snapshot changed')
    assets = Path(plan['assets'])
    before = {str(p.relative_to(assets)):sha(p) for p in sorted(assets.rglob('*')) if p.is_file()}
    from isaacsim import SimulationApp
    app = SimulationApp(dict(headless=True, renderer='PathTracing', width=640, height=480,
        multi_gpu=False, extra_args=['--/app/extensions/registryEnabled=false',
        '--/telemetry/enableAnonymousData=false','--/crashreporter/enabled=false']))
    try:
        import numpy as np
        from PIL import Image
        from pxr import UsdGeom, UsdShade, Gf, Sdf
        # All dependencies beside this script are frozen copies.
        from live_twin_isaac import Station, CAMERA
        from box_annotations import semantic_mask, mask_box
        station = Station(app, assets, plan['spp'], plan['subframes'])
        stage = station.stage
        original_glass={str(vial.GetPath()):UsdShade.MaterialBindingAPI(vial).ComputeBoundMaterial()[0] for vial in station.vials}
        from liquid_geometry import transmission_material, set_transmission
        glass,glass_shader=transmission_material(stage,'TargetedGlass')
        glass_shader.GetInput('specular_reflection_ior').Set(1.47)
        set_transmission(glass_shader,dict(transmission_color=[.97,.99,.995],absorption_depth_m=.6,roughness=.06),UsdGeom.GetStageMetersPerUnit(stage))
        for vial in station.vials:UsdShade.MaterialBindingAPI.Apply(vial).Bind(glass)
        inventory = []
        glass_attrs = []; label_attrs = []
        for prim in stage.Traverse():
            name = str(prim.GetPath())
            if prim.GetTypeName() in ('Mesh','Shader'):
                inventory.append(dict(path=name,type=prim.GetTypeName(),
                    inputs={at.GetName():str(at.Get()) for at in prim.GetAttributes() if at.GetName().startswith('inputs:')},
                    material=str(UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0].GetPath()) if prim.GetTypeName()=='Mesh' else None))
            # Only alter roughness inputs of original glass shaders. Preserve
            # bindings and paper texture/subsets rather than replacing a vial.
            if prim.GetTypeName()=='Shader' and 'glass' in name.lower():
                for attr in prim.GetAttributes():
                    if attr.GetName().lower() in ('inputs:roughness','inputs:specular_reflection_roughness'):
                        glass_attrs.append((attr,attr.Get()))
            if prim.GetTypeName()=='Shader' and name.startswith('/FillStation/_materials/') and any(k in name.lower() for k in ('paper','label')):
                for attr in prim.GetAttributes():
                    if attr.GetName().lower() in ('inputs:diffuse_color_constant','inputs:diffusecolor','inputs:diffuse_reflection_color'):
                        value=attr.Get()
                        if value is not None: label_attrs.append((attr,value))
        write(a.out/'scene-inventory.json',inventory)
        write(a.out/'appearance-controls.json',dict(glass_attributes=[str(x[0].GetPath()) for x in glass_attrs],
            label_attributes=[str(x[0].GetPath()) for x in label_attrs], label_control_applied=bool(label_attrs),
            glass_control_applied=bool(glass_attrs)))
        cam = UsdGeom.Xformable(stage.GetPrimAtPath(CAMERA))
        camera_world = cam.ComputeLocalToWorldTransform(station.Usd.TimeCode.Default())
        # Camera parent transforms remain intact; use a new terminal local op.
        shift = cam.AddTranslateOp(opSuffix='targetedSyntheticOffset')
        for folder in ('rgb','masks','labels'): (a.out/folder).mkdir()
        records=[]; empty_rgb={}; started=time.perf_counter()
        with (a.out/'manifest.jsonl').open('w',encoding='utf-8') as log:
            for case in plan['cases']:
                n=case['nuisance']
                for vial in station.vials:
                    material=original_glass[str(vial.GetPath())] if n.get('glass_profile')=='original' else glass
                    UsdShade.MaterialBindingAPI.Apply(vial).Bind(material)
                for attr,value in station.lights: attr.Set(value*n['light_scale'])
                for attr,_ in glass_attrs: attr.Set(n['glass_roughness'])
                for attr,value in label_attrs:
                    attr.Set(type(value)(*[min(1.,float(v)*n['label_brightness']) for v in value]))
                # At the active-vial depth, image-plane translations approximate
                # a few pixels of fixture mounting variation, without changing K.
                depth=np.linalg.norm(np.asarray(camera_world)[3,:3]-station.center)
                fx=station.camera['supplied']['intrinsics_K_px'][0][0]
                shift.Set(Gf.Vec3d(n['camera_dx_px']*depth/fx, n['camera_dy_px']*depth/fx, 0.))
                row=dict(level=case['level'],stream=case['stream'],phase=n['phase'],
                    pool_radius_fraction=case['pool_radius_fraction'],transmission_color=n['color'],
                    absorption_depth_m=n['absorption_depth_m'],roughness=.035,meniscus_m=.0012,
                    ripple_m=.0002,stream_radius_m=.001,jet_length_fraction=.25,jet_wobble_m=.0002)
                points,faces,detail=station.body_mesh(row,station.radius,station.height)
                station.set_mesh(station.body,points,faces,detail['normal_group_faces'])
                station.body_translation.Set(Gf.Vec3d(float(station.center[0]),float(station.center[1]),station.base if row['level']>0 else station.base-10.))
                points,faces,_=station.stream_mesh(row,station.nozzle_z,station.base+max(row['level']*station.height,.00008))
                station.set_mesh(station.stream,points,faces)
                station.stream_translation.Set(Gf.Vec3d(float(station.nozzle_xy[0]),float(station.nozzle_xy[1]),0. if row['stream']!='off' else -10.))
                for shader in (station.body_shader,station.stream_shader):
                    station.set_transmission(shader,row,UsdGeom.GetStageMetersPerUnit(stage))
                for mesh,material in ((station.body,station.body_mat),(station.stream,station.stream_mat)):
                    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
                for vial in station.vials: station.visible(vial,True)
                station.guard.Set(station.guard_initial)
                app.update();station.capture();rgb=station.capture()
                for vial in station.vials:station.visible(vial,False)
                station.guard.Set('invisible')
                for mesh in (station.body,station.stream):
                    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(station.label_mat)
                app.update()
                for attempt in range(4):
                    station.capture();station.capture()
                    mask=semantic_mask(station.semantic.get_data(),'liquid__201')
                    if bool(mask.any()) == bool(row['level']>0):break
                else:raise RuntimeError('Geometric annotation identity failure: '+case['id'])
                stem=case['id'];image=a.out/'rgb'/(stem+'.png');label=a.out/'labels'/(stem+'.txt')
                Image.fromarray(rgb).save(image)
                Image.fromarray(mask.astype('uint8')*255).save(a.out/'masks'/(stem+'.png'))
                box=mask_box(mask)
                visibility=None;eligible=True
                if case['state']=='empty':empty_rgb[case['group']]=rgb.copy()
                if box is not None:
                    import cv2
                    region=cv2.dilate(mask.astype('uint8'),np.ones((11,11),np.uint8)).astype(bool)
                    diff=np.abs(rgb.astype(float)-empty_rgb[case['group']].astype(float))
                    mean=float(diff[region].mean());strong=int((diff.max(2)[region]>12).sum())
                    eligible=mean>3. and strong>=10
                    visibility=dict(paired_empty_mean_rgb_difference=mean,pixels_difference_gt12=strong,
                        screen_passed=eligible,meaning='Exploratory synthetic signal screen; not visible-pixel ground truth')
                if case['split']=='review_only':eligible=False
                text='' if box is None else '0 '+' '.join(f'{v:.8f}' for v in [(box[0]+box[2])/1280,(box[1]+box[3])/960,(box[2]-box[0])/640,(box[3]-box[1])/480])+'\n'
                label.write_text(text,encoding='utf-8')
                record=dict(case,parameters=row,image='rgb/'+image.name,label='labels/'+label.name,
                    image_sha256=sha(image),label_sha256=sha(label),box=box,mask_pixels=int(mask.sum()),
                    training_eligible=eligible,synthetic_visibility=visibility,
                    camera_to_world=np.asarray(cam.ComputeLocalToWorldTransform(station.Usd.TimeCode.Default())).T.tolist(),
                    annotation='Independent geometric body pass; not approved visible-pixel truth')
                log.write(json.dumps(record)+'\n');log.flush();records.append(record)
                print(json.dumps(dict(rendered=stem,box=box,elapsed_s=time.perf_counter()-started)),flush=True)
        after={str(p.relative_to(assets)):sha(p) for p in sorted(assets.rglob('*')) if p.is_file()}
        if before!=after:raise ValueError('Source assets changed')
        write(a.out/'completed-render.json',dict(images=len(records),source_assets_unchanged=True,
            source_assets_sha256=before,plan_sha256=sha(a.out/'render-plan.json'),manifest_sha256=sha(a.out/'manifest.jsonl'),
            small_body_boxes_le32=sum(r['box'] is not None and r['box'][3]-r['box'][1]<=32 for r in records),
            training_real_runs=[],run5_used=False,wall_s=time.perf_counter()-started))
    except BaseException as error:
        write(a.out/'failed-render.json',dict(type=type(error).__name__,error=str(error)))
        raise
    finally:app.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['prepare-render','launch','render','prepare-fit'])
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--assets',type=Path,default=Path('C:/git/FDS-Dark/ALL-Applied-AI-Network/msoe-ai-club-rockwell-automation-nvidia-hackath-austin-koske-s-team-8e90f213c6/datasets/mixing_twin/assets/csi_digital_twin_v1'))
    p.add_argument('--groups',type=int,default=48)
    p.add_argument('--renders',type=Path)
    p.add_argument('--tools',type=Path,default=Path('C:/Users/Austin/ai-club-tools'))
    p.add_argument('--spp',type=int,default=8)
    p.add_argument('--subframes',type=int,default=2)
    p.add_argument('--no-ros-env',action='store_true')
    args=p.parse_args()
    if args.action=='prepare-render':prepare_render(args)
    elif args.action=='launch':launch(args)
    elif args.action=='prepare-fit':prepare_fit(args)
    else:render(args)
