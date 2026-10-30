"""Synthetic five-class, two-fixture extension. Render only on the GB10.

Original layers are never saved. Defect meshes are controlled geometric proxies,
not measurements of real damage. Real imagery never enters plan or fitting.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import time

CLASSES=['pass','missing_cap','crooked_cap','damaged_vial','no_vial']
SLOTS=[f'{fixture}{i:02d}' for fixture in ('G','S') for i in range(1,31)]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def families(seed=62319,counts=(72,24,24)):
    rng=random.Random(seed);rows=[]
    for i in range(sum(counts)):
        labels={}
        for fixture in ('G','S'):
            values=CLASSES*6;rng.shuffle(values)
            labels.update({f'{fixture}{j+1:02d}':v for j,v in enumerate(values)})
        # Body lean is shared by every class, so lean alone cannot define crooked.
        parameters={slot:dict(body_lean_deg=rng.choice([0,0,8,18,30]),body_azimuth_deg=rng.uniform(0,360),
            cap_tilt_deg=rng.uniform(12,32),cap_tilt_azimuth_deg=rng.uniform(0,360),cap_lift_m=rng.uniform(.001,.004),
            damage_kind=rng.choice(['chip','dent']),damage_azimuth_deg=rng.uniform(0,360),damage_strength=rng.uniform(.25,.52)) for slot in SLOTS}
        base=rng.choice([[.85,.75,.1],[.12,.4,.8],[.6,.15,.1],[.65,.65,.65]])
        rows.append(dict(id=f'scene-{i:03d}',group=i,split='train' if i<counts[0] else 'val' if i<sum(counts[:2]) else 'test',
            labels=labels,parameters=parameters,light_scale=rng.uniform(.65,1.4),
            cap_color=[max(.04,min(.95,c*rng.uniform(.85,1.1))) for c in base],
            camera_dx_px=rng.uniform(-3,3),camera_dy_px=rng.uniform(-3,3)))
    return rows


def prepare(root):
    root.mkdir(parents=True,exist_ok=True)
    if (root/'experiment-plan.json').exists():raise ValueError('Preserve existing plan')
    source=root/'input/datasets/csi_digital_twin_v1'
    plan=dict(schema='dataset1-fiveclass/v1',classes=CLASSES,slots=SLOTS,families=families(),
        groups=dict(train=72,val=24,test=24),seed=62319,model_seed=19,
        source_stage_sha256=sha(source/'stations/tray_inspection/tray_inspection.usdc'),
        source_camera_sha256=sha(source/'cameras/CapSealCam.json'),renderer_sha256=sha(__file__),
        renderer=dict(spp=16,subframes=4,width=768,height=576),crop_pixels=96,model_pixels=96,
        models=['rgb','target_rgb','target_paired'],epochs=35,batch_size=128,
        target_identity='Fixed aperture plane, source pixel coordinates; optional Gaussian target-conditioning channel. No labels painted into RGB.',
        second_fixture='Session-only canonical G vial copies at tray-relative S aperture positions; original scattered S objects hidden.',
        defects=dict(crooked_cap='Cap-only relative tilt/lift; vial lean independently varied across all classes.',
            damaged_vial='Upper-wall chip or dent proxy, without class-specific material/color; finite geometry only.'),
        checkpoint_selection='Maximum synthetic validation macro recall, earliest tie; all arms freeze before test decoding.',
        threshold_selection='Minimum synthetic validation 1-P(pass) across all occupied defect classes; freeze before test.',
        training_real_runs=[],selection_real_runs=[],run5_used=False,generic_pretraining=False,
        limits=['Geometric chip/dent proxies do not span all physical damage.',
            'Same CAD/camera/positions across splits; unseen station geometry not established.',
            'Real labels, slot numbering, visibility and S mapping are not approved by rendering.',
            'No physical placement signals or camera connections; placement module is opt-in.'])
    write(root/'experiment-plan.json',plan)
    print(json.dumps(dict(prepared=True,families=120,targets=7200,plan_sha256=sha(root/'experiment-plan.json'))),flush=True)


def render(root,smoke=False):
    import fcntl
    lock=(Path.home()/'ai-club-tools/spark-gpu.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    plan=read(root/'experiment-plan.json')
    if sha(__file__)!=plan['renderer_sha256']:raise ValueError('Frozen renderer differs')
    assets=root/'input/datasets/csi_digital_twin_v1';stage_file=assets/'stations/tray_inspection/tray_inspection.usdc'
    if sha(stage_file)!=plan['source_stage_sha256']:raise ValueError('Stage differs')
    if sha(assets/'cameras/CapSealCam.json')!=plan['source_camera_sha256']:raise ValueError('Camera differs')
    out=root/('smoke-v1' if smoke else 'rendered-v1');out.mkdir(exist_ok=False)
    from isaacsim import SimulationApp
    app=SimulationApp(dict(headless=True,renderer='PathTracing',width=768,height=576,multi_gpu=False,
        extra_args=['--/app/extensions/registryEnabled=false','--/telemetry/enableAnonymousData=false','--/crashreporter/enabled=false']))
    try:
        import carb
        import cv2
        import numpy as np
        import omni.usd
        import omni.replicator.core as rep
        from PIL import Image
        from pxr import Gf,Sdf,Usd,UsdGeom,UsdLux,UsdShade,Vt
        settings=carb.settings.get_settings()
        settings.set('/omni/replicator/asyncRendering',False)
        settings.set('/rtx/pathtracing/spp',plan['renderer']['spp'])
        settings.set('/rtx/pathtracing/totalSpp',plan['renderer']['spp'])
        settings.set('/rtx/pathtracing/maxBounces',12)
        settings.set('/rtx/pathtracing/maxSpecularAndTransmissionBounces',12)
        context=omni.usd.get_context()
        if not context.open_stage(str(stage_file)):raise RuntimeError('Cannot open station')
        deadline=time.monotonic()+120
        while context.get_stage_loading_status()[2]:
            if time.monotonic()>deadline:raise TimeoutError('Scene loading timed out')
            app.update()
        for _ in range(10):app.update()
        stage=context.get_stage();stage.SetEditTarget(stage.GetSessionLayer())
        cache=UsdGeom.BBoxCache(Usd.TimeCode.Default(),[UsdGeom.Tokens.default_],useExtentsHint=False)
        good=UsdGeom.Xformable(stage.GetPrimAtPath('/TrayInspection/tray_good_top')).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        scrap=UsdGeom.Xformable(stage.GetPrimAtPath('/TrayInspection/tray_scrap_top')).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        parent_world=UsdGeom.Xformable(stage.GetPrimAtPath('/TrayInspection')).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        # Hide authored scattered S examples; add fresh, attributed clones in session.
        for prim in list(stage.GetPrimAtPath('/TrayInspection').GetChildren()):
            if prim.GetName().startswith('S') and prim.GetName().endswith(('_cap','_tube')):
                UsdGeom.Imageable(prim).GetVisibilityAttr().Set('invisible')
        parts={};apertures={};cap_centers={}
        for i in range(1,31):
            g=f'G{i:02d}';body=stage.GetPrimAtPath(f'/TrayInspection/{g}_tube');cap=stage.GetPrimAtPath(f'/TrayInspection/{g}_cap')
            bw=UsdGeom.Xformable(body).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            cw=UsdGeom.Xformable(cap).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            base=np.array(bw.ExtractTranslation());bounds=cache.ComputeWorldBound(cap).ComputeAlignedRange()
            cap_center=(np.array(bounds.GetMin())+np.array(bounds.GetMax()))/2
            for prefix in ('G','S'):
                slot=f'{prefix}{i:02d}';mapping=Gf.Matrix4d(1) if prefix=='G' else good.GetInverse()*scrap
                world_body=bw*mapping;world_cap=cw*mapping
                center=np.array(mapping.Transform(Gf.Vec3d(*cap_center)))
                aperture=np.array(world_body.ExtractTranslation());aperture[2]=float(scrap.ExtractTranslation()[2] if prefix=='S' else good.ExtractTranslation()[2])
                apertures[slot]=aperture;cap_centers[slot]=center
                entries={}
                for kind,original,world in [('cap',cap,world_cap),('tube',body,world_body)]:
                    path=f'/TrayInspection/Experiment_{slot}_{kind}'
                    if not Sdf.CopySpec(stage.GetRootLayer(),original.GetPath(),stage.GetSessionLayer(),Sdf.Path(path)):
                        raise ValueError('Cannot clone geometry')
                    clone=stage.GetPrimAtPath(path);clone.SetActive(True);UsdGeom.Imageable(clone).GetVisibilityAttr().Set('inherited')
                    xf=UsdGeom.Xformable(clone);xf.ClearXformOpOrder();op=xf.AddTransformOp(opSuffix='experiment')
                    op.Set(world*parent_world.GetInverse())
                    mesh=next(p for p in clone.GetChildren() if p.IsA(UsdGeom.Mesh))
                    entries[kind]=dict(prim=clone,op=op,world=world,mesh=UsdGeom.Mesh(mesh),
                        points=np.asarray(UsdGeom.Mesh(mesh).GetPointsAttr().Get(),dtype=np.float64))
                parts[slot]=entries
            # Original G shapes replaced by the attributed clone as well.
            UsdGeom.Imageable(body).GetVisibilityAttr().Set('invisible');UsdGeom.Imageable(cap).GetVisibilityAttr().Set('invisible')
        # No source geometry or surface color encodes the target class.
        lights=[(p.GetAttribute('inputs:intensity'),float(p.GetAttribute('inputs:intensity').Get())) for p in stage.Traverse()
            if p.HasAPI(UsdLux.LightAPI) and p.GetAttribute('inputs:intensity').Get() is not None]
        material=UsdShade.Material(stage.GetPrimAtPath('/TrayInspection/_materials/tw_cap_yellow'))
        shader=UsdShade.Shader(material.GetSurfaceOutput().GetConnectedSource()[0].GetPrim())
        cap_color=shader.GetInput('diffuseColor')
        spec=read(assets/'cameras/CapSealCam.json');camera_path=spec['prim_in_station_file']
        camera=UsdGeom.Camera(stage.GetPrimAtPath(camera_path));camera_xf=UsdGeom.Xformable(camera)
        camera_world=camera_xf.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        camera_parent=UsdGeom.Xformable(camera.GetPrim().GetParent()).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        camera_xf.ClearXformOpOrder();camera_op=camera_xf.AddTransformOp(opSuffix='experiment')
        camera_op.Set(camera_world*camera_parent.GetInverse())
        f=spec['focal_px_final'];R=np.array(spec['R_world_to_cv']);position=np.array(spec['position_m'])
        K=np.array([[f,0,320],[0,f,240],[0,0,1.]],np.float64);D=np.array([spec['lens_k1'],0,0,0,0.])
        rvec,_=cv2.Rodrigues(R);u,v=np.meshgrid(np.arange(640,dtype=np.float32),np.arange(480,dtype=np.float32))
        uv=cv2.undistortPoints(np.stack([u.ravel(),v.ravel()],1)[:,None,:],K,D).reshape(480,640,2)
        mapx=(uv[...,0]*f+384).astype(np.float32);mapy=(uv[...,1]*f+288).astype(np.float32)
        product=rep.create.render_product(camera_path,(768,576));annotator=rep.AnnotatorRegistry.get_annotator('rgb');annotator.attach([product])
        rep.orchestrator.set_capture_on_play(False);app.update()
        clip=camera.GetClippingRangeAttr();near_far=clip.Get();clip.Set(Gf.Vec2f(near_far[0]*.5,near_far[1]));app.update();clip.Set(near_far);app.update()

        def around(point,axis,degrees):
            a=Gf.Matrix4d(1);a.SetTranslate(Gf.Vec3d(*(-point)))
            b=Gf.Matrix4d(1);b.SetRotate(Gf.Rotation(Gf.Vec3d(*axis),degrees))
            c=Gf.Matrix4d(1);c.SetTranslate(Gf.Vec3d(*point))
            return a*b*c

        rows=[]
        for family in (plan['families'][:1] if smoke else plan['families']):
            delta=R[0]*family['camera_dx_px']*1.1/f+R[1]*family['camera_dy_px']*1.1/f
            new_position=position+delta;world=Gf.Matrix4d(camera_world);world.SetTranslateOnly(Gf.Vec3d(*new_position));camera_op.Set(world*camera_parent.GetInverse())
            for attr,base in lights:attr.Set(base*family['light_scale'])
            cap_color.Set(Gf.Vec3f(*family['cap_color']))
            points=np.array([apertures[slot] for slot in SLOTS]);pixels,_=cv2.projectPoints(points,rvec,-R@new_position,K,D)
            boxes={};targets={}
            for slot,(x,y) in zip(SLOTS,pixels.reshape(-1,2)):
                x,y=int(round(x)),int(round(y));box=[x-48,y-48,x+48,y+48]
                if not(0<=box[0]<box[2]<=640 and 0<=box[1]<box[3]<=480):raise ValueError('Target crop outside source')
                boxes[slot]=box;targets[slot]=[x,y]
            for variant in ('golden','inspection'):
                for slot,entries in parts.items():
                    params=family['parameters'][slot];label='pass' if variant=='golden' else family['labels'][slot]
                    angle=np.deg2rad(params['body_azimuth_deg']);axis=np.array([np.cos(angle),np.sin(angle),0.])
                    lean=around(apertures[slot],axis,params['body_lean_deg'])
                    for kind,entry in entries.items():
                        world=Gf.Matrix4d(entry['world'])
                        if kind=='cap' and label=='crooked_cap':
                            a=np.deg2rad(params['cap_tilt_azimuth_deg']);cap_axis=np.array([np.cos(a),np.sin(a),0.])
                            pivot=np.array(world.ExtractTranslation());world=world*around(pivot,cap_axis,params['cap_tilt_deg'])
                            trans=np.array(world.ExtractTranslation());trans[2]+=params['cap_lift_m'];world.SetTranslateOnly(Gf.Vec3d(*trans))
                        entry['op'].Set(world*lean*parent_world.GetInverse())
                        hidden=label=='no_vial' or (kind=='cap' and label=='missing_cap')
                        UsdGeom.Imageable(entry['prim']).GetVisibilityAttr().Set('invisible' if hidden else 'inherited')
                        if kind=='tube':
                            pts=entry['points'].copy()
                            if label=='damaged_vial':
                                direction=np.deg2rad(params['damage_azimuth_deg']);angles=np.arctan2(pts[:,1],pts[:,0])
                                distance=np.abs(np.arctan2(np.sin(angles-direction),np.cos(angles-direction)))
                                wedge=(distance<.65)&(pts[:,2]>1.8)
                                if params['damage_kind']=='chip':
                                    pts[wedge,2]-=(pts[wedge,2]-1.8)*params['damage_strength']
                                else:
                                    pts[wedge,:2]*=(1-params['damage_strength'])
                            mesh=entry['mesh'];mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(pts.astype(np.float32)))
                            mesh.GetExtentAttr().Set(UsdGeom.PointBased.ComputeExtent(Vt.Vec3fArray.FromNumpy(pts.astype(np.float32))))
                            mesh.GetNormalsAttr().Block()
                app.update();rep.orchestrator.step(rt_subframes=plan['renderer']['subframes'],delta_time=0.,wait_for_render=True)
                rgb=np.asarray(annotator.get_data())[...,:3].astype(np.uint8).copy()
                if rgb.shape!=(576,768,3) or rgb.std()<2:raise ValueError('Bad RGB capture')
                final=cv2.remap(rgb,mapx,mapy,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
                path=out/f'{family["id"]}-{variant}.png';Image.fromarray(final).save(path)
                if variant=='golden':golden_hash=sha(path)
                else:rows.append(dict(id=family['id'],group=family['group'],split=family['split'],image=path.name,image_sha256=sha(path),
                    golden=f'{family["id"]}-golden.png',golden_sha256=golden_hash,boxes=boxes,targets_xy=targets,
                    labels=family['labels'],nuisance=family,source_dimensions=[640,480],data_domain='synthetic',target_mapping_approved=False))
            (out/'manifest.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
            print('RENDERED',family['id'],flush=True)
        if sha(stage_file)!=plan['source_stage_sha256']:raise ValueError('Source layer changed')
        write(out/'completed.json',dict(families=len(rows),images=2*len(rows),targets=60*len(rows),
            plan_sha256=sha(root/'experiment-plan.json'),manifest_sha256=sha(out/'manifest.jsonl'),source_stage_unchanged=True,renderer_sha256=sha(__file__)))
    except BaseException as error:
        write(out/'failed.json',dict(type=type(error).__name__,error=str(error)));app.close(exit_code=1);raise
    app.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','render']);p.add_argument('--root',type=Path,required=True);p.add_argument('--smoke',action='store_true')
    args,kit_args=p.parse_known_args();args.root=args.root.resolve()
    import sys
    sys.argv=[sys.argv[0],*kit_args]
    prepare(args.root) if args.action=='prepare' else render(args.root,args.smoke)
