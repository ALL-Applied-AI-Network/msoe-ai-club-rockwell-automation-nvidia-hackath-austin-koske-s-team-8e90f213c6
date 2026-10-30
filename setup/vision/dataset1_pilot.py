"""Synthetic-only, target-slot Dataset 1 pilot; GPU operations run on GB10.

The three labels describe rendered geometry, not human-reviewed real acceptance.
Authored scene layers remain unchanged; rendering edits use the session layer.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import time

CLASSES = ['pass', 'missing_cap', 'no_vial']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def read(path):
    return json.loads(Path(path).read_text())


def prepare(args):
    root = args.root
    if (root/'experiment-plan.json').exists():
        raise ValueError('Preserve prior plans; do not duplicate this pilot')
    inputs = root/'input'
    for relative, expected in read(inputs/'input-manifest.json').items():
        if sha(inputs/relative) != expected:
            raise ValueError('Input differs: '+relative)
    rng = random.Random(3337)
    families = []
    for group in range(60):
        labels = CLASSES*10
        rng.shuffle(labels)
        color = rng.choice([[.85, .75, .1], [.12, .4, .8], [.6, .15, .1], [.65, .65, .65]])
        families.append(dict(id=f'scene-{group:03d}', group=group,
            split='train' if group < 36 else 'val' if group < 48 else 'test',
            labels={f'G{i+1:02d}': label for i, label in enumerate(labels)},
            light_scale=rng.uniform(.7, 1.35), cap_color=[max(.03, min(.95, c*rng.uniform(.8, 1.15))) for c in color],
            camera_dx_px=rng.uniform(-4, 4), camera_dy_px=rng.uniform(-4, 4)))
    plan = dict(schema='dataset1-synthetic-pilot/v1', task='G fixture occupancy/cap-presence at explicit target slots',
        classes=CLASSES, deferred_classes=['crooked_cap', 'damaged_vial'], families=families,
        seed=3337, model_seed=19, groups={'train':36, 'val':12, 'test':12}, crops_per_scene=30,
        source_stage_sha256=sha(inputs/'datasets/csi_digital_twin_v1/stations/tray_inspection/tray_inspection.usdc'),
        source_camera_sha256=sha(inputs/'datasets/csi_digital_twin_v1/cameras/CapSealCam.json'),
        helper_sha256=sha(__file__), camera='Supplied CapSealCam overscan and radial distortion; recorded image-plane translation.',
        crop='48 x 48 native pixels about projected authored G cap center, resized to64x64; explicit target identity; no marker in image.',
        models=['rgb', 'paired_reference'], architecture='Random-initialized Conv32/64/128 with GroupNorm, central+global features, three-class head.',
        generic_pretraining=False, epochs=30, batch_size=64, optimizer='AdamW lr.001 weight_decay.0001',
        checkpoint_selection='Highest synthetic development macro recall per epoch; earliest tie. Synthetic test read only after both checkpoints freeze.',
        threshold_selection='For defect score 1-P(pass), choose largest development threshold catching every known synthetic missing-cap defect; freeze before test.',
        no_vial_policy='Report occupancy separately; missing placement requires external expected-position/order context.',
        training_real_runs=[], selection_real_runs=[], real_data_used=False, run5_used=False,
        split_policy='Whole scene family and its golden image/crops remain in one split. All use the same CAD and same30 positions; no claim of new-geometry/position generalization.',
        renderer=dict(spp=16, subframes=4, width=768, height=576),
        limitations=['Three-class geometry pilot, not full five-class Dataset1 acceptance.',
            'Paired golden contains all correctly capped G slots under matched simulated nuisance. Real golden acquisition is not implemented.',
            'No real Dataset1 transfer metrics or cross-station physical identity. No equipment commands.'])
    write(root/'experiment-plan.json', plan)
    print(json.dumps(dict(prepared=True, families=60, crops=1800, plan_sha256=sha(root/'experiment-plan.json'))), flush=True)


def render(args):
    import fcntl
    import os
    root = args.root
    plan = read(root/'experiment-plan.json')
    if sha(__file__) != plan['helper_sha256']:
        raise ValueError('Frozen renderer helper differs')
    out = root/('smoke' if args.smoke else 'rendered')
    out.mkdir(exist_ok=False)
    lock = (Path.home()/'ai-club-tools/spark-gpu.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assets = root/'input/datasets/csi_digital_twin_v1'
    stage_file = assets/'stations/tray_inspection/tray_inspection.usdc'
    if sha(stage_file) != plan['source_stage_sha256']:
        raise ValueError('Scene changed')
    from isaacsim import SimulationApp
    app = SimulationApp(dict(headless=True, renderer='PathTracing', width=768, height=576, multi_gpu=False,
        extra_args=['--/app/extensions/registryEnabled=false', '--/telemetry/enableAnonymousData=false', '--/crashreporter/enabled=false']))
    try:
        import carb
        import cv2
        import numpy as np
        import omni.usd
        import omni.replicator.core as rep
        from PIL import Image
        from pxr import Gf, Usd, UsdGeom, UsdLux, UsdShade
        settings = carb.settings.get_settings()
        settings.set('/omni/replicator/asyncRendering', False)
        settings.set('/rtx/pathtracing/spp', plan['renderer']['spp'])
        settings.set('/rtx/pathtracing/totalSpp', plan['renderer']['spp'])
        settings.set('/rtx/pathtracing/maxBounces', 12)
        settings.set('/rtx/pathtracing/maxSpecularAndTransmissionBounces', 12)
        context = omni.usd.get_context()
        if not context.open_stage(str(stage_file)):
            raise RuntimeError('Cannot open tray station')
        deadline = time.monotonic()+120
        while context.get_stage_loading_status()[2]:
            if time.monotonic() > deadline:
                raise TimeoutError('Scene loading timed out')
            app.update()
        for _ in range(10):
            app.update()
        stage = context.get_stage()
        stage.SetEditTarget(stage.GetSessionLayer())
        camera_spec = read(assets/'cameras/CapSealCam.json')
        camera_path = camera_spec['prim_in_station_file']
        camera = UsdGeom.Camera(stage.GetPrimAtPath(camera_path))
        original_world = UsdGeom.Xformable(camera).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        parent_inverse = UsdGeom.Xformable(camera.GetPrim().GetParent()).ComputeLocalToWorldTransform(Usd.TimeCode.Default()).GetInverse()
        camera.ClearXformOpOrder()
        camera_op = camera.AddTransformOp(opSuffix='pilot')
        camera_op.Set(original_world*parent_inverse)
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_], useExtentsHint=False)
        centers, parts = {}, {}
        for i in range(1, 31):
            slot = f'G{i:02d}'
            cap = stage.GetPrimAtPath(f'/TrayInspection/{slot}_cap')
            body = stage.GetPrimAtPath(f'/TrayInspection/{slot}_tube')
            if not cap or not body:
                raise ValueError('Missing supplied geometry: '+slot)
            bounds = cache.ComputeWorldBound(cap).ComputeAlignedRange()
            centers[slot] = (np.array(bounds.GetMin())+np.array(bounds.GetMax()))/2
            parts[slot] = (UsdGeom.Imageable(cap), UsdGeom.Imageable(body))
        lights = [(p.GetAttribute('inputs:intensity'), float(p.GetAttribute('inputs:intensity').Get()))
                  for p in stage.Traverse() if p.HasAPI(UsdLux.LightAPI) and p.GetAttribute('inputs:intensity').Get() is not None]
        cap_material = UsdShade.Material(stage.GetPrimAtPath('/TrayInspection/_materials/tw_cap_yellow'))
        source = cap_material.GetSurfaceOutput().GetConnectedSource()
        shader = UsdShade.Shader(source[0].GetPrim())
        color_input = shader.GetInput('diffuseColor')
        if not color_input:
            raise ValueError('Cannot identify cap diffuse input')
        product = rep.create.render_product(camera_path, (768,576))
        annotator = rep.AnnotatorRegistry.get_annotator('rgb')
        annotator.attach([product])
        rep.orchestrator.set_capture_on_play(False)
        app.update()
        clip = camera.GetClippingRangeAttr()
        near_far = clip.Get()
        clip.Set(Gf.Vec2f(near_far[0]*.5, near_far[1])); app.update()
        clip.Set(near_far); app.update()
        f = camera_spec['focal_px_final']; k1 = camera_spec['lens_k1']
        K = np.array([[f,0,320],[0,f,240],[0,0,1.]],dtype=np.float64)
        D = np.array([k1,0,0,0,0.])
        R = np.array(camera_spec['R_world_to_cv'])
        rvec, _ = cv2.Rodrigues(R)
        position = np.array(camera_spec['position_m'])
        np.testing.assert_allclose(np.array(original_world)[3,:3], position, atol=1e-6)
        u,v = np.meshgrid(np.arange(640,dtype=np.float32),np.arange(480,dtype=np.float32))
        uv = cv2.undistortPoints(np.stack([u.ravel(),v.ravel()],1)[:,None,:],K,D).reshape(480,640,2)
        mapx = (uv[...,0]*f+384).astype(np.float32)
        mapy = (uv[...,1]*f+288).astype(np.float32)
        manifest = []
        for family in (plan['families'][:1] if args.smoke else plan['families']):
            camera_delta = R[0]*family['camera_dx_px']*1.1/f + R[1]*family['camera_dy_px']*1.1/f
            new_position = position+camera_delta
            world = Gf.Matrix4d(original_world)
            world.SetTranslateOnly(Gf.Vec3d(*new_position))
            camera_op.Set(world*parent_inverse)
            for attr, base in lights:
                attr.Set(base*family['light_scale'])
            color_input.Set(Gf.Vec3f(*family['cap_color']))
            points = np.array([centers[f'G{i:02d}'] for i in range(1,31)])
            pixels, _ = cv2.projectPoints(points,rvec,-R@new_position,K,D)
            boxes = {}
            for i,(x,y) in enumerate(pixels.reshape(-1,2),1):
                x,y = int(round(x)),int(round(y))
                box = [x-24,y-24,x+24,y+24]
                if not(0<=box[0]<box[2]<=640 and 0<=box[1]<box[3]<=480):
                    raise ValueError('Target crop outside source image')
                boxes[f'G{i:02d}']=box
            for variant in ('golden','inspection'):
                for slot,(cap,body) in parts.items():
                    label = 'pass' if variant=='golden' else family['labels'][slot]
                    cap.GetVisibilityAttr().Set('inherited' if label=='pass' else 'invisible')
                    body.GetVisibilityAttr().Set('invisible' if label=='no_vial' else 'inherited')
                app.update()
                rep.orchestrator.step(rt_subframes=plan['renderer']['subframes'],delta_time=0.,wait_for_render=True)
                rgb=np.asarray(annotator.get_data())[...,:3].astype(np.uint8).copy()
                if rgb.shape!=(576,768,3) or rgb.std()<2:
                    raise ValueError('Bad RGB capture')
                distorted=cv2.remap(rgb,mapx,mapy,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
                path=out/f'{family["id"]}-{variant}.png'
                Image.fromarray(distorted).save(path)
                if variant=='golden':
                    golden_hash=sha(path)
                else:
                    manifest.append(dict(id=family['id'],group=family['group'],split=family['split'],
                        image=path.name,image_sha256=sha(path),golden=f'{family["id"]}-golden.png',golden_sha256=golden_hash,
                        boxes=boxes,labels=family['labels'],camera_position=new_position.tolist(),nuisance=family,
                        target_definition='Projected authored G cap centers; full-scene context retained; geometry-generated labels.'))
            print('RENDERED',family['id'],flush=True)
        (out/'manifest.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in manifest))
        assert sha(stage_file)==plan['source_stage_sha256']
        write(out/'completed.json',dict(families=len(manifest),images=2*len(manifest),source_stage_unchanged=True,
            plan_sha256=sha(root/'experiment-plan.json'),helper_sha256=sha(__file__),manifest_sha256=sha(out/'manifest.jsonl')))
    except BaseException as error:
        write(out/'failed.json',dict(type=type(error).__name__,error=str(error)))
        app.close(exit_code=1)
        raise
    app.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','render'])
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--smoke',action='store_true')
    args,kit_args=parser.parse_known_args()
    args.root=args.root.resolve()
    # Kit also parses argv. Keep our --root and experiment action out of its
    # application options; retain explicitly passed Kit arguments only.
    import sys
    sys.argv=[sys.argv[0],*kit_args]
    (prepare if args.action=='prepare' else render)(args)


if __name__=='__main__':
    main()
