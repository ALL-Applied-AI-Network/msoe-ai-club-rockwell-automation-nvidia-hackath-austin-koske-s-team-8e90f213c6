"""Live Isaac station -> RGB socket -> frozen inference -> acknowledged next step.

New synthetic experiment only. Existing renderer/asset files remain untouched.
Liquid surfaces and optical parameters are procedural, not validated CFD.
"""
import argparse
import json
from pathlib import Path
import socket
import sys
import time
import traceback

from twin_contract import digest,episode_states,send,receive,validate_plan

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'setup/isaac'))
from box_annotations import sha256,write_json,semantic_mask,lens_mm

PREFIX='/FillStation/FillStation_LocalRoot/'
CAMERA=PREFIX+'FillCam/FillCam'


class Station:
    def __init__(self,app,assets,spp,subframes):
        import numpy as np
        import omni.usd
        import omni.timeline
        import omni.replicator.core as rep
        import carb.settings
        from pxr import Usd,UsdGeom,UsdShade,Gf,Sdf,UsdLux
        from isaacsim.core.experimental.utils.semantics import add_labels
        from liquid_geometry import body_mesh,stream_mesh,set_mesh,transmission_material,set_transmission
        self.app,self.np,self.rep=app,np,rep
        self.Usd,self.UsdGeom,self.UsdShade,self.Gf=Usd,UsdGeom,UsdShade,Gf
        self.body_mesh,self.stream_mesh,self.set_mesh,self.set_transmission=body_mesh,stream_mesh,set_mesh,set_transmission
        self.spp,self.subframes=spp,subframes
        self.settings=carb.settings.get_settings()
        self.settings.set('/omni/replicator/asyncRendering',False)
        self.settings.set('/rtx/pathtracing/maxBounces',12)
        self.settings.set('/rtx/pathtracing/maxSpecularAndTransmissionBounces',12)
        context=omni.usd.get_context()
        if not context.open_stage(str(assets/'stations/fill_station/fill_station.usda')):
            raise RuntimeError('Station stage failed to open')
        started=time.monotonic()
        while context.get_stage_loading_status()[2]:
            if time.monotonic()-started>120:raise TimeoutError('Station loading timed out')
            app.update()
        for _ in range(10):app.update()
        stage=context.get_stage();self.stage=stage
        stage.SetEditTarget(stage.GetSessionLayer())
        frozen=[(a,a.Get(686)) for p in stage.Traverse() for a in p.GetAttributes() if a.GetNumTimeSamples()]
        for attr,value in frozen:
            if value is not None:attr.Set(value)
        timeline=omni.timeline.get_timeline_interface();timeline.pause();timeline.set_current_time(0.)
        app.update()
        self.lights=[(p.GetAttribute('inputs:intensity'),float(p.GetAttribute('inputs:intensity').Get()))
            for p in stage.Traverse() if p.HasAPI(UsdLux.LightAPI) and p.GetAttribute('inputs:intensity').Get() is not None]
        spec=json.loads((assets/'cameras/FillCam.json').read_text())
        cam=UsdGeom.Camera(stage.GetPrimAtPath(CAMERA))
        world=np.asarray(UsdGeom.Xformable(cam).ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T
        np.testing.assert_allclose(world,spec['camera_to_world_in_station_file'],atol=1e-6)
        optics=lens_mm([cam.GetFocalLengthAttr().Get(),cam.GetHorizontalApertureAttr().Get(),cam.GetVerticalApertureAttr().Get()],UsdGeom.GetStageMetersPerUnit(stage))
        np.testing.assert_allclose(optics,[spec['focal_length_mm'],spec['horizontal_aperture_mm'],spec['vertical_aperture_mm']],atol=1e-5)
        def bounds(path):
            b=UsdGeom.BBoxCache(Usd.TimeCode.Default(),[UsdGeom.Tokens.default_]).ComputeWorldBound(stage.GetPrimAtPath(path)).ComputeAlignedRange()
            return np.asarray(b.GetMin()),np.asarray(b.GetMax())
        low,high=bounds(PREFIX+'Fill_liquid/Fill_liquid')
        self.radius=float((high[0]-low[0])/2);self.base=float(low[2]);self.height=float(high[2]-low[2]);self.center=(low+high)/2
        nl,nh=bounds(PREFIX+'Fill_nozzle_stainless/Fill_nozzle_stainless')
        self.nozzle_xy=(nl[:2]+nh[:2])/2;self.nozzle_z=float(nl[2])
        for path in ['Fill_liquid','Fill_dispensing_stream']:
            UsdGeom.Imageable(stage.GetPrimAtPath(PREFIX+path)).GetVisibilityAttr().Set('invisible')
        UsdGeom.Xform.Define(stage,'/SyntheticReview')
        self.body_mat,self.body_shader=transmission_material(stage,'LiveBody')
        self.stream_mat,self.stream_shader=transmission_material(stage,'LiveStream')
        self.label_mat=UsdShade.Material.Define(stage,'/SyntheticReview/Materials/LiveOpaqueLabel')
        label_shader=UsdShade.Shader.Define(stage,'/SyntheticReview/Materials/LiveOpaqueLabel/Surface')
        label_shader.CreateIdAttr('UsdPreviewSurface')
        label_shader.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(.8,.8,.8))
        label_shader.CreateInput('opacity',Sdf.ValueTypeNames.Float).Set(1.)
        self.label_mat.CreateSurfaceOutput().ConnectToSource(label_shader.ConnectableAPI(),'surface')
        UsdGeom.Xform.Define(stage,'/SyntheticReview/Liquid')
        UsdGeom.Xform.Define(stage,'/SyntheticReview/Stream')
        self.body=UsdGeom.Mesh.Define(stage,'/SyntheticReview/Liquid/Geometry')
        self.stream=UsdGeom.Mesh.Define(stage,'/SyntheticReview/Stream/Geometry')
        self.body_translation=self.body.AddTranslateOp();self.stream_translation=self.stream.AddTranslateOp()
        UsdShade.MaterialBindingAPI.Apply(self.body.GetPrim()).Bind(self.body_mat)
        UsdShade.MaterialBindingAPI.Apply(self.stream.GetPrim()).Bind(self.stream_mat)
        add_labels(stage.GetPrimAtPath('/SyntheticReview/Liquid'),labels=['liquid__201'],taxonomy='class')
        add_labels(stage.GetPrimAtPath('/SyntheticReview/Stream'),labels=['stream__301'],taxonomy='class')
        self.vials=[p for p in stage.Traverse() if p.GetTypeName()=='Mesh' and '/Fill_vial_' in str(p.GetPath())]
        self.guard=UsdGeom.Imageable(stage.GetPrimAtPath(PREFIX+'Fill_splash_guard/Cube_001')).GetVisibilityAttr()
        self.guard_initial=self.guard.Get()
        product=rep.create.render_product(CAMERA,(640,480))
        self.rgb=rep.AnnotatorRegistry.get_annotator('rgb');self.rgb.attach([product])
        self.semantic=rep.AnnotatorRegistry.get_annotator('semantic_segmentation');self.semantic.attach([product])
        rep.orchestrator.set_capture_on_play(False)
        self.camera=dict(supplied=spec,actual_camera_to_world=world.tolist(),
            actual_optics_mm=optics.tolist(),source_camera_sha256=sha256(assets/'cameras/FillCam.json'),
            height_envelope_m=self.height,radius_m=self.radius,physical_calibration=False)

    def visible(self,prim,yes):
        self.UsdGeom.Imageable(prim).GetVisibilityAttr().Set('inherited' if yes else 'invisible')

    def capture(self):
        self.settings.set('/rtx/pathtracing/spp',self.spp)
        self.settings.set('/rtx/pathtracing/totalSpp',self.spp)
        self.rep.orchestrator.step(rt_subframes=self.subframes,delta_time=0.,wait_for_render=True)
        return self.np.asarray(self.rgb.get_data())[...,:3].astype(self.np.uint8).copy()

    def render(self,state,plan,episode=None,*,labels=True):
        episode=episode or {}
        np=self.np;Gf=self.Gf
        for attr,intensity in self.lights:attr.Set(intensity*episode.get('light_scale',1.))
        actual=np.array(state['actual_components_ml']);total=float(actual.sum())
        fraction=actual/max(total,1e-9)
        absorbance=np.array([[.1,1.25,1.35],[1.35,.9,.15],[1.15,.1,1.4]])
        transmission=np.exp(-1.7*(fraction@absorbance.T)).tolist()
        row=dict(level=total/plan['vial_envelope_ml'],stream='continuous' if state['active_tank'] is not None else 'off',
            phase=state['sim_time_s']*4,pool_radius_fraction=.997 if total>1. else max(.25,.997*total),
            transmission_color=transmission,absorption_depth_m=episode.get('absorption_depth_m',.025),roughness=.035,
            meniscus_m=.0012,ripple_m=.0002,stream_radius_m=episode.get('stream_radius_m',.001))
        points,faces,detail=self.body_mesh(row,self.radius,self.height)
        self.set_mesh(self.body,points,faces,detail['normal_group_faces'])
        # Keep renderable identities alive across empty/fill resets. RTX can drop
        # semantic identity when an animated mesh is removed/reinserted by visibility.
        self.body_translation.Set(Gf.Vec3d(float(self.center[0]),float(self.center[1]),self.base if total>0 else self.base-10.))
        points,faces,_=self.stream_mesh(row,self.nozzle_z,self.base+max(row['level']*self.height,.00008))
        self.set_mesh(self.stream,points,faces)
        self.stream_translation.Set(Gf.Vec3d(float(self.nozzle_xy[0]),float(self.nozzle_xy[1]),0. if state['active_tank'] is not None else -10.))
        self.set_transmission(self.body_shader,row,self.UsdGeom.GetStageMetersPerUnit(self.stage))
        inventory=np.array(state['path_components_ml'])
        stream_row=dict(row,transmission_color=np.exp(-1.7*((inventory/max(inventory.sum(),1e-9))@absorbance.T)).tolist())
        self.set_transmission(self.stream_shader,stream_row,self.UsdGeom.GetStageMetersPerUnit(self.stage))
        self.visible(self.body.GetPrim(),True);self.visible(self.stream.GetPrim(),True)
        self.UsdShade.MaterialBindingAPI.Apply(self.body.GetPrim()).Bind(self.body_mat)
        self.UsdShade.MaterialBindingAPI.Apply(self.stream.GetPrim()).Bind(self.stream_mat)
        for vial in self.vials:self.visible(vial,True)
        self.guard.Set(self.guard_initial)
        self.app.update()
        # Two settled captures mirror the existing renderer's synchronization.
        self.capture();rgb=self.capture()
        if rgb.shape!=(480,640,3) or rgb.std()<.1:raise RuntimeError('Invalid station RGB')
        # Explicit synthetic sensor tests; transformations are not optical calibration.
        rgb=np.clip(np.rint(rgb.astype(float)*np.array(episode.get('sensor_gain',[1.,1.,1.]))),0,255).astype(np.uint8)
        if episode.get('anchor_occlusion') and 1.<=state['sim_time_s']<=4.:
            rgb[160:208,112:160]=(rgb[160:208,112:160].astype(float)*.75).astype(np.uint8)
        if not labels:
            return rgb,None,dict(row,geometry=detail,label_capture_attempts=0,label_capture='not_requested')
        # Independent geometric content pass is private evaluator evidence only.
        for vial in self.vials:self.visible(vial,False)
        self.guard.Set('invisible')
        for prim in [self.body.GetPrim(),self.stream.GetPrim()]:
            self.UsdShade.MaterialBindingAPI.Apply(prim).Bind(self.label_mat)
        self.app.update()
        for attempt in range(4):
            self.capture();self.capture()
            data=self.semantic.get_data();mask=semantic_mask(data,'liquid__201')
            if (total==0 and not mask.any()) or (total>0 and mask.any()):break
        else:
            raise RuntimeError('Geometric label pass failed after four unchanged captures; mapping='+
                json.dumps(data['info']['idToLabels'])+'; IDs='+str(np.unique(data['data']).tolist()))
        for vial in self.vials:self.visible(vial,True)
        self.guard.Set(self.guard_initial)
        self.UsdShade.MaterialBindingAPI.Apply(self.body.GetPrim()).Bind(self.body_mat)
        self.UsdShade.MaterialBindingAPI.Apply(self.stream.GetPrim()).Bind(self.stream_mat)
        return rgb,mask,dict(row,geometry=detail,label_capture_attempts=attempt+1)


def main(a):
    if a.out.exists():raise ValueError('Preserve existing simulation evidence; use a fresh output')
    a.out.mkdir(parents=True)
    plan=json.loads(a.plan.read_text());endpoint=json.loads(a.endpoint.read_text())
    validate_plan(plan)
    if endpoint['host']!='127.0.0.1':raise ValueError('Local simulation only')
    before={str(p.relative_to(a.assets)):sha256(p) for p in sorted(a.assets.rglob('*')) if p.is_file()}
    write_json(a.out/'plan.json',plan)
    from isaacsim import SimulationApp
    app=SimulationApp(dict(headless=not a.gui,renderer='PathTracing',width=640,height=480,
        multi_gpu=False,extra_args=['--/app/extensions/registryEnabled=false',
        '--/telemetry/enableAnonymousData=false','--/crashreporter/enabled=false']))
    start=time.perf_counter()
    try:
        from PIL import Image
        from isaacsim.core.version import get_version
        station=Station(app,a.assets,a.spp,a.subframes)
        viewport_camera=None
        if a.gui:
            from omni.kit.viewport.utility import get_active_viewport,create_viewport_window
            viewport=get_active_viewport();viewport_window=None
            if viewport is None:
                viewport_window=create_viewport_window('Live liquid twin',width=960,height=720,camera_path=CAMERA)
                viewport=viewport_window.viewport_api if viewport_window else None
            if viewport is None:raise RuntimeError('Requested native viewport is unavailable')
            viewport.camera_path=CAMERA
            app.update()
            viewport_camera=str(viewport.camera_path)
            if viewport_camera!=CAMERA:raise RuntimeError('Native viewport camera differs from supplied FillCam')
            print('NATIVE VIEWPORT CAMERA',viewport_camera,flush=True)
        write_json(a.out/'camera.json',station.camera)
        write_json(a.out/'runtime.json',dict(isaac_version=list(get_version()),python=sys.version,
            renderer='PathTracing',spp=a.spp,subframes=a.subframes,gui=a.gui,native_viewport_camera=viewport_camera))
        for folder in ['rgb','geometric-masks']:(a.out/folder).mkdir()
        summary=[]
        with socket.create_connection((endpoint['host'],endpoint['port']),timeout=180.) as conn, (a.out/'private-truth.jsonl').open('w') as truthlog:
            conn.settimeout(180.)
            for episode_number,episode in enumerate(plan['episodes']):
                public_id=f'episode-{episode_number:03d}'
                send(conn,dict(type='start_episode',episode_id=public_id,reference=episode['reference']))
                reply,_=receive(conn)
                if reply!=dict(type='episode_ready',episode_id=public_id):raise ValueError('Unexpected episode response')
                records=[]
                for state in episode_states(plan,episode,a.limit_frames):
                    t0=time.perf_counter();rgb,mask,parameters=station.render(state,plan,episode)
                    pixels=rgb.tobytes();frame=state['frame_index'];stem=f'{public_id}-{frame:05d}'
                    Image.fromarray(rgb).save(a.out/'rgb'/(stem+'.png'))
                    Image.fromarray(mask.astype('uint8')*255).save(a.out/'geometric-masks'/(stem+'.png'))
                    header={k:state[k] for k in ['frame_index','sim_time_s','phase_key','recipe_key','metered_ml','dispense_complete']}
                    header.update(schema_version=1,type='frame',session_id='synthetic-live-v1',episode_id=public_id,
                        rgb_sha256=digest(pixels),shape=[480,640,3])
                    sent_at=time.perf_counter();send(conn,header,pixels)
                    reply,payload=receive(conn)
                    if payload or any(reply.get(k)!=header[k] for k in ['episode_id','frame_index','rgb_sha256']) or reply.get('type')!='frame_ack':
                        raise ValueError('Missing or mismatched inference acknowledgement')
                    if reply.get('simulated_control_command')!='none':raise ValueError('No autonomous intervention is authorized in this prototype')
                    record=dict(episode_id=public_id,scenario=episode['id'],episode_parameters=episode,state=state,parameters=parameters,
                        rgb_file='rgb/'+stem+'.png',mask_file='geometric-masks/'+stem+'.png',rgb_sha256=digest(pixels),
                        geometric_mask_pixels=int(mask.sum()),ack=reply,
                        render_save_s=sent_at-t0,inference_roundtrip_s=time.perf_counter()-sent_at,
                        frame_wall_s=time.perf_counter()-t0)
                    truthlog.write(json.dumps(record,allow_nan=False)+'\n');truthlog.flush();records.append(record)
                    print('LIVE ACK',public_id,frame,reply['sample_qa_valid'],flush=True)
                send(conn,dict(type='end_episode'));reply,_=receive(conn)
                if reply!=dict(type='episode_ended',episode_id=public_id):raise ValueError('Episode did not complete')
                summary.append(dict(episode_id=public_id,scenario=episode['id'],frames=len(records),
                    final_components_ml=records[-1]['state']['actual_components_ml'],scripted_flush_ml=episode['flush_ml']))
            send(conn,dict(type='finish'));reply,_=receive(conn)
            if reply!=dict(type='finished'):raise ValueError('Observer did not finish')
        after={str(p.relative_to(a.assets)):sha256(p) for p in sorted(a.assets.rglob('*')) if p.is_file()}
        if before!=after:raise RuntimeError('Source assets changed')
        write_json(a.out/'completed.json',dict(episodes=summary,source_assets_sha256=before,
            source_assets_unchanged=True,live_inference_ack_every_frame=True,wall_seconds=time.perf_counter()-start,
            hardware_actions=0,model_actions=0,interventions='predeclared simulated operator flush only',
            physical_calibration=False,run5_used=False))
    except BaseException as error:
        write_json(a.out/'failed.json',dict(error=type(error).__name__,message=str(error),traceback=traceback.format_exc()))
        traceback.print_exc();sys.stdout.flush();sys.stderr.flush()
        raise
    finally:app.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['out','plan','assets','endpoint']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--limit-frames',type=int)
    p.add_argument('--spp',type=int,default=8)
    p.add_argument('--subframes',type=int,default=2)
    p.add_argument('--gui',action='store_true',help='Optional native Isaac window, not a browser preview')
    p.add_argument('--no-ros-env',action='store_true',help=argparse.SUPPRESS)
    main(p.parse_args())
