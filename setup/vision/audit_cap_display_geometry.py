"""CPU CAD audit of frozen cap poses and aperture-versus-cap display anchors.

Synthetic geometric projection is diagnostic evidence, not a learned cap detector
or a visibility annotation. Source USD and completed renderers are never edited.
"""
import argparse
import json
from pathlib import Path

from dataset1_fiveclass import read,sha,write,SLOTS


def audit(parent,scenes,out):
    import numpy as np
    from pxr import Gf,Usd,UsdGeom
    if out.exists():raise ValueError('Preserve existing audit')
    assets=parent/'input/datasets/csi_digital_twin_v1';file=assets/'stations/tray_inspection/tray_inspection.usdc'
    before=sha(file);stage=Usd.Stage.Open(str(file));time=Usd.TimeCode.Default()
    good=UsdGeom.Xformable(stage.GetPrimAtPath('/TrayInspection/tray_good_top')).ComputeLocalToWorldTransform(time)
    scrap=UsdGeom.Xformable(stage.GetPrimAtPath('/TrayInspection/tray_scrap_top')).ComputeLocalToWorldTransform(time)
    cache=UsdGeom.BBoxCache(time,[UsdGeom.Tokens.default_],useExtentsHint=False);spec=read(assets/'cameras/CapSealCam.json')
    R=np.array(spec['R_world_to_cv']);position=np.array(spec['position_m']);f=spec['focal_px_final'];base={};source_geometry=[]
    for i in range(1,31):
        body=stage.GetPrimAtPath(f'/TrayInspection/G{i:02d}_tube');cap=stage.GetPrimAtPath(f'/TrayInspection/G{i:02d}_cap')
        bw=UsdGeom.Xformable(body).ComputeLocalToWorldTransform(time);cw=UsdGeom.Xformable(cap).ComputeLocalToWorldTransform(time)
        mesh=next(p for p in Usd.PrimRange(cap) if p.IsA(UsdGeom.Mesh));mw=UsdGeom.Xformable(mesh).ComputeLocalToWorldTransform(time)
        points=np.asarray(UsdGeom.Mesh(mesh).GetPointsAttr().Get(),dtype=float);world=np.asarray(mw);vertices=points@world[:3,:3]+world[3,:3]
        bounds=cache.ComputeWorldBound(cap).ComputeAlignedRange();center=(np.array(bounds.GetMin())+np.array(bounds.GetMax()))/2
        for prefix in ('G','S'):
            slot=f'{prefix}{i:02d}';mapping=Gf.Matrix4d(1) if prefix=='G' else good.GetInverse()*scrap;matrix=np.asarray(mapping)
            v=vertices@matrix[:3,:3]+matrix[3,:3];pivot=np.array((cw*mapping).ExtractTranslation())
            aperture=np.array((bw*mapping).ExtractTranslation());aperture[2]=float((good if prefix=='G' else scrap).ExtractTranslation()[2])
            mapped_center=np.array(mapping.Transform(Gf.Vec3d(*center)))
            base[slot]=dict(vertices=v,pivot=pivot,center=mapped_center,aperture=aperture)
            source_geometry.append(dict(slot=slot,cap_pivot_world=pivot.tolist(),cap_bounds_center_world=mapped_center.tolist(),
                cap_pivot_to_center_m=float(np.linalg.norm(pivot-mapped_center))))

    def around(point,axis,degrees):
        a=Gf.Matrix4d(1);a.SetTranslate(Gf.Vec3d(*(-point)));b=Gf.Matrix4d(1);b.SetRotate(Gf.Rotation(Gf.Vec3d(*axis),degrees));c=Gf.Matrix4d(1);c.SetTranslate(Gf.Vec3d(*point));return a*b*c
    def transform(vertices,matrix):
        m=np.asarray(matrix);return vertices@m[:3,:3]+m[3,:3]
    rows=[]
    for row in map(json.loads,(scenes/'rendered-v1/manifest.jsonl').read_text().splitlines()):
        family=row['nuisance'];delta=R[0]*family['camera_dx_px']*1.1/f+R[1]*family['camera_dy_px']*1.1/f;camera=position+delta
        def project(vertices):
            xyz=(vertices-camera)@R.T;xy=xyz[:,:2]/xyz[:,2:];rad=(xy*xy).sum(1);return xy*(1+spec['lens_k1']*rad[:,None])*f+np.array([320,240])
        for normal in (True,False):
            targets=[]
            for slot in SLOTS:
                b=base[slot];params=family['parameters'][slot];label='pass' if normal else row['labels'][slot];vertices=b['vertices'].copy()
                if label=='crooked_cap':
                    a=np.deg2rad(params['cap_tilt_azimuth_deg']);vertices=transform(vertices,around(b['pivot'],np.array([np.cos(a),np.sin(a),0.]),params['cap_tilt_deg']))
                    vertices[:,2]+=params['cap_lift_m']
                a=np.deg2rad(params['body_azimuth_deg']);vertices=transform(vertices,around(b['aperture'],np.array([np.cos(a),np.sin(a),0.]),params['body_lean_deg']))
                pixels=project(vertices);center=pixels.mean(0);aperture=np.asarray(row['targets_xy'][slot]);box=np.r_[pixels.min(0),pixels.max(0)]
                targets.append(dict(slot=slot,synthetic_geometry_label=label,cap_present=label not in ('missing_cap','no_vial'),
                    aperture_xy=aperture.tolist(),projected_cap_mean_xy=center.tolist(),projected_cap_box_xyxy=box.tolist(),
                    aperture_to_cap_pixels=float(np.linalg.norm(center-aperture)),cap_pivot_to_center_m=float(np.linalg.norm(b['pivot']-b['center'])),
                    geometric_projection_only=True,visibility_evaluated=False))
            rows.append(dict(scene=row['id']+('-normal' if normal else ''),source_sha256=row['golden_sha256'] if normal else row['image_sha256'],targets=targets))
    if sha(file)!=before:raise ValueError('Source stage changed')
    write(out,dict(schema='cap-display-geometry-audit/v1',source_stage_sha256=before,source_unchanged=True,source_geometry=source_geometry,
        scenes=rows,helper_sha256=sha(__file__),training=False,new_inference=False,display_geometry_is_not_model_prediction=True))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('parent','scenes','out'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();audit(a.parent,a.scenes,a.out)
