"""Read authored vial principal axes on CPU; do not change the supplied stage."""
import argparse
import math
from pathlib import Path

from dataset1_fiveclass import sha,write


def audit(stage_file,out):
    import numpy as np
    from pxr import Usd,UsdGeom
    if out.exists():raise ValueError('Existing audit; preserve it')
    before=sha(stage_file);stage=Usd.Stage.Open(str(stage_file));rows=[]
    for prim in stage.GetPrimAtPath('/TrayInspection').GetChildren():
        name=prim.GetName()
        if not name.startswith(('G','S')) or not name.endswith('_tube'):continue
        meshes=[p for p in Usd.PrimRange(prim) if p.IsA(UsdGeom.Mesh)]
        points=[]
        for mesh in meshes:
            xyz=np.asarray(UsdGeom.Mesh(mesh).GetPointsAttr().Get(),dtype=float)
            world=np.asarray(UsdGeom.Xformable(mesh).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
            points.extend((xyz@world[:3,:3]+world[3,:3]).tolist())
        array=np.asarray(points);cov=np.cov(array.T);eigenvectors=np.linalg.eigh(cov)[1];axis=eigenvectors[:,-1]
        rows.append(dict(prim=str(prim.GetPath()),fixture=name[0],principal_axis_world=axis.tolist(),
            lean_from_vertical_deg=math.degrees(math.acos(min(1,abs(float(axis[2]))))),points=len(array),
            defect_class=None,semantic_label_approved=False))
    if sha(stage_file)!=before:raise ValueError('Original stage changed')
    write(out,dict(source_stage_sha256=before,source_unchanged=True,cpu_only=True,poses=rows,
        rendered_training_body_lean_range_deg=[0,30],interpretation='Geometry coverage audit, not semantic defect adjudication or physical pose truth.',helper_sha256=sha(__file__)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();audit(a.stage,a.out)
