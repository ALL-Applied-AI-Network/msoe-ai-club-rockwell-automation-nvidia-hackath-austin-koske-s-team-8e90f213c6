"""CPU-only export of proposed two-fixture aperture geometry; no layer changes."""
import argparse
from pathlib import Path
from dataset1_fiveclass import sha,write


if __name__=='__main__':
    from pxr import Gf,Usd,UsdGeom
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists():raise ValueError('Preserve existing geometry')
    before=sha(a.stage);stage=Usd.Stage.Open(str(a.stage));time=Usd.TimeCode.Default()
    good=UsdGeom.Xformable(stage.GetPrimAtPath('/TrayInspection/tray_good_top')).ComputeLocalToWorldTransform(time)
    scrap=UsdGeom.Xformable(stage.GetPrimAtPath('/TrayInspection/tray_scrap_top')).ComputeLocalToWorldTransform(time)
    points={}
    for i in range(1,31):
        body=UsdGeom.Xformable(stage.GetPrimAtPath(f'/TrayInspection/G{i:02d}_tube')).ComputeLocalToWorldTransform(time)
        for fixture,transform in [('G',Gf.Matrix4d(1)),('S',good.GetInverse()*scrap)]:
            point=list((body*transform).ExtractTranslation());point[2]=(good if fixture=='G' else scrap).ExtractTranslation()[2]
            points[f'{fixture}{i:02d}']=point
    assert sha(a.stage)==before
    write(a.out,dict(schema='proposed-two-fixture-geometry/v1',stage_sha256=before,apertures_world_m=points,
        real_mapping_confirmed=False,source_layer_unchanged=True,helper_sha256=sha(__file__)))
