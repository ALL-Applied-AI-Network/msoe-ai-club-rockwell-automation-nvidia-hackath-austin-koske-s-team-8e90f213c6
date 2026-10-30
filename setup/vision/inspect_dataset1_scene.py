"""CPU-only inventory of the supplied Dataset 1 station; never edits its layer."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def main():
    from pxr import Usd, UsdGeom, UsdShade
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError('Preserve prior inventories; use a fresh path')
    before = hashlib.sha256(args.stage.read_bytes()).hexdigest()
    stage = Usd.Stage.Open(str(args.stage))
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_], useExtentsHint=False)
    meshes, cameras, transforms = [], [], []
    for prim in stage.Traverse():
        if prim.GetTypeName() == 'Mesh':
            bounds = cache.ComputeWorldBound(prim).ComputeAlignedRange()
            material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
            meshes.append(dict(path=str(prim.GetPath()), minimum=list(bounds.GetMin()), maximum=list(bounds.GetMax()),
                               material=str(material.GetPath()) if material else None,
                               visibility=UsdGeom.Imageable(prim).ComputeVisibility()))
        elif prim.GetTypeName() == 'Camera':
            cameras.append(dict(path=str(prim.GetPath()), matrix=[list(r) for r in UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())]))
        elif prim.GetTypeName() == 'Xform':
            transforms.append(str(prim.GetPath()))
    record = dict(stage_sha256=before, meters_per_unit=UsdGeom.GetStageMetersPerUnit(stage),
                  type_counts=dict(Counter(p.GetTypeName() for p in stage.Traverse())),
                  meshes=meshes, cameras=cameras, transforms=transforms)
    assert hashlib.sha256(args.stage.read_bytes()).hexdigest() == before
    args.out.write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(dict(meshes=len(meshes), cameras=cameras, candidate_paths=[r['path'] for r in meshes if any(k in r['path'].lower() for k in ('vial', 'cap', 'tray'))][:100])), flush=True)


if __name__ == '__main__':
    main()
