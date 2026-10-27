"""Render a self-contained scene before attempting the much larger CSI assets."""
import argparse
import json
import os
from pathlib import Path
import time

parser = argparse.ArgumentParser()
parser.add_argument('--out', type=Path, required=True)
args = parser.parse_args()
if os.environ.get('OMNI_KIT_ACCEPT_EULA', '').upper() not in {'YES', 'Y', '1'}:
    raise SystemExit('Accept the NVIDIA Omniverse license, then set OMNI_KIT_ACCEPT_EULA=YES.')
args.out.mkdir(parents=True, exist_ok=True)
started = time.monotonic()
from isaacsim import SimulationApp
app = SimulationApp({'headless': True, 'width': 320, 'height': 240,
                     'renderer': 'RayTracedLighting'})
try:
    import numpy as np
    import omni.replicator.core as rep
    import omni.usd
    from PIL import Image
    from pxr import Gf, UsdGeom, UsdLux
    stage = omni.usd.get_context().get_stage()
    cube = UsdGeom.Cube.Define(stage, '/World/SetupCube')
    cube.CreateSizeAttr(1.0)
    cube.CreateDisplayColorAttr([Gf.Vec3f(0.1, 0.5, 0.9)])
    ground = UsdGeom.Cube.Define(stage, '/World/SetupGround')
    ground.CreateSizeAttr(1.0)
    ground.AddScaleOp().Set(Gf.Vec3f(8, 8, 0.1))
    ground.AddTranslateOp().Set(Gf.Vec3f(0, 0, -6))
    light = UsdLux.DomeLight.Define(stage, '/World/SetupLight')
    light.CreateIntensityAttr(1500)
    camera = rep.create.camera(position=(3, 3, 2), look_at=(0, 0, 0))
    product = rep.create.render_product(camera, (320, 240))
    annotator = rep.AnnotatorRegistry.get_annotator('rgb')
    annotator.attach(product)
    for _ in range(12):
        rep.orchestrator.step(rt_subframes=4)
    image = np.asarray(annotator.get_data())
    if image.size == 0 or image.std() < 3:
        raise RuntimeError('Renderer did not return a useful nonuniform image.')
    path = args.out / 'isaac-smoke.png'
    Image.fromarray(image[..., :3].astype(np.uint8)).save(path)
    report = {'ok': True, 'image': path.name, 'shape': list(image.shape),
              'pixel_std': float(image.std()), 'seconds': time.monotonic() - started,
              'scope': 'self-contained RTX render; not CSI optical calibration'}
    (args.out / 'result.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))
finally:
    app.close()
