"""Isaac 6.0 adapter for the existing twin; source assets stay untouched."""
from pathlib import Path
import faulthandler
import sys

faulthandler.enable()
faulthandler.dump_traceback_later(120, repeat=True)

project = Path(__file__).resolve().parents[2] / 'datasets' / 'mixing_twin'
sys.path.insert(0, str(project))


def apply_sample(stage, sample, timecode=None):
    from mixing import usd_bridge as bridge
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade
    proxy_path = '/SetupLiquidProxy'
    if not stage.GetPrimAtPath(proxy_path):
        original = stage.GetPrimAtPath(original_liquid)
        mesh = UsdGeom.Mesh.Define(stage, proxy_path)
        UsdGeom.Xformable(mesh).AddTransformOp().Set(
            UsdGeom.Xformable(original).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
        UsdGeom.Imageable(original).GetVisibilityAttr().Set('invisible')
        material = UsdShade.Material.Define(stage, '/FillStation/SetupLiquidMaterial')
        shader = UsdShade.Shader.Define(stage, '/FillStation/SetupLiquidMaterial/Surface')
        shader.CreateIdAttr('UsdPreviewSurface')
        shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(1, 0, 0))
        shader.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(0.3)
        shader.CreateInput('opacity', Sdf.ValueTypeNames.Float).Set(0.6)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), 'surface')
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
        bridge.LIQUID = proxy_path
        bridge.SHADERS = [str(shader.GetPath())]
    result = original_apply(stage, sample, timecode)
    result['setup_adapter'] = 'unanimated world-space liquid proxy at frozen pose; source CAD unchanged'
    return result


def install_adapter():
    # Called by the injected import below, strictly after Kit has initialized.
    from mixing import usd_bridge as bridge
    global original_apply, original_liquid
    original_apply, original_liquid = bridge.apply_sample, bridge.LIQUID
    bridge.apply_sample = apply_sample


source_path = project / 'scripts' / 'render_isaac.py'
source = source_path.read_text(encoding='utf-8')
anchor = '        from mixing.usd_bridge import apply_sample, check_stage, CAMERA'
if source.count(anchor) != 1:
    raise SystemExit('The upstream renderer import changed; review the setup adapter before rendering.')
source = source.replace(anchor, '        install_setup_adapter()\n' + anchor)
# Compile an in-memory wrapper; never write to the project renderer or assets.
exec(compile(source, str(source_path), 'exec'),
     dict(__name__='__main__', __file__=str(source_path), install_setup_adapter=install_adapter))
