"""
NVIDIA Isaac Sim Replicator Synthetic Data Generator
=====================================================
Headless standalone script for running inside NVIDIA Isaac Sim on GPU instances (Rosie HPC / Brev).
"""

import os
import sys

# Ensure the synthetic_pipeline directory is on Python search path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import argparse
import traceback
import yaml
import numpy as np
from PIL import Image

simulation_app = None
omni = None
rep = None
Usd = UsdGeom = UsdShade = Sdf = Gf = None
import_error_detail = ""
ISAAC_SIM_AVAILABLE = False

def init_simulation_app():
    """Initializes SimulationApp safely at runtime rather than on module import."""
    global simulation_app, omni, rep, Usd, UsdGeom, UsdShade, Sdf, Gf, ISAAC_SIM_AVAILABLE, import_error_detail
    if ISAAC_SIM_AVAILABLE and simulation_app is not None:
        return True

    try:
        try:
            from isaacsim.simulation_app import SimulationApp
        except ImportError:
            try:
                from isaacsim import SimulationApp
            except ImportError:
                from omni.isaac.kit import SimulationApp

        simulation_app = SimulationApp({
            "headless": True,
            "extra_args": [
                "--/rtx/verifyDriverVersion/enabled=false",
                "--/renderer/multiGpu/enabled=false",
                "--/renderer/activeGpu=0"
            ]
        })
        import omni as _omni
        import omni.replicator.core as _rep
        from pxr import Usd as _Usd, UsdGeom as _UsdGeom, UsdShade as _UsdShade, Sdf as _Sdf, Gf as _Gf
        
        omni = _omni
        rep = _rep
        Usd = _Usd
        UsdGeom = _UsdGeom
        UsdShade = _UsdShade
        Sdf = _Sdf
        Gf = _Gf
        ISAAC_SIM_AVAILABLE = True
        return True
    except Exception as e:
        import_error_detail = traceback.format_exc()
        ISAAC_SIM_AVAILABLE = False
        return False

from physics_fluid_shader import PhysicsFluidShader
from dataset_schema import DatasetAnnotator

def parse_args():
    parser = argparse.ArgumentParser(description="Isaac Sim Replicator Synthetic Data Generator")
    parser.add_argument("--usd-stage", type=str, default="", help="Path to CSI digital twin USD file")
    parser.add_argument("--config", type=str, default="synthetic_pipeline/config.yaml", help="Path to config.yaml")
    parser.add_argument("--output-dir", type=str, default="synthetic_data_output", help="Output directory")
    parser.add_argument("--num-repeats", type=int, default=3, help="Number of repetitions per transition")
    parser.add_argument("--headless", action="store_true", default=True, help="Run in headless mode")
    # Absorb any kit-level flags like --/rtx/...
    args, unknown = parser.parse_known_args()
    return args

def create_pbr_material(stage, mat_path: str, diffuse_color: tuple, opacity: float = 1.0, roughness: float = 0.5, metallic: float = 0.0, ior: float = 1.5):
    """Creates a UsdPreviewSurface material on the stage and returns the UsdShade.Material."""
    mat = UsdShade.Material.Define(stage, mat_path)
    shader = UsdShade.Shader.Define(stage, mat_path + "/PbrShader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*diffuse_color))
    shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(float(opacity))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(roughness))
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(float(metallic))
    shader.CreateInput("ior", Sdf.ValueTypeNames.Float).Set(float(ior))
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return mat

def bind_material(prim, mat):
    """Safely binds a USD Material to a geometry prim."""
    UsdShade.MaterialBindingAPI.Apply(prim.GetPrim()).Bind(mat)

def setup_isaac_sim_scene(usd_path: str, camera_res: tuple = (640, 480)):
    if not ISAAC_SIM_AVAILABLE:
        raise RuntimeError("Isaac Sim / Omniverse Replicator is not available in current environment.")

    if usd_path and os.path.exists(usd_path):
        omni.usd.get_context().open_stage(usd_path)
    else:
        omni.usd.get_context().new_stage()
        stage = omni.usd.get_context().get_stage()

        # 1. White Backing Sheet (Inspection Backdrop)
        backdrop = UsdGeom.Cube.Define(stage, "/World/CSI_Line/Backdrop")
        backdrop.AddTranslateOp().Set(Gf.Vec3f(0.04, 0.08, 0.12))
        backdrop.AddScaleOp().Set(Gf.Vec3f(0.35, 0.005, 0.18))
        mat_backdrop = create_pbr_material(stage, "/World/CSI_Line/Backdrop/Material",
                                           diffuse_color=(0.88, 0.88, 0.86), roughness=0.90)
        bind_material(backdrop, mat_backdrop)

        # 2. MagneMotion Stainless Steel Track Rail
        rail = UsdGeom.Cube.Define(stage, "/World/CSI_Line/TrackRail")
        rail.AddTranslateOp().Set(Gf.Vec3f(0.04, 0.0, -0.015))
        rail.AddScaleOp().Set(Gf.Vec3f(0.40, 0.06, 0.015))
        mat_rail = create_pbr_material(stage, "/World/CSI_Line/TrackRail/Material",
                                       diffuse_color=(0.70, 0.70, 0.72), metallic=0.9, roughness=0.25)
        bind_material(rail, mat_rail)

        # 3. Aluminum Carrier Puck (Base)
        puck = UsdGeom.Cube.Define(stage, "/World/CSI_Line/CarrierPuck")
        puck.AddTranslateOp().Set(Gf.Vec3f(0.0, 0.0, 0.015))
        puck.AddScaleOp().Set(Gf.Vec3f(0.038, 0.038, 0.015))
        mat_puck = create_pbr_material(stage, "/World/CSI_Line/CarrierPuck/Material",
                                       diffuse_color=(0.78, 0.78, 0.80), metallic=0.8, roughness=0.3)
        bind_material(puck, mat_puck)

        # 4. Black Anodized Bracket Clamp
        bracket = UsdGeom.Cube.Define(stage, "/World/CSI_Line/PuckBracket")
        bracket.AddTranslateOp().Set(Gf.Vec3f(0.0, 0.0, 0.085))
        bracket.AddScaleOp().Set(Gf.Vec3f(0.032, 0.032, 0.010))
        mat_bracket = create_pbr_material(stage, "/World/CSI_Line/PuckBracket/Material",
                                          diffuse_color=(0.08, 0.08, 0.08), roughness=0.45)
        bind_material(bracket, mat_bracket)

        # 5. Clear Acrylic/Glass Cylindrical Vial (30mm OD, 140mm tall)
        glass = UsdGeom.Cylinder.Define(stage, "/World/CSI_Line/VialStation/GlassVial")
        glass.AddTranslateOp().Set(Gf.Vec3f(0.0, 0.0, 0.10))
        glass.AddScaleOp().Set(Gf.Vec3f(0.015, 0.015, 0.07))
        mat_glass = create_pbr_material(stage, "/World/CSI_Line/VialStation/GlassVial/Material",
                                        diffuse_color=(0.95, 0.96, 0.98), opacity=0.18, roughness=0.02, ior=1.49)
        bind_material(glass, mat_glass)

        # 6. Upper Barcode/Identification Label
        label = UsdGeom.Cylinder.Define(stage, "/World/CSI_Line/VialStation/VialLabel")
        label.AddTranslateOp().Set(Gf.Vec3f(0.0, 0.0, 0.13))
        label.AddScaleOp().Set(Gf.Vec3f(0.0152, 0.0152, 0.015))
        mat_label = create_pbr_material(stage, "/World/CSI_Line/VialStation/VialLabel/Material",
                                        diffuse_color=(0.96, 0.96, 0.95), roughness=0.7)
        bind_material(label, mat_label)

        # 7. Liquid Fluid Cylinder (inside the vial)
        fluid = UsdGeom.Cylinder.Define(stage, "/World/CSI_Line/VialStation/LiquidCylinder")
        fluid.AddTranslateOp().Set(Gf.Vec3f(0.0, 0.0, 0.065))
        fluid.AddScaleOp().Set(Gf.Vec3f(0.0135, 0.0135, 0.035))
        mat_fluid = create_pbr_material(stage, "/World/CSI_Line/VialStation/LiquidCylinder/Material",
                                        diffuse_color=(0.1, 0.5, 0.8), opacity=0.85, roughness=0.01, ior=1.333)
        bind_material(fluid, mat_fluid)

        # 8. Stainless Steel Dispensing Nozzle (above vial opening)
        nozzle = UsdGeom.Cylinder.Define(stage, "/World/CSI_Line/VialStation/Nozzle")
        nozzle.AddTranslateOp().Set(Gf.Vec3f(0.0, 0.0, 0.20))
        nozzle.AddScaleOp().Set(Gf.Vec3f(0.005, 0.005, 0.03))
        mat_nozzle = create_pbr_material(stage, "/World/CSI_Line/VialStation/Nozzle/Material",
                                         diffuse_color=(0.75, 0.75, 0.78), metallic=0.95, roughness=0.15)
        bind_material(nozzle, mat_nozzle)

        # 9. Adjacent Queued Vial (Carrier Puck on Right side, exactly like real photo)
        adj_puck = UsdGeom.Cube.Define(stage, "/World/CSI_Line/AdjCarrierPuck")
        adj_puck.AddTranslateOp().Set(Gf.Vec3f(0.08, 0.0, 0.015))
        adj_puck.AddScaleOp().Set(Gf.Vec3f(0.038, 0.038, 0.015))
        bind_material(adj_puck, mat_puck)

        adj_bracket = UsdGeom.Cube.Define(stage, "/World/CSI_Line/AdjBracket")
        adj_bracket.AddTranslateOp().Set(Gf.Vec3f(0.08, 0.0, 0.085))
        adj_bracket.AddScaleOp().Set(Gf.Vec3f(0.032, 0.032, 0.010))
        bind_material(adj_bracket, mat_bracket)

        adj_glass = UsdGeom.Cylinder.Define(stage, "/World/CSI_Line/AdjGlassVial")
        adj_glass.AddTranslateOp().Set(Gf.Vec3f(0.08, 0.0, 0.10))
        adj_glass.AddScaleOp().Set(Gf.Vec3f(0.015, 0.015, 0.07))
        bind_material(adj_glass, mat_glass)

        adj_label = UsdGeom.Cylinder.Define(stage, "/World/CSI_Line/AdjVialLabel")
        adj_label.AddTranslateOp().Set(Gf.Vec3f(0.08, 0.0, 0.13))
        adj_label.AddScaleOp().Set(Gf.Vec3f(0.0152, 0.0152, 0.015))
        bind_material(adj_label, mat_label)

    stage = omni.usd.get_context().get_stage()

    # 10. Camera positioned directly facing the inspection station
    # 28 cm in front of the line, elevated at 12 cm, looking directly at the vials
    camera_prim = rep.create.camera(
        position=(0.015, -0.28, 0.12),
        look_at=(0.015, 0.0, 0.09),
        focal_length=32.0
    )

    # Overhead diffused factory inspection lighting (calibrated to prevent whiteout)
    dome_light = rep.create.light(
        light_type="dome",
        intensity=600.0,
        temperature=5800.0
    )
    overhead_spot = rep.create.light(
        light_type="rect",
        position=(0.04, -0.12, 0.35),
        scale=(0.25, 0.25, 1.0),
        intensity=1200.0,
        temperature=6200.0
    )

    return stage, camera_prim, dome_light, overhead_spot

def update_usd_fluid_primitive(stage, fluid_prim_path: str, usd_mat_props: dict):
    fluid_prim = stage.GetPrimAtPath(fluid_prim_path)
    if fluid_prim.IsValid():
        xform = UsdGeom.Xformable(fluid_prim)
        scale_op = None
        translate_op = None
        for op in xform.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeScale:
                scale_op = op
            elif op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                translate_op = op
        if not scale_op:
            scale_op = xform.AddScaleOp()
        if not translate_op:
            translate_op = xform.AddTranslateOp()

        fill_scale = max(0.05, float(usd_mat_props.get("mesh_scale_z", 1.0)))
        fluid_height = 0.11 * fill_scale
        half_height = fluid_height / 2.0
        scale_op.Set(Gf.Vec3f(0.0135, 0.0135, half_height))
        translate_op.Set(Gf.Vec3f(0.0, 0.0, 0.032 + half_height))

    mat_path = fluid_prim_path + "/Material"
    shader_prim = stage.GetPrimAtPath(mat_path + "/PbrShader")
    if not shader_prim.IsValid():
        shader_prim = stage.GetPrimAtPath(mat_path)
    if shader_prim.IsValid():
        shader = UsdShade.Shader(shader_prim)
        diff_color = Gf.Vec3f(*usd_mat_props["diffuse_color"])
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(diff_color)
        shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(float(usd_mat_props["opacity"]))
        shader.CreateInput("ior", Sdf.ValueTypeNames.Float).Set(float(usd_mat_props.get("ior", 1.333)))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(usd_mat_props.get("roughness", 0.01)))

def run_replicator_generation():
    args = parse_args()

    if not init_simulation_app():
        print("[-] Error: Omniverse Isaac Sim Python modules failed to load or initialize.")
        if import_error_detail:
            print("[-] Exception Traceback:")
            print(import_error_detail)
        sys.exit(1)

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    physics = PhysicsFluidShader(
        base_holdup_volume_ml=config["physics_parameters"]["base_holdup_volume_ml"],
        holdup_std_ml=config["physics_parameters"]["holdup_std_ml"],
        defect_threshold_delta_e=config["physics_parameters"]["defect_threshold_delta_e"]
    )
    annotator = DatasetAnnotator(args.output_dir)

    print(f"[+] Initializing Isaac Sim Stage: {args.usd_stage}")
    stage, camera, dome_light, overhead_spot = setup_isaac_sim_scene(
        args.usd_stage, tuple(config["output"]["resolution"])
    )

    render_product = rep.create.render_product(camera, config["output"]["resolution"])
    rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
    rgb_annotator.attach([render_product])
    bbox_annotator = rep.AnnotatorRegistry.get_annotator("bounding_box_2d_tight")
    bbox_annotator.attach([render_product])

    raw_writer_dir = os.path.join(annotator.output_dir, "replicator_raw")
    basic_writer = None
    try:
        basic_writer = rep.WriterRegistry.get("BasicWriter")
        basic_writer.initialize(output_dir=raw_writer_dir, rgb=True)
        basic_writer.attach([render_product])
        print(f"[+] Replicator BasicWriter attached to: {raw_writer_dir}", flush=True)
    except Exception as e:
        print(f"[!] BasicWriter notice: {e}", flush=True)

    fluid_prim_path = "/World/CSI_Line/VialStation/LiquidCylinder"

    total_samples = 0
    transitions = config["transitions"]
    recipes = config["recipes"]

    print(f"[+] Generating synthetic data across {len(transitions)} transitions x {args.num_repeats} repeats...", flush=True)

    # RTX Warm-up steps to ensure shaders compile and framebuffers initialize in headless mode
    print("[+] Warming up Isaac Sim RTX renderer...", flush=True)
    for _ in range(10):
        rep.orchestrator.step()
        if simulation_app:
            simulation_app.update()

    for rep_idx in range(args.num_repeats):
        for trans_idx, (prior_id, next_id) in enumerate(transitions):
            prior_rec = recipes[prior_id]
            prior_rec["recipe_id"] = prior_id
            next_rec = recipes[next_id]
            next_rec["recipe_id"] = next_id

            for batch_pos in [1, 2, 3]:
                sample_id = f"sim_r{prior_id}_to_r{next_id}_v{batch_pos}_rep{rep_idx:02d}"
                
                sim_data = physics.simulate_recipe_transition(
                    prior_rec, next_rec, batch_position=batch_pos, randomize_holdup=True
                )

                update_usd_fluid_primitive(stage, fluid_prim_path, sim_data["usd_material"])

                # Statically configured lighting and camera to prevent Replicator OmniGraph bloat
                light_intensity = 1.0
                dx = 0.0
                dy = 0.0

                rep.orchestrator.step(rt_subframes=4)
                if simulation_app:
                    simulation_app.update()

                rgb_data = rgb_annotator.get_data()
                if rgb_data is None or getattr(rgb_data, "size", 0) == 0:
                    if simulation_app:
                        simulation_app.update()
                    rgb_data = rgb_annotator.get_data()

                img_filename = f"{sample_id}.jpg"
                img_path = os.path.join(annotator.images_dir, img_filename)
                
                if rgb_data is not None and getattr(rgb_data, "size", 0) > 0:
                    if len(rgb_data.shape) == 3 and rgb_data.shape[-1] == 4:
                        rgb_array = rgb_data[:, :, :3]
                    else:
                        rgb_array = rgb_data
                    im = Image.fromarray(rgb_array.astype(np.uint8))
                    im.save(img_path, quality=95)
                    print(f"[+] Rendered {sample_id} ({rgb_array.shape})", flush=True)
                else:
                    saved_from_raw = False
                    if os.path.exists(raw_writer_dir):
                        raw_files = sorted([
                            os.path.join(root, f)
                            for root, _, files in os.walk(raw_writer_dir)
                            for f in files if f.endswith(('.png', '.jpg', '.jpeg'))
                        ])
                        if raw_files:
                            latest_raw = raw_files[-1]
                            try:
                                with Image.open(latest_raw) as raw_im:
                                    raw_im.convert("RGB").save(img_path, quality=95)
                                saved_from_raw = True
                                print(f"[+] Captured {sample_id} via BasicWriter fallback ({latest_raw})", flush=True)
                            except Exception:
                                pass
                    if not saved_from_raw:
                        print(f"[-] Warning: sample {sample_id} render buffer empty", flush=True)

                fill_frac = float(sim_data["fill_fraction"])
                bbox_coords = {
                    "vial": [0.470, 0.580, 0.110, 0.500],
                    "meniscus": [0.470, 0.740 - (fill_frac * 0.280), 0.090, 0.035],
                    "nozzle": [0.470, 0.140, 0.035, 0.140]
                }

                annotator.export_sample(
                    sample_id=sample_id,
                    image_filename=img_filename,
                    sim_data=sim_data,
                    camera_params={"dx": dx, "dy": dy, "res": config["output"]["resolution"]},
                    lighting_params={"intensity": light_intensity},
                    bbox_coords=bbox_coords
                )
                total_samples += 1

    annotator.finalize()
    print(f"[+] Replicator generation complete! Total synthetic samples: {total_samples}")
    print(f"[+] Output directory: {os.path.abspath(args.output_dir)}")
    
    if simulation_app:
        simulation_app.close()

if __name__ == "__main__":
    run_replicator_generation()
