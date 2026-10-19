"""Small camera-preserving RGB + geometric-label-pass review batch on GB10.

No source USD saves, real evaluation labels, learned masks, or training calls.
Transparent shells require separate vial/content annotation passes. These are
geometric visibility masks, NOT calibrated refracted RGB segmentation truth.
"""
import argparse
import faulthandler
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

from box_annotations import COLORS, instance_mask, semantic_mask, lens_mm, mask_box, sha256, validate_fixture, write_json, yolo_line


ROOT = Path(__file__).resolve().parents[2]
PREFIX = "/FillStation/FillStation_LocalRoot/"
CAMERA = PREFIX + "FillCam/FillCam"
LIQUID = PREFIX + "Fill_liquid/Fill_liquid"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, default=ROOT / "datasets/mixing_twin/assets/csi_digital_twin_v1")
    parser.add_argument("--fixture", type=Path, default=Path(__file__).with_name("box_fixture.json"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--subframes", type=int, default=16)
    parser.add_argument("--pt-spp", type=int, help="Explicit path-tracing samples per pixel; default preserves runtime settings")
    parser.add_argument("--compact", action="store_true", help="Keep native ID planes/maps but omit annotation-pass RGB PNGs")
    parser.add_argument("--label-annotator", "--instance-annotator", dest="label_annotator",
                        choices=["semantic_segmentation", "instance_segmentation", "instance_segmentation_fast"],
                        default="semantic_segmentation")
    args = parser.parse_args()
    fixture = json.loads(args.fixture.read_text(encoding="utf-8"))
    validate_fixture(fixture)
    if args.limit < 1 or args.subframes < 1:
        parser.error("positive limit and subframes required")
    if args.pt_spp is not None and not 1 <= args.pt_spp <= 32:
        parser.error("pt-spp must be 1..32")
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        parser.error("Use a fresh output directory")
    out.mkdir(parents=True, exist_ok=True)
    assets = args.asset_root.resolve()
    source = assets / "stations/fill_station/fill_station.usda"
    camera_file = assets / "cameras/FillCam.json"
    camera_spec = json.loads(camera_file.read_text(encoding="utf-8"))
    before = {str(p.relative_to(assets)): sha256(p) for p in sorted(assets.rglob("*")) if p.is_file()}
    faulthandler.enable()
    faulthandler.dump_traceback_later(180, repeat=True)
    start = time.monotonic()
    from isaacsim import SimulationApp
    app = SimulationApp({"headless": True, "renderer": "PathTracing", "width": 640, "height": 480})
    failed = False
    try:
        import numpy as np
        from PIL import Image, ImageDraw
        from isaacsim.core.utils.extensions import enable_extension
        enable_extension("omni.replicator.core")
        from isaacsim.core.experimental.utils.semantics import add_labels
        import omni.replicator.core as rep
        import omni.usd
        import omni.timeline
        import carb.settings
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, UsdLux
        try:
            isaac_version = importlib.metadata.version("isaacsim")
        except importlib.metadata.PackageNotFoundError:
            # The optional local source build is not a pip distribution.
            from isaacsim.core.version import get_version
            isaac_version = ".".join(get_version()[:2])
        carb.settings.get_settings().set("/omni/replicator/asyncRendering", False)
        if args.pt_spp is not None:
            carb.settings.get_settings().set("/rtx/pathtracing/spp", args.pt_spp)
            carb.settings.get_settings().set("/rtx/pathtracing/totalSpp", args.pt_spp)
        ctx = omni.usd.get_context()
        if not ctx.open_stage(str(source)):
            raise RuntimeError("Could not open supplied fill station")
        loaded = time.monotonic()
        while ctx.get_stage_loading_status()[2]:
            if time.monotonic() - loaded > 120:
                raise TimeoutError("Scene loading timed out")
            app.update()
        for _ in range(10):
            app.update()
        stage = ctx.get_stage()
        stage.SetEditTarget(stage.GetSessionLayer())
        frozen = [(a, a.Get(fixture["station_frame"])) for p in stage.Traverse()
                  for a in p.GetAttributes() if a.GetNumTimeSamples()]
        for attr, value in frozen:
            if value is not None:
                attr.Set(value)
        timeline = omni.timeline.get_timeline_interface()
        timeline.pause()
        timeline.set_current_time(0)
        app.update()
        light_attributes = [(p.GetAttribute("inputs:intensity"), float(p.GetAttribute("inputs:intensity").Get()))
            for p in stage.Traverse() if p.HasAPI(UsdLux.LightAPI) and p.GetAttribute("inputs:intensity").Get() is not None]
        cam = UsdGeom.Camera(stage.GetPrimAtPath(CAMERA))
        camera_world = np.asarray(UsdGeom.Xformable(cam).ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T
        np.testing.assert_allclose(camera_world, camera_spec["camera_to_world_in_station_file"], atol=1e-6)
        optics_mm = lens_mm([cam.GetFocalLengthAttr().Get(), cam.GetHorizontalApertureAttr().Get(),
                             cam.GetVerticalApertureAttr().Get()], UsdGeom.GetStageMetersPerUnit(stage))
        np.testing.assert_allclose(optics_mm,
                                  [camera_spec["focal_length_mm"], camera_spec["horizontal_aperture_mm"],
                                   camera_spec["vertical_aperture_mm"]], atol=1e-5)
        assert camera_spec["final_image_px"] == [640, 480]
        camera_inverse = np.linalg.inv(camera_world)
        K = np.asarray(camera_spec["intrinsics_K_px"])

        def world_bounds(path):
            cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
            bound = cache.ComputeWorldBound(stage.GetPrimAtPath(path)).ComputeAlignedRange()
            return np.asarray(bound.GetMin()), np.asarray(bound.GetMax())

        low, high = world_bounds(LIQUID)
        radius = float((high[0] - low[0]) / 2)
        base = float(low[2])
        max_height = float(high[2] - low[2])
        center = (low + high) / 2
        nozzle_low, nozzle_high = world_bounds(PREFIX + "Fill_nozzle_stainless/Fill_nozzle_stainless")
        nozzle_xy = (nozzle_low[:2] + nozzle_high[:2]) / 2
        nozzle_z = float(nozzle_low[2])
        # Original liquid/jet animation is never used as a label or modified on disk.
        UsdGeom.Imageable(stage.GetPrimAtPath(PREFIX + "Fill_liquid")).GetVisibilityAttr().Set("invisible")
        UsdGeom.Imageable(stage.GetPrimAtPath(PREFIX + "Fill_dispensing_stream")).GetVisibilityAttr().Set("invisible")
        UsdGeom.Xform.Define(stage, "/SyntheticReview")

        def material(name, color, opacity=1):
            path = "/SyntheticReview/Materials/" + name
            mat = UsdShade.Material.Define(stage, path)
            shader = UsdShade.Shader.Define(stage, path + "/Surface")
            shader.CreateIdAttr("UsdPreviewSurface")
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
            shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(opacity)
            shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.12)
            shader.CreateInput("ior", Sdf.ValueTypeNames.Float).Set(1.333)
            mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
            return mat, shader

        beauty_mat, beauty_shader = material("LiquidAppearance", (0.01, 0.2, 0.3), 0.6)
        label_mat, _ = material("OpaqueLabelPass", (0.8, 0.8, 0.8), 1)
        UsdGeom.Xform.Define(stage, "/SyntheticReview/Liquid")
        liquid = UsdGeom.Cylinder.Define(stage, "/SyntheticReview/Liquid/Geometry")
        liquid.GetAxisAttr().Set("Z")
        liquid.GetRadiusAttr().Set(radius)
        liquid_translate = liquid.AddTranslateOp()
        UsdGeom.Xform.Define(stage, "/SyntheticReview/Stream")
        stream = UsdGeom.Cylinder.Define(stage, "/SyntheticReview/Stream/Geometry")
        stream.GetAxisAttr().Set("Z")
        stream.GetRadiusAttr().Set(0.0015)
        stream_translate = stream.AddTranslateOp()
        objects = []
        for prim in stage.Traverse():
            if prim.GetTypeName() == "Mesh" and "/Fill_vial_" in str(prim.GetPath()):
                parent = prim.GetParent()
                identity = int(parent.GetAttribute("userProperties:synthetic_instance_id").Get())
                objects.append(dict(class_name="vial", instance_id=identity, prim_path=str(prim.GetPath())))
        objects += [dict(class_name="liquid", instance_id=201, prim_path=str(liquid.GetPath())),
                    dict(class_name="stream", instance_id=301, prim_path=str(stream.GetPath()))]
        for obj in objects:
            obj["semantic_label"] = f"{obj['class_name']}__{obj['instance_id']}"
            obj["label_prim_path"] = str(stage.GetPrimAtPath(obj["prim_path"]).GetParent().GetPath())
            label = obj["semantic_label"] if args.label_annotator == "semantic_segmentation" else obj["class_name"]
            # Keep identity on the persistent object Xform. RTX can recreate a
            # leaf renderable when its visibility/material changes between passes.
            add_labels(stage.GetPrimAtPath(obj["label_prim_path"]), labels=[label], taxonomy="class")
        vial_objects = [o for o in objects if o["class_name"] == "vial"]
        # Labels look through transparent plastic/guard, but keep opaque paper,
        # barcode, holder and nozzle as occluders. RGB appearance stays untouched.
        guard = stage.GetPrimAtPath(PREFIX + "Fill_splash_guard/Cube_001")
        guard_vis = UsdGeom.Imageable(guard).GetVisibilityAttr()
        guard_original = guard_vis.Get()
        vial_bindings = {o["prim_path"]: list(stage.GetPrimAtPath(o["prim_path"]).GetRelationship("material:binding").GetTargets())
                         for o in vial_objects}

        def bind(prim, mat):
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat)

        def visible(prim, yes):
            UsdGeom.Imageable(prim).GetVisibilityAttr().Set("inherited" if yes else "invisible")

        def configure_pass(mode, row):
            guard_vis.Set(guard_original if mode == "rgb" else "invisible")
            for obj in vial_objects:
                prim = stage.GetPrimAtPath(obj["prim_path"])
                visible(prim, mode != "content-labels")
                if mode == "vial-labels":
                    bind(prim, label_mat)
                else:
                    prim.GetRelationship("material:binding").SetTargets(vial_bindings[obj["prim_path"]])
            for proxy, present in ((liquid, row["level"] > 0), (stream, row["stream"] != "off")):
                visible(proxy.GetPrim(), present and mode != "vial-labels")
                bind(proxy.GetPrim(), beauty_mat if mode == "rgb" else label_mat)

        rp = rep.create.render_product(CAMERA, (640, 480))
        rgb_ann = rep.AnnotatorRegistry.get_annotator("rgb")
        instance_ann = rep.AnnotatorRegistry.get_annotator(args.label_annotator)
        rgb_ann.attach([rp])
        instance_ann.attach([rp])
        rep.orchestrator.set_capture_on_play(False)

        def step():
            app.update()
            for _ in range(2):
                # Stage/render-product startup may restore its renderer defaults.
                # Apply an explicit training-quality profile at capture time.
                if args.pt_spp is not None:
                    carb.settings.get_settings().set("/rtx/pathtracing/spp", args.pt_spp)
                    carb.settings.get_settings().set("/rtx/pathtracing/totalSpp", args.pt_spp)
                rep.orchestrator.step(rt_subframes=args.subframes, delta_time=1/30, wait_for_render=True)
                if args.pt_spp is not None and any(carb.settings.get_settings().get(k)!=args.pt_spp
                        for k in ["/rtx/pathtracing/spp","/rtx/pathtracing/totalSpp"]):
                    raise RuntimeError("Runtime changed the explicit path-tracing sample profile")

        def rgb_capture():
            rgb = np.asarray(rgb_ann.get_data())[..., :3].astype(np.uint8).copy()
            if rgb.shape != (480, 640, 3) or rgb.std() < 0.1:
                raise RuntimeError("Invalid RGB capture")
            return rgb

        def projected_box(prim):
            if prim.GetTypeName() == "Mesh":
                points = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(), dtype=float)
            else:
                cyl = UsdGeom.Cylinder(prim)
                angles = np.linspace(0, 2*np.pi, 128, endpoint=False)
                r, h = cyl.GetRadiusAttr().Get(), cyl.GetHeightAttr().Get()/2
                points = np.array([(r*np.cos(t), r*np.sin(t), z) for z in [-h, h] for t in angles])
            world = np.asarray(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T
            camera_points = (camera_inverse @ world @ np.column_stack((points, np.ones(len(points)))).T).T
            if np.any(camera_points[:, 2] >= 0):
                raise ValueError("Geometry crosses/behind camera; projection needs clipping")
            x = K[0, 0] * camera_points[:, 0] / -camera_points[:, 2] + K[0, 2]
            y = K[1, 2] - K[1, 1] * camera_points[:, 1] / -camera_points[:, 2]
            return [float(x.min()), float(y.min()), float(x.max()), float(y.max())]

        records = []
        for folder in ["rgb", "overlays", "masks", "label-passes", "labels"]:
            (out / folder).mkdir()
        write_json(out / "camera.json", dict(supplied=camera_spec, actual_camera_to_world=camera_world.tolist(),
                   actual_focal_horizontal_vertical_aperture_mm=optics_mm.tolist(),
                   source_sha256=sha256(camera_file), resolution=[640, 480],
                   real_calibration_status="supplied camera; no new physical calibration; review against Run 1 only"))
        for row in fixture["cases"][:args.limit]:
            name = row["id"]
            print("CASE", name, flush=True)
            height = max(row["level"] * max_height, 1e-6)
            for attr, original_intensity in light_attributes:
                attr.Set(original_intensity * row.get("light_scale", 1.0))
            stream.GetRadiusAttr().Set(row.get("stream_radius_m", 0.0015))
            beauty_shader.GetInput("roughness").Set(row.get("roughness", 0.12))
            liquid.GetHeightAttr().Set(height)
            liquid_translate.Set(Gf.Vec3d(float(center[0]), float(center[1]), base + height/2))
            bottom = nozzle_z - 0.012 if row["stream"] == "short" else base + row["level"] * max_height
            stream.GetHeightAttr().Set(nozzle_z - bottom)
            stream_translate.Set(Gf.Vec3d(float(nozzle_xy[0]), float(nozzle_xy[1]), (nozzle_z + bottom)/2))
            beauty_shader.GetInput("diffuseColor").Set(Gf.Vec3f(*row["rgb"]))
            beauty_shader.GetInput("opacity").Set(row["opacity"])
            configure_pass("rgb", row)
            step()
            rgb = rgb_capture()
            rgb_path = out / "rgb" / (name + ".png")
            Image.fromarray(rgb).save(rgb_path)
            annotations = []
            annotation_capture = {}
            for mode, targets in [("vial-labels", vial_objects), ("content-labels", objects[-2:])]:
                configure_pass(mode, row)
                rejected_attempts = []
                for attempt in range(1, 5):
                    step()
                    data = instance_ann.get_data()
                    measured = []
                    try:
                        for obj in targets:
                            present = obj["class_name"] == "vial" or (row["level"] > 0 if obj["class_name"] == "liquid" else row["stream"] != "off")
                            mask = (semantic_mask(data, obj["semantic_label"]) if args.label_annotator == "semantic_segmentation"
                                    else instance_mask(data, obj["label_prim_path"]))
                            if not present and mask.any():
                                raise ValueError("Absent object has stale mask")
                            if present and fixture.get("require_visible_present_objects") and not mask.any():
                                raise ValueError(f"Fixture requires a visible mask for {obj['semantic_label']}")
                            if obj["class_name"] == "vial" and "active" in obj["prim_path"] and not mask.any():
                                raise ValueError("Active vial mask missing")
                            measured.append((obj, present, mask))
                    except ValueError as error:
                        # No scene, threshold or identity change: allow metadata to
                        # synchronize after visibility/material changes, bounded.
                        stem = f"{name}-{mode}-rejected-{attempt}"
                        np.save(out / "label-passes" / (stem+"-ids.npy"), np.asarray(data["data"]))
                        write_json(out / "label-passes" / (stem+"-mapping.json"), data["info"]["idToLabels"])
                        rejected_attempts.append(dict(attempt=attempt, reason=str(error), raw_stem="label-passes/"+stem))
                        if attempt == 4:
                            raise RuntimeError(f"Annotation metadata did not become valid: {name}/{mode}") from error
                        print("ANNOTATION SYNC RETRY", name, mode, attempt, str(error), flush=True)
                        continue
                    break
                annotation_capture[mode] = dict(attempts=attempt, rejected_attempts=rejected_attempts)
                np.save(out / "label-passes" / f"{name}-{mode}-ids.npy", np.asarray(data["data"]))
                write_json(out / "label-passes" / f"{name}-{mode}-mapping.json", data["info"]["idToLabels"])
                if not args.compact:
                    Image.fromarray(rgb_capture()).save(out / "label-passes" / f"{name}-{mode}.png")
                for obj, present, mask in measured:
                    filename = f"masks/{name}-{obj['instance_id']}.png"
                    Image.fromarray(mask.astype(np.uint8)*255).save(out / filename)
                    box = mask_box(mask)
                    projected = projected_box(stage.GetPrimAtPath(obj["prim_path"])) if present else None
                    annotations.append(dict(**obj, present=present, mask_file=filename, mask_sha256=sha256(out/filename),
                        visible_pixels=int(mask.sum()), bbox_xyxy=box, projected_geometry_bbox_xyxy=projected,
                        status="visible-geometric-proxy" if box else ("not-present" if not present else "occluded-or-outside-view"),
                        label_pass=mode, bbox_convention="half-open XYXY; max pixel + 1"))
            # Restore original RGB bindings/visibility before advancing to next case.
            configure_pass("rgb", row)
            overlay = Image.fromarray(rgb)
            draw = ImageDraw.Draw(overlay)
            for obj in annotations:
                box = obj["bbox_xyxy"]
                if box:
                    x0, y0, x1, y1 = box
                    color = COLORS[obj["class_name"]]
                    draw.rectangle([x0, y0, x1-1, y1-1], outline=color, width=2)
                    draw.text((x0, max(0, y0-12)), f"{obj['class_name']} {obj['instance_id']}", fill=color, stroke_width=1, stroke_fill="black")
            draw.rectangle([0, 460, 640, 480], fill="black")
            draw.text((5, 464), name + " | GEOMETRIC PROXY LABELS / REVIEW ONLY", fill="white")
            overlay.save(out / "overlays" / (name + ".png"))
            (out / "labels" / (name + ".txt")).write_text("\n".join(yolo_line(o["class_name"], o["bbox_xyxy"])
                for o in annotations if o["bbox_xyxy"]) + "\n", encoding="utf-8")
            record = dict(schema_version=1, sample_id=name, parameters=row, image_file="rgb/"+name+".png",
                require_visible_present_objects=bool(fixture.get("require_visible_present_objects")),
                annotation_capture=annotation_capture,
                image_sha256=sha256(rgb_path), annotations=annotations, training_eligible=False,
                label_policy="separate opaque vial and contents passes; transparent shells/guard excluded from contents occlusion; opaque label/holder/nozzle retained",
                geometry=dict(radius_m=radius, base_z_m=base, liquid_height_m=height if row["level"] else 0,
                              envelope_height_m=max_height, stream_radius_m=row.get("stream_radius_m", 0.0015), nozzle_bottom_z_m=nozzle_z),
                physical_optical_calibration=False)
            records.append(record)
            with (out / "manifest.jsonl").open("a", encoding="utf-8") as stream_file:
                stream_file.write(json.dumps(record) + "\n")
            print("SAVED", name, [(o["class_name"], o["visible_pixels"]) for o in annotations], flush=True)
        after = {str(p.relative_to(assets)): sha256(p) for p in sorted(assets.rglob("*")) if p.is_file()}
        if before != after:
            raise RuntimeError("Source asset hashes changed")
        write_json(out / "run.json", dict(schema_version=1, complete=True, captured=len(records),
            requested=min(args.limit, len(fixture["cases"])), seconds=time.monotonic()-start,
            host=platform.node(), isaac_version=isaac_version, renderer="PathTracing",
            gpu=subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"], text=True).strip(),
            resolution=[640, 480], instance_annotator=args.label_annotator,
            path_tracing_spp=carb.settings.get_settings().get("/rtx/pathtracing/spp"),
            path_tracing_total_spp=carb.settings.get_settings().get("/rtx/pathtracing/totalSpp"),
            subframes=args.subframes, label_pass_rgb_saved=not args.compact,
            annotation_encoding=("unique-object-semantic-tags-v1" if args.label_annotator == "semantic_segmentation" else "instance-prim-paths-v1"),
            fixture_sha256=sha256(args.fixture), source_assets=before,
            script_sha256=sha256(__file__), annotation_helper_sha256=sha256(Path(__file__).with_name("box_annotations.py")),
            source_assets_unchanged=True, camera_parameters_unchanged=True, station_frame=fixture["station_frame"],
            frozen_attributes=len(frozen), training_eligible=False, run5_used=False))
        rgb_ann.detach([rp])
        instance_ann.detach([rp])
        rp.destroy()
    except BaseException:
        import traceback
        failed = True
        write_json(out / "failure.json", dict(complete=False, traceback=traceback.format_exc()))
        traceback.print_exc()
        raise
    finally:
        faulthandler.cancel_dump_traceback_later()
        if "exit_code" in inspect.signature(app.close).parameters:
            app.close(exit_code=int(failed))
        elif failed:
            # Older Kit fast shutdown can swallow an active exception with exit 0.
            # failure.json is durable; preserve failure for automation/launchers.
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(1)
        else:
            app.close()


if __name__ == "__main__":
    main()
