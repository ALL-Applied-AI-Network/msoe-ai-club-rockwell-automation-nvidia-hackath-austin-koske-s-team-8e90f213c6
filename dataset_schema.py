"""
Standardized Dataset Schema & Multi-Format Annotator
====================================================
Formats synthetic dataset outputs into industrial and modern AI formats:
1. Comprehensive Industrial Metadata JSON (Digital Twin State)
2. YOLO Bounding Box & Class Format (Object Detection / Defect Localization)
3. NVIDIA Cosmos Reason / VLM Conversation JSONL (Vision-Language Reasoning)
4. Tabular CSV Manifest (Fast Data Loading & Filtering)
"""

import os
import json
import csv
from typing import Dict, Any, List

# Class Definitions for YOLO
YOLO_CLASSES = {
    0: "vial_nominal",
    1: "vial_carryover_defect",
    2: "vial_volume_defect",
    3: "meniscus",
    4: "dispense_nozzle"
}

class DatasetAnnotator:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.images_dir = os.path.join(output_dir, "images")
        self.labels_yolo_dir = os.path.join(output_dir, "labels_yolo")
        self.metadata_dir = os.path.join(output_dir, "metadata")
        self.manifest_path = os.path.join(output_dir, "dataset_manifest.csv")
        self.vlm_jsonl_path = os.path.join(output_dir, "cosmos_reason_prompts.jsonl")
        
        for d in [self.images_dir, self.labels_yolo_dir, self.metadata_dir]:
            os.makedirs(d, exist_ok=True)

        # Clear existing VLM JSONL to prevent duplicate records across multiple runs
        if os.path.exists(self.vlm_jsonl_path):
            try:
                os.remove(self.vlm_jsonl_path)
            except OSError:
                pass
            
        self.manifest_records: List[Dict[str, Any]] = []

    def export_sample(
        self,
        sample_id: str,
        image_filename: str,
        sim_data: Dict[str, Any],
        camera_params: Dict[str, Any],
        lighting_params: Dict[str, Any],
        bbox_coords: Dict[str, List[float]] # {"vial": [xc, yc, w, h], "meniscus": [xc, yc, w, h], "nozzle": [xc, yc, w, h]}
    ) -> Dict[str, str]:
        """
        Exports annotations for a single synthetic image across all formats.
        """
        # 1. Industrial Metadata JSON
        meta_filename = f"{sample_id}.json"
        meta_filepath = os.path.join(self.metadata_dir, meta_filename)
        
        full_metadata = {
            "sample_id": sample_id,
            "image_filename": image_filename,
            "simulation_engine": "NVIDIA Isaac Sim Replicator / PhysicsFluidShader",
            "physical_state": sim_data,
            "camera_parameters": camera_params,
            "lighting_parameters": lighting_params,
            "bounding_boxes_normalized": bbox_coords
        }
        
        with open(meta_filepath, "w", encoding="utf-8") as f:
            json.dump(full_metadata, f, indent=2)

        # 2. YOLO Annotation File (.txt)
        yolo_filename = f"{sample_id}.txt"
        yolo_filepath = os.path.join(self.labels_yolo_dir, yolo_filename)
        
        yolo_lines = []
        # Determine vial class
        if sim_data["is_defective"]:
            vial_class = 1  # vial_carryover_defect
        else:
            vial_class = 0  # vial_nominal

        # Vial bounding box
        if "vial" in bbox_coords:
            vb = bbox_coords["vial"]
            yolo_lines.append(f"{vial_class} {vb[0]:.6f} {vb[1]:.6f} {vb[2]:.6f} {vb[3]:.6f}")

        # Meniscus bounding box
        if "meniscus" in bbox_coords:
            mb = bbox_coords["meniscus"]
            yolo_lines.append(f"3 {mb[0]:.6f} {mb[1]:.6f} {mb[2]:.6f} {mb[3]:.6f}")

        # Nozzle bounding box
        if "nozzle" in bbox_coords:
            nb = bbox_coords["nozzle"]
            yolo_lines.append(f"4 {nb[0]:.6f} {nb[1]:.6f} {nb[2]:.6f} {nb[3]:.6f}")

        with open(yolo_filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(yolo_lines) + "\n")

        # 3. NVIDIA Cosmos Reason / VLM Prompt Entry
        thresh_val = sim_data.get("defect_threshold", 10.0)
        vlm_record = {
            "id": sample_id,
            "image": os.path.join("images", image_filename),
            "conversations": [
                {
                    "from": "human",
                    "value": (
                        f"Inspect this filled vial from the Rockwell fluid dispensing line. "
                        f"Recipe formulation targets {sim_data['nominal_volumes']['total_ml']} mL total: "
                        f"T-100(Clear)={sim_data['nominal_volumes']['t100_ml']}mL, "
                        f"T-200(Orange)={sim_data['nominal_volumes']['t200_ml']}mL, "
                        f"T-400(Blue)={sim_data['nominal_volumes']['t400_ml']}mL. "
                        f"This is batch position {sim_data['batch_position']}. "
                        f"Evaluate whether the vial exhibits off-spec discoloration, identify the likely root cause, and recommend operator action."
                    )
                },
                {
                    "from": "gpt",
                    "value": (
                        f"Verdict: {'DEFECT DETECTED' if sim_data['is_defective'] else 'NOMINAL PASS'}.\n"
                        f"Color Distance (ΔE): {sim_data['delta_e']:.1f} (Threshold: {thresh_val:.1f}).\n"
                        f"Observed RGB: {sim_data['actual_rgb']}.\n"
                        f"Target Expected RGB: {sim_data['target_rgb']}.\n"
                        f"Root Cause: {sim_data['root_cause']}\n"
                        f"Prescribed Mitigation: {'Execute a 15 mL T-100 clear diluent flush before subsequent batches.' if sim_data['is_defective'] else 'Line clean; proceed to downstream cap/seal station.'}"
                    )
                }
            ]
        }
        with open(self.vlm_jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(vlm_record) + "\n")

        # 4. Manifest Record
        light_val = lighting_params.get("intensity_gain", lighting_params.get("intensity", 1.0))
        rec = {
            "sample_id": sample_id,
            "image_filename": image_filename,
            "batch_position": sim_data["batch_position"],
            "t100_ml": sim_data["actual_volumes"]["t100_ml"],
            "t200_ml": sim_data["actual_volumes"]["t200_ml"],
            "t400_ml": sim_data["actual_volumes"]["t400_ml"],
            "total_ml": sim_data["actual_volumes"]["total_ml"],
            "actual_r": sim_data["actual_rgb"][0],
            "actual_g": sim_data["actual_rgb"][1],
            "actual_b": sim_data["actual_rgb"][2],
            "delta_e": sim_data["delta_e"],
            "is_defective": 1 if sim_data["is_defective"] else 0,
            "defect_class": sim_data["defect_class"],
            "carryover_fraction": sim_data["carryover_fraction"],
            "holdup_ml": sim_data["simulated_holdup_ml"],
            "light_intensity": round(float(light_val), 3)
        }
        self.manifest_records.append(rec)
        
        return {
            "metadata_path": meta_filepath,
            "yolo_path": yolo_filepath
        }

    def finalize(self):
        """Finalizes the dataset by writing the tabular manifest and YOLO dataset config with 80/20 train/val split."""
        if self.manifest_records:
            fieldnames = list(self.manifest_records[0].keys())
            with open(self.manifest_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.manifest_records)
            print(f"Exported tabular manifest to {self.manifest_path} ({len(self.manifest_records)} samples)")

        # Create train and val image lists (80/20 split)
        all_imgs = [os.path.abspath(os.path.join(self.images_dir, r["image_filename"])) for r in self.manifest_records]
        num_train = int(len(all_imgs) * 0.8)
        train_imgs = all_imgs[:num_train]
        val_imgs = all_imgs[num_train:]

        train_txt_path = os.path.join(self.output_dir, "train.txt")
        val_txt_path = os.path.join(self.output_dir, "val.txt")

        with open(train_txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(train_imgs) + "\n")
        with open(val_txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(val_imgs) + "\n")

        # Write YOLO data.yaml
        yolo_yaml_path = os.path.join(self.output_dir, "data.yaml")
        yaml_content = f"""# YOLOv8 / YOLOv11 Dataset Configuration
# CSI Vial Fill Physical AI Defect Detection
path: {os.path.abspath(self.output_dir)}
train: {os.path.abspath(train_txt_path)}
val: {os.path.abspath(val_txt_path)}

names:
  0: vial_nominal
  1: vial_carryover_defect
  2: vial_volume_defect
  3: meniscus
  4: dispense_nozzle
"""
        with open(yolo_yaml_path, "w", encoding="utf-8") as f:
            f.write(yaml_content)
        print(f"Exported YOLO config to {yolo_yaml_path} ({len(train_imgs)} train / {len(val_imgs)} val)")
