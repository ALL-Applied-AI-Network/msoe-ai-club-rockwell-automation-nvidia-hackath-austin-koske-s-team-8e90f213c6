"""
Physics-Informed Fluid Mixing & Subtractive Optical Shader
==========================================================
Models the physical fluid dynamics and optical absorption for the
Rockwell Automation + CSI liquid-dosing testbed in NVIDIA Isaac Sim.

Features:
- Beer-Lambert subtractive optical absorption (cures linear RGB blend failure)
- Manifold hold-up volume dead-zone modeling with domain randomization
- Volumetric fill height to geometric USD mesh scale mapping
- Automated ground-truth defect labeling (Defect severity, Delta E, RCA explanation)
- MDL and USD Preview Surface material parameter synthesis
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional

class PhysicsFluidShader:
    def __init__(
        self,
        base_holdup_volume_ml: float = 13.0,
        holdup_std_ml: float = 2.0,
        defect_threshold_delta_e: float = 10.0,
        vial_capacity_ml: float = 80.0
    ):
        self.base_holdup = base_holdup_volume_ml
        self.holdup_std = holdup_std_ml
        self.threshold = defect_threshold_delta_e
        self.vial_capacity = vial_capacity_ml
        
        # Calibrated incident backdrop illumination (white backlight diffusers)
        self.I0 = np.array([180.0, 185.0, 180.0])
        
        # Calibrated pure dye extinction coefficients (A = -log10(I / I0) for 100% pure reagent)
        # Measured directly from CSI testbed reference fills:
        # T100 (Water): clear transmission across all wavelengths
        # T200 (Orange): strongly absorbs blue (5.84), moderate green (3.32), passes red (1.72)
        # T400 (Blue): strongly absorbs red (2.38), moderate green (0.98), passes blue (0.76)
        self.extinction_coefficients = {
            "T100": np.array([0.02, 0.02, 0.02]),  # Clear diluent (water)
            "T200": np.array([1.72, 3.32, 5.84]),  # Pure Orange azo dye
            "T400": np.array([2.38, 0.98, 0.76]),  # Pure Blue dye
        }
        
        # Standard dispense order on CSI line: T200 -> T400 -> T100
        self.dispense_order = ["T200", "T400", "T100"]

    @staticmethod
    def rgb_to_lab(rgb_norm: np.ndarray) -> np.ndarray:
        """
        Converts normalized RGB [0..1] array of shape (N, 3) or (3,) to CIE L*a*b* space.
        """
        rgb_arr = np.atleast_2d(rgb_norm)
        mask = rgb_arr > 0.04045
        linear = np.zeros_like(rgb_arr)
        linear[mask] = ((rgb_arr[mask] + 0.055) / 1.055) ** 2.4
        linear[~mask] = rgb_arr[~mask] / 12.92

        M = np.array([
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041]
        ])
        xyz = np.dot(linear, M.T)

        Xn, Yn, Zn = 0.95047, 1.00000, 1.08883
        xyz[:, 0] /= Xn
        xyz[:, 1] /= Yn
        xyz[:, 2] /= Zn

        delta = 6.0 / 29.0
        def f(t):
            mask_t = t > delta**3
            out = np.zeros_like(t)
            out[mask_t] = np.cbrt(t[mask_t])
            out[~mask_t] = t[~mask_t] / (3 * delta**2) + (4.0 / 29.0)
            return out

        fx = f(xyz[:, 0])
        fy = f(xyz[:, 1])
        fz = f(xyz[:, 2])

        L = 116.0 * fy - 16.0
        a = 500.0 * (fx - fy)
        b = 200.0 * (fy - fz)
        lab = np.column_stack([L, a, b])
        return lab[0] if rgb_norm.ndim == 1 else lab

    def compute_optical_absorbance(self, t100_ml: float, t200_ml: float, t400_ml: float, total_ml: float) -> np.ndarray:
        """
        Computes theoretical optical absorbance vector [A_R, A_G, A_B]
        using linear superposition of dye concentrations (Beer-Lambert law).
        """
        total_ml = max(total_ml, 1e-3)
        c100 = t100_ml / total_ml
        c200 = t200_ml / total_ml
        c400 = t400_ml / total_ml
        
        A = (c100 * self.extinction_coefficients["T100"] +
             c200 * self.extinction_coefficients["T200"] +
             c400 * self.extinction_coefficients["T400"])
        return A

    def absorbance_to_rgb(self, A: np.ndarray, illumination_multiplier: float = 1.0) -> np.ndarray:
        """
        Converts optical absorbance to transmitted RGB color vector.
        """
        I = (self.I0 * illumination_multiplier) * (10.0 ** (-A))
        return np.clip(I, 0.0, 255.0)

    def compute_mixture_rgb(self, t100_ml: float, t200_ml: float, t400_ml: float, total_ml: float = None) -> np.ndarray:
        """
        Convenience method to compute transmitted RGB for a given recipe volume mixture.
        """
        if total_ml is None or total_ml <= 0:
            total_ml = t100_ml + t200_ml + t400_ml
        A = self.compute_optical_absorbance(t100_ml, t200_ml, t400_ml, total_ml)
        return self.absorbance_to_rgb(A)

    def simulate_recipe_transition(
        self,
        prior_recipe: Dict[str, float],
        next_recipe: Dict[str, float],
        batch_position: int,
        randomize_holdup: bool = True
    ) -> Dict[str, Any]:
        """
        Simulates the physical fluid transition from prior_recipe to next_recipe
        for a specific vial batch position (1 = transition, 2 = flush, 3 = steady).

        Returns:
            Dictionary containing:
            - physical volumes in vial (t100, t200, t400, total_ml)
            - physical color (transmitted RGB, normalized RGB, optical absorbance)
            - nominal target color (ideal recipe without carryover)
            - defect metrics (delta_e, delta_e_rgb, is_defective, defect_class)
            - root cause explanation
            - USD material parameters for Isaac Sim rendering
        """
        V_next = float(next_recipe.get("total_ml", 70.0))
        V_prior = max(float(prior_recipe.get("total_ml", 70.0)), 1.0)
        
        # Sample manifold holdup volume with industrial manufacturing noise
        if randomize_holdup:
            V_hold = max(4.0, float(np.random.normal(self.base_holdup, self.holdup_std)))
        else:
            V_hold = float(self.base_holdup)

        # Nominal target recipe absorbance and color
        A_target = self.compute_optical_absorbance(
            float(next_recipe.get("t100_ml", 0.0)),
            float(next_recipe.get("t200_ml", 0.0)),
            float(next_recipe.get("t400_ml", 0.0)),
            V_next
        )
        target_rgb = self.absorbance_to_rgb(A_target)

        # Mass-conserving transferred carryover volume based on batch position:
        # Vial 1 receives primary manifold dead-zone fluid; subsequent vials clear exponentially
        if batch_position == 1:
            v_transferred = min(V_hold, 0.65 * V_next)
        elif batch_position == 2:
            v_transferred = min(V_hold * 0.20, 0.18 * V_next)
        else:
            v_transferred = 0.0

        v_fresh = max(0.0, V_next - v_transferred)
        carryover_fraction = v_transferred / V_next

        # True physical volumetric mixture inside the vial (mass-conserving)
        f_prior = v_transferred / V_prior
        f_next = v_fresh / V_next

        actual_t100 = f_prior * float(prior_recipe.get("t100_ml", 0.0)) + f_next * float(next_recipe.get("t100_ml", 0.0))
        actual_t200 = f_prior * float(prior_recipe.get("t200_ml", 0.0)) + f_next * float(next_recipe.get("t200_ml", 0.0))
        actual_t400 = f_prior * float(prior_recipe.get("t400_ml", 0.0)) + f_next * float(next_recipe.get("t400_ml", 0.0))
        actual_total = actual_t100 + actual_t200 + actual_t400

        # Subtractive optical absorbance of the resulting mixture
        A_actual = self.compute_optical_absorbance(actual_t100, actual_t200, actual_t400, actual_total)
        actual_rgb = self.absorbance_to_rgb(A_actual)

        # Standard CIE L*a*b* Delta E calculation (perceptual color space)
        actual_lab = self.rgb_to_lab(actual_rgb / 255.0)
        target_lab = self.rgb_to_lab(target_rgb / 255.0)
        delta_e = float(np.linalg.norm(actual_lab - target_lab))
        delta_e_rgb = float(np.linalg.norm(actual_rgb - target_rgb))

        is_defective = delta_e >= self.threshold

        # Defect classification
        if is_defective:
            defect_class = "carryover_contamination"
            root_cause = (
                f"Residual manifold holdup ({V_hold:.1f} mL) from Recipe {prior_recipe.get('recipe_id', 'prev')} "
                f"contaminated Recipe {next_recipe.get('recipe_id', 'curr')} on batch position {batch_position}. "
                f"Predicted color divergence ΔE = {delta_e:.1f} exceeds threshold ({self.threshold:.1f})."
            )
        else:
            defect_class = "nominal"
            root_cause = "Nominal batch fill within acceptable color tolerance."

        # USD / Isaac Sim Material Parameters
        # Liquid height (normalized 0 to 1 based on 80 mL max capacity)
        fill_fraction = np.clip(actual_total / self.vial_capacity, 0.1, 1.0)
        
        # Convert RGB [0..255] to normalized linear color for USD Preview Surface / MDL
        norm_rgb = (actual_rgb / 255.0).tolist()
        
        # Transmission / scattering properties for photorealistic fluid rendering
        # High dye concentration reduces transmission, increasing opacity/scatter
        total_dye_ml = actual_t200 + actual_t400
        opacity = float(np.clip(0.45 + (total_dye_ml / actual_total) * 0.50, 0.45, 0.95))
        
        usd_material_props = {
            "diffuse_color": norm_rgb,
            "transmittance_color": norm_rgb,
            "opacity": opacity,
            "ior": 1.333,             # Water/aqueous dye refractive index
            "roughness": 0.02,         # Smooth fluid surface
            "specular": 0.95,
            "mesh_scale_z": float(fill_fraction)  # Z-scale factor for fluid cylinder primitive
        }

        return {
            "batch_position": batch_position,
            "actual_volumes": {
                "t100_ml": round(float(actual_t100), 2),
                "t200_ml": round(float(actual_t200), 2),
                "t400_ml": round(float(actual_t400), 2),
                "total_ml": round(float(actual_total), 2)
            },
            "nominal_volumes": {
                "t100_ml": float(next_recipe.get("t100_ml", 0.0)),
                "t200_ml": float(next_recipe.get("t200_ml", 0.0)),
                "t400_ml": float(next_recipe.get("t400_ml", 0.0)),
                "total_ml": float(V_next)
            },
            "actual_rgb": [round(float(x), 2) for x in actual_rgb],
            "target_rgb": [round(float(x), 2) for x in target_rgb],
            "delta_e": round(delta_e, 2),
            "delta_e_rgb": round(delta_e_rgb, 2),
            "is_defective": is_defective,
            "defect_class": defect_class,
            "root_cause": root_cause,
            "carryover_fraction": round(float(carryover_fraction), 3),
            "simulated_holdup_ml": round(float(V_hold), 2),
            "fill_fraction": round(float(fill_fraction), 3),
            "usd_material": usd_material_props
        }
