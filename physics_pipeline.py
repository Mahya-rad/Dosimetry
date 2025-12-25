# physics_pipeline.py
# Full physics-based simulation pipeline for LLLT dosimetry

import numpy as np

from src.core.patient_model import estimate_fat_thickness_from_bmi
from src.core.monte_carlo import MonteCarloLightTransport, TissueLayer
from src.core.bioheat import PennesBioheatSolver
from src.core.arrhenius import ArrheniusDamageCalculator
from src.core.clinical_utils import ClinicalDosimetry


def run_advanced_simulation(
    bmi: float,
    exposure_time_s: float,
    power_mW: float,
    spot_size_cm2: float,
    grid_size=(51, 51, 60),
    domain_size_cm=(4.0, 4.0, 3.0),
    target_tissue="cartilage",
    verbose=True
):
    """
    Full physics-based LLLT simulation:
    Monte Carlo → Bioheat → Arrhenius → Clinical metrics
    """

    results = {}

    # ======================================================
    # 1. Build Tissue Layers from BMI
    # ======================================================
    fat_thickness = estimate_fat_thickness_from_bmi(bmi)

    layers = []
    z = 0.0

    layers.append(TissueLayer(z, z+0.2, mu_a=0.1, mu_s=100, g=0.9, n=1.4, name="Skin"))
    z += 0.2

    layers.append(TissueLayer(z, z+fat_thickness, mu_a=0.05, mu_s=50, g=0.9, n=1.38, name="Fat"))
    z += fat_thickness

    layers.append(TissueLayer(z, z+0.7, mu_a=0.15, mu_s=80, g=0.9, n=1.37, name="Muscle"))
    z += 0.7

    layers.append(TissueLayer(z, z+1.0, mu_a=0.08, mu_s=60, g=0.85, n=1.35, name="Joint"))
    z += 1.0

    layers.append(TissueLayer(z, z+0.5, mu_a=0.18, mu_s=90, g=0.88, n=1.38, name="Cartilage"))

    if verbose:
        print("✓ Tissue layers built from BMI")
    # ======================================================
    # 2. Monte Carlo Light Transport
    # ======================================================

    mc = MonteCarloLightTransport(
        tissue_layers=layers,
        grid_dimensions=grid_size,
        domain_size_cm=domain_size_cm,
        beam_radius_cm=np.sqrt(spot_size_cm2 / np.pi)
    )

    mc_results = mc.run_simulation(
        n_photons=200_000,
        total_power_mw=power_mW
    )

    # heat source
    q_laser_W_m3 = mc_results["volumetric_heat_source_W_m3"]

    results["mc"] = mc_results

    if verbose:
     print("✓ Monte Carlo simulation completed")

    # ======================================================
    # 3. Pennes Bioheat Equation
    # ======================================================
    bioheat = PennesBioheatSolver(
        grid_size=grid_size,
        domain_size=domain_size_cm,
        tissue_layers=layers
    )

    bioheat.set_laser_heat_source_from_mc(
        mc_results=mc_results,
        mc_simulator=mc,
        power_scale=1.0
    )

    dt = bioheat.compute_time_step()
    bioheat.solve_explicit(
        dt=dt,
        total_time=exposure_time_s + 120,
        save_interval=2.0
    )

    results["temperature_history"] = bioheat.temp_history
    results["time_history"] = bioheat.time_history

    if verbose:
        print("✓ Bioheat equation solved")

    # ======================================================
    # 4. Arrhenius Thermal Damage
    # ======================================================
    arrhenius = ArrheniusDamageCalculator()

    ix = grid_size[0] // 2
    iy = grid_size[1] // 2
    iz = grid_size[2] - 3
    target_index = (ix, iy, iz)

    T_target = [T[ix, iy, iz] for T in bioheat.temp_history]

    omega_target = arrhenius.integrate_damage(
        np.array(T_target),
        np.array(bioheat.time_history),
        tissue_type=target_tissue
    )

    results["omega_target"] = omega_target
    results["target_index"] = target_index

    if verbose:
        print("✓ Arrhenius damage computed")

    # ======================================================
    # 5. Relaxation Time
    # ======================================================
    omega_max = omega_target.max()
    relaxation_time = None

    for t, omega in zip(bioheat.time_history, omega_target):
        if t > exposure_time_s and omega <= 0.1 * omega_max:
            relaxation_time = t - exposure_time_s
            break

    results["relaxation_time_s"] = relaxation_time

    if verbose:
        print("✓ Relaxation time evaluated")

    # ======================================================
    # 6. Clinical Dosimetry
    # ======================================================
    dosimetry = ClinicalDosimetry()

    clinical_outputs = dosimetry.compute(
        exposure_time_s=exposure_time_s,
        power_mW=power_mW,
        spot_size_cm2=spot_size_cm2,
        q_laser_W_m3=q_laser * 1e6,
        T_history=bioheat.temp_history,
        time_history=bioheat.time_history,
        omega_cart_history=omega_target,
        omega_skin_history=np.zeros_like(omega_target),
        target_index=target_index
    )

    results["clinical"] = clinical_outputs

    if verbose:
        print("✓ Clinical dosimetry computed")

    return results
