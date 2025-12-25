# patient_model.py
# Patient-specific preprocessing for LLLT dosimetry
# Stage-1 (clinical prior) planning — NO physics solver here

import numpy as np

# ======================================================
# 1. Anthropometric Parameters
# ======================================================

def compute_bmi(height_cm: float, weight_kg: float) -> float:
    if height_cm <= 0:
        raise ValueError("Height must be positive")
    return weight_kg / (height_cm / 100.0) ** 2


# ======================================================
# 2. BMI → Fat Thickness Mapping
# ======================================================

def estimate_fat_thickness_from_bmi(bmi: float) -> float:
    """
    Empirical approximation (cm)
    """
    fat_cm = 0.04 * bmi - 0.5
    return float(np.clip(fat_cm, 0.2, 2.0))


# ======================================================
# 3. Disease Severity Modifiers
# ======================================================

def kl_time_modifier(kl_grade: int) -> float:
    return {
        1: -10.0,
        2: 0.0,
        3: +20.0,
        4: +40.0
    }.get(int(kl_grade), 0.0)


def bmi_time_modifier(bmi: float) -> float:
    return 0.8 * (bmi - 25.0)


# ======================================================
# 4. Initial Exposure Time (Clinical Prior)
# ======================================================

def estimate_initial_exposure_time(
    bmi: float,
    kl_grade: int,
    base_time: float = 60.0,
    min_time: float = 30.0,
    max_time: float = 180.0
) -> float:
    t_est = base_time + kl_time_modifier(kl_grade) + bmi_time_modifier(bmi)
    return float(np.clip(t_est, min_time, max_time))


# ======================================================
# 5. Power Limitation (Skin + BMI)
# ======================================================

def limit_power_by_skin_and_bmi(
    requested_power_mw: float,
    fitzpatrick_type: int,
    bmi: float
) -> float:
    max_power_by_fitz = {
        1: 500,
        2: 500,
        3: 450,
        4: 400,
        5: 350,
        6: 300
    }

    max_power = max_power_by_fitz.get(int(fitzpatrick_type), 500)

    if bmi > 30:
        max_power -= 20.0 * (bmi - 30.0) / 5.0

    return float(np.clip(requested_power_mw, 200.0, max_power))


# ======================================================
# 6. Basic Clinical Outputs (No Simulation)
# ======================================================

def compute_basic_clinical_outputs(
    exposure_time_s: float,
    power_mw: float,
    spot_size_cm2: float
) -> dict:

    power_W = power_mw / 1000.0
    total_energy_J = power_W * exposure_time_s
    intensity_mW_cm2 = power_mw / spot_size_cm2
    fluence_J_cm2 = total_energy_J / spot_size_cm2

    return {
        "exposure_time_s": round(exposure_time_s, 1),
        "power_mW": round(power_mw, 1),
        "total_energy_J": round(total_energy_J, 2),
        "intensity_mW_cm2": round(intensity_mW_cm2, 1),
        "fluence_J_cm2": round(fluence_J_cm2, 2)
    }


# ======================================================
# 7. Stage-1 Planning Wrapper (for UI)
# ======================================================

def run_basic_planning(
    age: int,
    height_cm: float,
    weight_kg: float,
    kl_grade: int,
    fitzpatrick_type: int,
    device_power_mw: float,
    spot_area_cm2: float
) -> dict:
    """
    Lightweight clinical planning — no physics simulation
    """

    bmi = compute_bmi(height_cm, weight_kg)

    exposure_time = estimate_initial_exposure_time(
        bmi=bmi,
        kl_grade=kl_grade
    )

    effective_power = limit_power_by_skin_and_bmi(
        requested_power_mw=device_power_mw,
        fitzpatrick_type=fitzpatrick_type,
        bmi=bmi
    )

    basic_outputs = compute_basic_clinical_outputs(
        exposure_time_s=exposure_time,
        power_mw=effective_power,
        spot_size_cm2=spot_area_cm2
    )

    return {
        "age": age,
        "BMI": round(bmi, 1),
        "fat_thickness_cm": round(estimate_fat_thickness_from_bmi(bmi), 2),
        "KL_grade": int(kl_grade),
        "fitzpatrick": int(fitzpatrick_type),
        "power_effective_mW": round(effective_power, 1),
        **basic_outputs
    }
