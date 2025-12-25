import numpy as np
from .monte_carlo import TissueLayer


def estimate_fat_thickness_from_bmi(
    bmi,
    fat_ref_cm=0.6,
    bmi_ref=25.0,
    slope_cm_per_bmi=0.02,
    min_fat=0.3,
    max_fat=1.5
):
    """
    Estimate subcutaneous fat thickness from BMI (cm)
    """
    fat = fat_ref_cm + slope_cm_per_bmi * (bmi - bmi_ref)
    return float(np.clip(fat, min_fat, max_fat))
class ClinicalDosimetry:
    """
    Clinical dosimetry post-processing for LLLT
    """

    # Safety limits
    MAX_INTENSITY_MW_CM2 = 100.0
    MAX_FLUENCE_J_CM2 = 12.0
    MAX_SKIN_TEMP_C = 41.0
    MAX_CARTILAGE_TEMP_C = 45.0
    @staticmethod
    def build_tissue_layers_from_bmi(bmi):
        z = 0.0
        layers = []

        layers.append(TissueLayer(z, z+0.2, 0.1, 100, 0.9, 1.4, "Skin"))
        z += 0.2

        fat = estimate_fat_thickness_from_bmi(bmi)
        layers.append(TissueLayer(z, z+fat, 0.05, 50, 0.9, 1.38, "Fat"))
        z += fat

        layers.append(TissueLayer(z, z+0.7, 0.15, 80, 0.9, 1.37, "Muscle"))
        z += 0.7

        layers.append(TissueLayer(z, z+1.0, 0.08, 60, 0.85, 1.35, "Joint"))
        z += 1.0

        layers.append(TissueLayer(z, z+0.5, 0.18, 90, 0.88, 1.38, "Cartilage"))

        return layers
    @staticmethod
    def extract_point_history(field_history, index):
        ix, iy, iz = index
        return np.array([f[ix, iy, iz] for f in field_history])
    def compute(
        self,
        exposure_time_s,
        power_mW,
        spot_size_cm2,
        q_laser_W_m3=None,
        T_history=None,
        time_history=None,
        omega_cart=None,
        omega_cart_history=None,
        omega_skin=None,
        target_index=None
    ):
        r = {}

        # --- Basic dosimetry
        r["exposure_time_s"] = float(exposure_time_s)
        r["power_mW"] = float(power_mW)
        r["spot_size_cm2"] = float(spot_size_cm2)

        r["total_energy_J"] = (power_mW / 1000) * exposure_time_s
        r["intensity_mW_cm2"] = power_mW / spot_size_cm2
        r["fluence_J_cm2"] = r["total_energy_J"] / spot_size_cm2
        if q_laser_W_m3 is not None:
            dose = q_laser_W_m3 * exposure_time_s
            r["absorbed_dose_map_J_cm3"] = dose / 1e6

            if target_index:
                try:
                    r["absorbed_dose_target_J_cm3"] = float(
                        r["absorbed_dose_map_J_cm3"][target_index]
                    )
                except IndexError:
                    r["absorbed_dose_target_J_cm3"] = None
        r["safe_intensity"] = r["intensity_mW_cm2"] <= self.MAX_INTENSITY_MW_CM2
        r["safe_fluence"] = r["fluence_J_cm2"] <= self.MAX_FLUENCE_J_CM2

        if T_history:
            final_T = T_history[-1]
            r["skin_temp_C"] = float(final_T[:, :, 0].max())
            r["cartilage_temp_C"] = float(final_T.max())

            r["safe_skin_temp"] = r["skin_temp_C"] <= self.MAX_SKIN_TEMP_C
            r["safe_cartilage_temp"] = r["cartilage_temp_C"] <= self.MAX_CARTILAGE_TEMP_C
        else:
            r["safe_skin_temp"] = True
            r["safe_cartilage_temp"] = True
        r["relaxation_time_s"] = None
        # Use omega_cart_history if provided, otherwise omega_cart
        omega_cart_data = omega_cart_history if omega_cart_history is not None else omega_cart
        if omega_cart_data and time_history:
            max_omega = max(omega_cart_data) if omega_cart_data else 0
            for t, oc in zip(time_history, omega_cart_data):
                if t > exposure_time_s and oc <= 0.1 * max_omega:
                    if omega_skin is None:
                        r["relaxation_time_s"] = t - exposure_time_s
                        break
                    # If omega_skin is provided, check both conditions
                    elif len(omega_skin) > len(time_history[:len(omega_cart_data)]):
                        os_idx = len(time_history[:len(omega_cart_data)]) - 1
                        if omega_skin[os_idx] <= 0.05:
                            r["relaxation_time_s"] = t - exposure_time_s
                            break
                    else:
                        r["relaxation_time_s"] = t - exposure_time_s
                        break
        safe = all([
            r["safe_intensity"],
            r["safe_fluence"],
            r["safe_skin_temp"],
            r["safe_cartilage_temp"]
        ])

        r["recommendation"] = " ایمن و قابل تجویز" if safe else "⚠️ نیاز به بازبینی"
        return r
