import numpy as np

class ArrheniusDamageParameters:
    def __init__(self, name, A, Ea):
        self.name, self.A, self.Ea = name, A, Ea

class ArrheniusDamageCalculator:
    R = 8.314  # J/mol·K

    def __init__(self):
        # پارامترهای Pearce (2011) — ویرایش شده نسبت به کد اصلی (!)
        self.params = {
            'cartilage': ArrheniusDamageParameters('Cartilage', 7.39e52, 2.09e5),  # ✅ اصلاح شد
            'skin': ArrheniusDamageParameters('Skin', 3.1e98, 6.28e5),
            'fat': ArrheniusDamageParameters('Fat', 9.4e104, 6.69e5)
        }
    
    def calculate_damage_rate(self, T_C, tissue='cartilage'):
        T_K = T_C + 273.15
        p = self.params[tissue]
        return p.A * np.exp(-p.Ea / (self.R * T_K))
    
    def integrate_damage(self, temp_hist, time_hist, tissue='cartilage'):
        damage = np.zeros(len(time_hist))
        for i in range(1, len(time_hist)):
            r1 = self.calculate_damage_rate(temp_hist[i-1], tissue)
            r2 = self.calculate_damage_rate(temp_hist[i], tissue)
            dt = time_hist[i] - time_hist[i-1]
            damage[i] = damage[i-1] + 0.5 * (r1 + r2) * dt
        return damage
    
    def find_optimal_time(self, temp_hist, time_hist, target_tissue='cartilage', surrounding='skin'):
        damage_target = self.integrate_damage(temp_hist, time_hist, target_tissue)
        damage_safe = self.integrate_damage(temp_hist, time_hist, surrounding)
        
        # t_p: کمینه زمانی که Ω_target ≥ 1 و Ω_surrounding < 0.5
        for i, (d_t, d_s, t) in enumerate(zip(damage_target, damage_safe, time_hist)):
            if d_t >= 1.0 and d_s < 0.5:
                return t, d_t, d_s
        return None, damage_target[-1], damage_safe[-1]