# src/core/monte_carlo.py
import numpy as np

try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    prange = range

class TissueLayer:
    """لایه بافت با پارامترهای اپتیکی — برای استفاده در Monte Carlo"""
    def __init__(self, z_min, z_max, mu_a, mu_s, g, n, name=""):
        self.z_min, self.z_max = z_min, z_max
        self.mu_a, self.mu_s, self.g, self.n = mu_a, mu_s, g, n
        self.mu_t = mu_a + mu_s  # ضریب کلی برهمکنش
        self.name = name

class MonteCarloLightTransport:
    """
    شبیه‌سازی انتقال نور در بافت زیستی چندلایه با در نظر گرفتن:
    - پراکندگی Henyey-Greenstein
    - برخورد فرنل در مرزها
    - جذب وابسته به نوع پوست (فیتزپاتریک)
    
    منبع: Wang et al., Phys Med Biol 1995; Jacques, Phys Med Biol 2013.
    """
    
    def __init__(self, tissue_layers, 
                 grid_dimensions=(40, 40, 50),
                 domain_size_cm=(2.5, 2.5, 3.5),
                 beam_radius_cm=1.13,  # → spot area = π × r² = 4.0 cm²
                 wavelength_nm=830):
        """
        Parameters:
        -----------
        tissue_layers : list of TissueLayer
            لیست لایه‌های بافت (از سطح به عمق)
        grid_dimensions : tuple (nx, ny, nz)
            اندازه شبکه برای ذخیره انرژی جذب‌شده
        domain_size_cm : tuple (Lx, Ly, Lz)
            ابعاد دامنه شبیه‌سازی (cm)
        beam_radius_cm : float
            شعاع پرتو گاوسی (cm)
        wavelength_nm : int
            طول موج (برای ارجاع به منابع)
        """
        self.layers = sorted(tissue_layers, key=lambda l: l.z_min)
        self.grid_dims = grid_dimensions
        self.domain_size_cm = domain_size_cm
        self.beam_radius_cm = beam_radius_cm
        self.wavelength_nm = wavelength_nm
        
        # شبکه‌بندی (cm)
        self.x_bins = np.linspace(-domain_size_cm[0]/2, domain_size_cm[0]/2, grid_dimensions[0])
        self.y_bins = np.linspace(-domain_size_cm[1]/2, domain_size_cm[1]/2, grid_dimensions[1])
        self.z_bins = np.linspace(0, domain_size_cm[2], grid_dimensions[2])
        self.dx = self.x_bins[1] - self.x_bins[0]
        self.dy = self.y_bins[1] - self.y_bins[0]
        self.dz = self.z_bins[1] - self.z_bins[0]
        self.dV_cm3 = self.dx * self.dy * self.dz  # cm³
        
        # ماتریس ذخیره انرژی جذب‌شده (J/cm³ پس از ضرب در زمان)
        self.absorbed_energy_cm3 = np.zeros(grid_dimensions)
    
    def _launch_photon_gaussian(self):
        """تابش گاوسی — متناسب با لیزردرمانی بالینی"""
        # r = r0 * sqrt(-ln(rand)) for Gaussian beam
        r = self.beam_radius_cm * np.sqrt(-np.log(np.random.rand()))
        phi = 2 * np.pi * np.random.rand()
        x = r * np.cos(phi)
        y = r * np.sin(phi)
        z = 0.0
        # جهت اولیه: عمود بر سطح
        ux, uy, uz = 0.0, 0.0, 1.0
        weight = 1.0
        return x, y, z, ux, uy, uz, weight
    
    def _get_layer(self, z):
        """پیدا کردن لایه حاوی عمق z"""
        for layer in self.layers:
            if layer.z_min <= z < layer.z_max:
                return layer
        return None
    
    def _fresnel_reflectance(self, n1, n2, cos_theta):
        """محاسبه بازتاب فرنل در مرز دو لایه"""
        if cos_theta <= 0:
            return 1.0
        sin_theta_t_sq = (n1 / n2)**2 * (1 - cos_theta**2)
        if sin_theta_t_sq >= 1.0:  # بازتاب کامل
            return 1.0
        cos_theta_t = np.sqrt(1 - sin_theta_t_sq)
        rs = (n1 * cos_theta - n2 * cos_theta_t) / (n1 * cos_theta + n2 * cos_theta_t)
        rp = (n2 * cos_theta - n1 * cos_theta_t) / (n2 * cos_theta + n1 * cos_theta_t)
        return 0.5 * (rs**2 + rp**2)
    
    def _scatter_photon(self, ux, uy, uz, g):
        """پراکندگی هنی-گرینشتاین"""
        if abs(g) < 1e-5:
            cos_theta = 2 * np.random.rand() - 1
        else:
            temp = (1 - g**2) / (1 - g + 2 * g * np.random.rand())
            cos_theta = (1 + g**2 - temp**2) / (2 * g)
            cos_theta = np.clip(cos_theta, -1, 1)
        sin_theta = np.sqrt(1 - cos_theta**2)
        psi = 2 * np.pi * np.random.rand()
        
        if abs(uz) > 0.99999:
            ux_new = sin_theta * np.cos(psi)
            uy_new = sin_theta * np.sin(psi)
            uz_new = cos_theta * np.sign(uz)
        else:
            temp = np.sqrt(1 - uz**2)
            ux_new = sin_theta * (ux * uz * np.cos(psi) - uy * np.sin(psi)) / temp + ux * cos_theta
            uy_new = sin_theta * (uy * uz * np.cos(psi) + ux * np.sin(psi)) / temp + uy * cos_theta
            uz_new = -sin_theta * np.cos(psi) * temp + uz * cos_theta
        
        # نرمال‌سازی (برای جلوگیری از خطای تجمعی)
        norm = np.sqrt(ux_new**2 + uy_new**2 + uz_new**2)
        return ux_new/norm, uy_new/norm, uz_new/norm
    
    def _deposit_energy(self, x, y, z, energy):
        """ذخیره انرژی جذب‌شده در شبکه"""
        ix = np.searchsorted(self.x_bins, x) - 1
        iy = np.searchsorted(self.y_bins, y) - 1
        iz = np.searchsorted(self.z_bins, z) - 1
        if 0 <= ix < self.grid_dims[0] and 0 <= iy < self.grid_dims[1] and 0 <= iz < self.grid_dims[2]:
            self.absorbed_energy_cm3[ix, iy, iz] += energy
    
    def _propagate_single_photon(self):
        """شبیه‌سازی یک فوتون — نسخه بهینه‌شده برای numba"""
        x, y, z, ux, uy, uz, weight = self._launch_photon_gaussian()
        
        while weight > 1e-4:
            layer = self._get_layer(z)
            if layer is None:
                break
            
            # مسافت آزاد میانگین (cm)
            s = -np.log(np.random.rand()) / layer.mu_t if layer.mu_t > 0 else 1e6
            x_new, y_new, z_new = x + s * ux, y + s * uy, z + s * uz
            
            # برخورد با مرز لایه
            if z_new < layer.z_min or z_new >= layer.z_max:
                # محاسبه مسافت تا مرز
                if uz != 0:
                    if z_new < layer.z_min:
                        s_boundary = (layer.z_min - z) / uz
                    else:
                        s_boundary = (layer.z_max - z) / uz
                    if s_boundary > 0:
                        x_b, y_b, z_b = x + s_boundary * ux, y + s_boundary * uy, z + s_boundary * uz
                else:
                    break
                
                # جذب تا مرز
                energy_absorbed = weight * layer.mu_a / layer.mu_t * (1 - np.exp(-layer.mu_t * s_boundary)) if layer.mu_t > 0 else 0
                self._deposit_energy(x + 0.5 * s_boundary * ux, 
                                   y + 0.5 * s_boundary * uy, 
                                   z + 0.5 * s_boundary * uz, 
                                   energy_absorbed)
                weight *= np.exp(-layer.mu_a * s_boundary)
                x, y, z = x_b, y_b, z_b
                
                # خروج از دامنه جانبی
                if abs(x) > self.domain_size_cm[0]/2 or abs(y) > self.domain_size_cm[1]/2:
                    break
                
                # بازتاب فرنل در مرز
                next_layer = self._get_layer(z + 1e-6 * uz)  # لایه بعدی
                if next_layer is not None:
                    R = self._fresnel_reflectance(layer.n, next_layer.n, abs(uz))
                    if np.random.rand() < R:
                        # بازتاب: معکوس کردن جهت z
                        uz = -uz
                        continue
                
                # پراکندگی در لایه جدید
                if layer.mu_s > 0:
                    ux, uy, uz = self._scatter_photon(ux, uy, uz, layer.g)
            else:
                # حرکت درون لایه
                energy_absorbed = weight * layer.mu_a / layer.mu_t * (1 - np.exp(-layer.mu_t * s)) if layer.mu_t > 0 else 0
                self._deposit_energy(x + 0.5 * s * ux, 
                                   y + 0.5 * s * uy, 
                                   z + 0.5 * s * uz, 
                                   energy_absorbed)
                weight *= np.exp(-layer.mu_a * s)
                x, y, z = x_new, y_new, z_new
                
                if layer.mu_s > 0:
                    ux, uy, uz = self._scatter_photon(ux, uy, uz, layer.g)
            
            # Russian roulette برای فوتون‌های کم‌وزن
            if weight < 1e-3 and np.random.rand() > 0.1:
                break
            elif weight < 1e-3:
                weight /= 0.1
        
        return 1  # success
    
    def run_simulation(self, n_photons=100_000, total_power_mw=500.0):
        """
        اجرای شبیه‌سازی مونت‌کارلو
        
        Returns:
        --------
        dict : شامل:
            - 'volumetric_heat_source_W_cm3': چگالی توان حجمی (W/cm³)
            - 'x_bins_cm', 'y_bins_cm', 'z_bins_cm': مختصات شبکه
        """
        self.absorbed_energy_cm3.fill(0)
        
        # اجرای فوتون‌ها
        for _ in range(n_photons):
            self._propagate_single_photon()
        
        # تبدیل به چگالی توان (W/cm³)
        # انرژی جذب‌شده کل = (absorbed_energy_cm3) × (توان کل / وزن کل فوتون‌ها)
        total_absorbed = self.absorbed_energy_cm3.sum()
        if total_absorbed == 0:
            scale_factor = 0
        else:
            scale_factor = (total_power_mw / 1000.0) / total_absorbed  # W / (J) → W/s = W
        
        heat_source_W_cm3 = self.absorbed_energy_cm3 * scale_factor
        
        return {
            'volumetric_heat_source_W_cm3': heat_source_W_cm3,
            'volumetric_heat_source_W_m3': heat_source_W_cm3 * 1e6,  # برای BHTE
            'x_bins_cm': self.x_bins,
            'y_bins_cm': self.y_bins,
            'z_bins_cm': self.z_bins,
            'grid_dimensions': self.grid_dims,
            'domain_size_cm': self.domain_size_cm
        }
# if NUMBA_AVAILABLE:
#     MonteCarloLightTransport._launch_photon_gaussian = jit(nopython=True)(
#         MonteCarloLightTransport._launch_photon_gaussian
#     )
#     MonteCarloLightTransport._fresnel_reflectance = jit(nopython=True)(
#         MonteCarloLightTransport._fresnel_reflectance
#     )
#     MonteCarloLightTransport._scatter_photon = jit(nopython=True)(
#         MonteCarloLightTransport._scatter_photon
#     )
