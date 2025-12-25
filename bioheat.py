import numpy as np

class TissueThermalProperties:
    def __init__(self, name, rho, c, k, omega_b, Q_met=0.0):
        self.name = name
        self.rho, self.c, self.k, self.omega_b, self.Q_met = rho, c, k, omega_b, Q_met

class PennesBioheatSolver:
    # Thermal properties database (rho: kg/m³, c: J/(kg·K), k: W/(m·K), omega_b: 1/s, Q_met: W/m³)
    THERMAL_PROPERTIES = {
        "skin": TissueThermalProperties("skin", rho=1109, c=3391, k=0.37, omega_b=0.001, Q_met=1620),
        "fat": TissueThermalProperties("fat", rho=911, c=2348, k=0.21, omega_b=0.0003, Q_met=300),
        "muscle": TissueThermalProperties("muscle", rho=1090, c=3421, k=0.49, omega_b=0.0008, Q_met=684),
        "joint": TissueThermalProperties("joint", rho=1100, c=3500, k=0.42, omega_b=0.0005, Q_met=500),
        "cartilage": TissueThermalProperties("cartilage", rho=1100, c=3500, k=0.45, omega_b=0.0002, Q_met=400),
    }
    
    def __init__(self, grid_size, domain_size, tissue_layers, T0=37.0):
        self.nx, self.ny, self.nz = grid_size
        self.Lx, self.Ly, self.Lz = domain_size[0]/100, domain_size[1]/100, domain_size[2]/100  # cm → m
        self.dx = self.Lx / (grid_size[0]-1)
        self.dy = self.Ly / (grid_size[1]-1)
        self.dz = self.Lz / (grid_size[2]-1)
        self.x = np.linspace(0, self.Lx, grid_size[0])
        self.y = np.linspace(0, self.Ly, grid_size[1])
        self.z = np.linspace(0, self.Lz, grid_size[2])
        self.T = np.full((self.nx, self.ny, self.nz), T0)
        self.T0 = T0
        self.tissue_layers = tissue_layers
        self._init_properties()
        self.q_laser = np.zeros_like(self.T)
        self.rho_blood, self.c_blood = 1060, 3770  # خون
    
    def _init_properties(self):
        self.rho = np.zeros_like(self.T)
        self.c = np.zeros_like(self.T)
        self.k = np.zeros_like(self.T)
        self.omega_b = np.zeros_like(self.T)
        self.Q_met = np.zeros_like(self.T)
        
        for layer in self.tissue_layers:
            z_min, z_max = layer.z_min/100, layer.z_max/100  # cm → m
            z_idx = np.where((self.z >= z_min) & (self.z < z_max))[0]
            
            # Get thermal properties based on tissue name
            tissue_name_lower = layer.name.lower() if layer.name else "muscle"
            # Try exact match first, then partial match
            props = None
            for key, prop in self.THERMAL_PROPERTIES.items():
                if key in tissue_name_lower:
                    props = prop
                    break
            
            # Default to muscle if not found
            if props is None:
                props = self.THERMAL_PROPERTIES["muscle"]
            
            for k in z_idx:
                self.rho[:, :, k] = props.rho
                self.c[:, :, k] = props.c
                self.k[:, :, k] = props.k
                self.omega_b[:, :, k] = props.omega_b
                self.Q_met[:, :, k] = props.Q_met
    
    def set_laser_heat_source_from_mc(self, mc_results, mc_simulator, power_scale=1.0):
        from scipy.interpolate import RegularGridInterpolator
        # تبدیل واحد: cm → m
        mc_x = mc_simulator.x_bins / 100
        mc_y = mc_simulator.y_bins / 100
        mc_z = mc_simulator.z_bins / 100
        interpolator = RegularGridInterpolator(
            (mc_x, mc_y, mc_z), mc_results['volumetric_heat_source'],
            method='linear', bounds_error=False, fill_value=0.0
        )
        X, Y, Z = np.meshgrid(self.x, self.y, self.z, indexing='ij')
        points = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1)
        self.q_laser = interpolator(points).reshape(self.nx, self.ny, self.nz) * power_scale
    
    def compute_time_step(self, safety=0.4):
        alpha = self.k / (self.rho * self.c)
        return safety * min(self.dx**2, self.dy**2, self.dz**2) / (6 * alpha.max())
    
    def solve_explicit(self, dt, total_time, save_interval=None):
        n_steps = int(total_time / dt)
        save_every = int((save_interval or total_time/10) / dt) or 1
        self.temp_history = [self.T.copy()]
        self.time_history = [0.0]
        wb_term = self.omega_b * self.rho_blood * self.c_blood
        
        for step in range(n_steps):
            T_old = self.T.copy()
            d2T_dx2 = np.zeros_like(self.T)
            d2T_dx2[1:-1, :, :] = (T_old[2:, :, :] - 2*T_old[1:-1, :, :] + T_old[:-2, :, :]) / self.dx**2
            d2T_dy2 = np.zeros_like(self.T)
            d2T_dy2[:, 1:-1, :] = (T_old[:, 2:, :] - 2*T_old[:, 1:-1, :] + T_old[:, :-2, :]) / self.dy**2
            d2T_dz2 = np.zeros_like(self.T)
            d2T_dz2[:, :, 1:-1] = (T_old[:, :, 2:] - 2*T_old[:, :, 1:-1] + T_old[:, :, :-2]) / self.dz**2
            
            dT_dt = (self.k * (d2T_dx2 + d2T_dy2 + d2T_dz2) +
                     wb_term * (self.T0 - T_old) +
                     self.Q_met + self.q_laser) / (self.rho * self.c)
            self.T = T_old + dt * dT_dt
            
            # شرط مرزی دیریکله
            self.T[0, :, :] = self.T0; self.T[-1, :, :] = self.T0
            self.T[:, 0, :] = self.T0; self.T[:, -1, :] = self.T0
            self.T[:, :, 0] = self.T0; self.T[:, :, -1] = self.T0
            
            if (step + 1) % save_every == 0:
                self.temp_history.append(self.T.copy())
                self.time_history.append((step + 1) * dt)
        
        return {
            'temperature': self.T,
            'temp_history': self.temp_history,
            'time_history': self.time_history,
            'max_temp': self.T.max()
        }