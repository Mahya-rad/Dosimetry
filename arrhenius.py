# arrhenius.py
# Arrhenius thermal damage model for LLLT safety assessment

import numpy as np


class ArrheniusDamageCalculator:
    """
    Arrhenius thermal damage calculator for biological tissues.
    
    Computes cumulative thermal damage (Omega) based on temperature-time history
    using the Arrhenius rate process model:
    
    Omega(t) = ∫ A * exp(-Ea/(R*T(t'))) dt'
    
    where:
    - A: frequency factor (pre-exponential constant)
    - Ea: activation energy (J/mol)
    - R: universal gas constant (8.314 J/(mol·K))
    - T: absolute temperature (Kelvin)
    """
    
    # Arrhenius parameters for different tissue types
    # Based on literature values for thermal damage modeling
    TISSUE_PARAMETERS = {
        "skin": {
            "A": 3.1e98,  # 1/s
            "Ea": 6.28e5,  # J/mol
        },
        "cartilage": {
            "A": 5.6e63,  # 1/s
            "Ea": 4.3e5,  # J/mol
        },
        "muscle": {
            "A": 7.39e39,  # 1/s
            "Ea": 2.577e5,  # J/mol
        },
        "fat": {
            "A": 5.6e63,  # 1/s
            "Ea": 4.3e5,  # J/mol (similar to cartilage)
        },
        "default": {
            "A": 3.1e98,  # 1/s (skin as default)
            "Ea": 6.28e5,  # J/mol
        }
    }
    
    R = 8.314  # Universal gas constant (J/(mol·K))
    
    def __init__(self):
        """Initialize the Arrhenius damage calculator."""
        pass
    
    def get_parameters(self, tissue_type="default"):
        """
        Get Arrhenius parameters for a specific tissue type.
        
        Args:
            tissue_type: String identifier for tissue type
                ("skin", "cartilage", "muscle", "fat", or "default")
        
        Returns:
            dict: Dictionary with 'A' (frequency factor) and 'Ea' (activation energy)
        """
        tissue_type_lower = tissue_type.lower() if tissue_type else "default"
        return self.TISSUE_PARAMETERS.get(
            tissue_type_lower,
            self.TISSUE_PARAMETERS["default"]
        )
    
    def damage_rate(self, T_C, tissue_type="default"):
        """
        Calculate instantaneous damage rate at a given temperature.
        
        Args:
            T_C: Temperature in Celsius
            tissue_type: Tissue type identifier
        
        Returns:
            float: Damage rate (1/s)
        """
        params = self.get_parameters(tissue_type)
        A = params["A"]
        Ea = params["Ea"]
        
        # Convert Celsius to Kelvin
        T_K = T_C + 273.15
        
        # Avoid division by zero or negative temperatures
        if T_K <= 0:
            return 0.0
        
        # Calculate Arrhenius rate: A * exp(-Ea/(R*T))
        try:
            rate = A * np.exp(-Ea / (self.R * T_K))
        except (OverflowError, ZeroDivisionError):
            rate = 0.0
        
        return rate
    
    def integrate_damage(self, T_history, time_history, tissue_type="default"):
        """
        Integrate Arrhenius damage over time for a given temperature history.
        
        Args:
            T_history: Array of temperatures in Celsius
            time_history: Array of corresponding times in seconds
            tissue_type: Tissue type identifier
        
        Returns:
            numpy.ndarray: Cumulative damage (Omega) at each time point
        """
        T_history = np.asarray(T_history)
        time_history = np.asarray(time_history)
        
        if len(T_history) != len(time_history):
            raise ValueError(
                "Temperature and time history arrays must have the same length"
            )
        
        if len(time_history) < 2:
            return np.array([0.0])
        
        # Get tissue parameters
        params = self.get_parameters(tissue_type)
        A = params["A"]
        Ea = params["Ea"]
        
        # Convert temperatures to Kelvin
        T_K = T_history + 273.15
        
        # Initialize damage array
        omega = np.zeros_like(T_history)
        
        # Trapezoidal integration of damage rate
        for i in range(1, len(time_history)):
            dt = time_history[i] - time_history[i-1]
            
            # Average temperature in this interval
            T_avg = (T_K[i] + T_K[i-1]) / 2.0
            
            if T_avg > 0:
                try:
                    # Damage rate at average temperature
                    rate = A * np.exp(-Ea / (self.R * T_avg))
                except (OverflowError, ZeroDivisionError):
                    rate = 0.0
            else:
                rate = 0.0
            
            # Cumulative damage (integral of rate over time)
            omega[i] = omega[i-1] + rate * dt
        
        return omega
    
    def is_safe(self, omega, threshold=1.0):
        """
        Determine if thermal damage is within safe limits.
        
        Args:
            omega: Cumulative damage value
            threshold: Damage threshold (default 1.0 for 63% damage)
        
        Returns:
            bool: True if damage is below threshold
        """
        return omega < threshold

