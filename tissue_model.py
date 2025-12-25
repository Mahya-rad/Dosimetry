import numpy as np

def estimate_tissue_layers_from_BMI(BMI, KL_grade):
    """تخمین ضخامت لایه‌های بافت از روی BMI و درجه KL (به cm)"""
    skin = 0.2  # cm (ثابت)
    fat = max(0.2, 0.4 * BMI - 5) / 10  # mm → cm
    muscle = 0.7  # cm (ثابت)
    joint = 1.0  # cm (فضای مفصلی)
    cartilage = {
        1: 0.40,
        2: 0.37,
        3: 0.33,
        4: 0.25
    }.get(KL_grade, 0.33)
    return {
        'skin': skin,
        'fat': fat,
        'muscle': muscle,
        'joint': joint,
        'cartilage': cartilage
    }

def get_optical_params(wavelength_nm):
    """دریافت پارامترهای اپتیکی بر اساس طول موج (μₐ, μₛ, g, n)"""
    # مقادیر مبتنی بر Jacques (2013) + IT'IS
    lut = {
        810: {
            'skin': (0.15, 100, 0.90, 1.40),
            'fat': (0.05, 50, 0.90, 1.38),
            'muscle': (0.15, 80, 0.90, 1.37),
            'joint': (0.08, 60, 0.85, 1.35),
            'cartilage': (0.18, 90, 0.88, 1.38)
        },
        830: {
            'skin': (0.14, 95, 0.90, 1.40),
            'fat': (0.045, 48, 0.90, 1.38),
            'muscle': (0.14, 78, 0.90, 1.37),
            'joint': (0.075, 58, 0.85, 1.35),
            'cartilage': (0.17, 88, 0.88, 1.38)
        },
        904: {
            'skin': (0.12, 85, 0.90, 1.40),
            'fat': (0.04, 45, 0.90, 1.38),
            'muscle': (0.12, 72, 0.90, 1.37),
            'joint': (0.065, 54, 0.85, 1.35),
            'cartilage': (0.15, 82, 0.88, 1.38)
        }
    }
    return lut.get(wavelength_nm, lut[830])