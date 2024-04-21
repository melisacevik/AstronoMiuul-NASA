import numpy as np
import pandas as pd

def calculate_orbital_velocity(a, GK=1.32712440018e11):
    """
    Yörüngesel hızı hesaplar.
    a: Yarı-büyük eksen (AU cinsinden)
    GM: Güneş Sistemi'nin standart kütleçekim parametresi (km^3/s^2 cinsinden)
    """
    # AU'yu km'ye çevir
    a_km = a * 149597870.7  # 1 AU = 149597870.7 km
    # Yörüngesel hız formülü V = sqrt(GM/a)
    velocity = np.sqrt(GK / a_km)  # km/s olarak
    return velocity

