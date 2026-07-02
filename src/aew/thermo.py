"""Moist thermodynamics for the trajectory diagnostics.

Equivalent potential temperature (theta-e) following Bolton (1980, MWR 108, 1046-1053),
the standard operational formulation (accurate to ~0.4 K against exact pseudoadiabatic
calculations). Theta-e is conserved in both dry and pseudoadiabatic displacements, so a
theta-e minimum along a 700 hPa trajectory marks a genuinely distinct dry airmass (the
Saharan Air Layer signature) rather than a temperature fluctuation, which plain relative
humidity cannot distinguish.
"""

from __future__ import annotations

import numpy as np

__all__ = ["saturation_vapor_pressure", "theta_e"]


def saturation_vapor_pressure(T):
    """Saturation vapor pressure over water in hPa (Bolton eq. 10), T in K."""
    Tc = np.asarray(T, dtype=float) - 273.15
    return 6.112 * np.exp(17.67 * Tc / (Tc + 243.5))


def theta_e(T, rh, p_hpa):
    """Equivalent potential temperature (K) from T (K), RH (%), pressure (hPa).

    Bolton (1980): vapor pressure e = (RH/100) es(T); mixing ratio
    r = 622 e / (p - e) in g/kg; the temperature at the lifting condensation level
    (eq. 21) T_L = 2840 / (3.5 ln T - ln e - 4.805) + 55; and theta-e (eq. 43)
    theta_e = T (1000/p)^(0.2854 (1 - 0.28e-3 r)) exp[(3.376/T_L - 0.00254) r
    (1 + 0.81e-3 r)]. RH is clipped to a small positive floor (the LCL temperature is
    undefined for exactly zero vapor).
    """
    T = np.asarray(T, dtype=float)
    rh = np.clip(np.asarray(rh, dtype=float), 0.1, None)
    p_hpa = np.asarray(p_hpa, dtype=float)
    e = (rh / 100.0) * saturation_vapor_pressure(T)          # hPa
    r = 622.0 * e / (p_hpa - e)                              # g/kg
    T_L = 2840.0 / (3.5 * np.log(T) - np.log(e) - 4.805) + 55.0
    exponent = 0.2854 * (1.0 - 0.28e-3 * r)
    return (T * (1000.0 / p_hpa) ** exponent
            * np.exp((3.376 / T_L - 0.00254) * r * (1.0 + 0.81e-3 * r)))
