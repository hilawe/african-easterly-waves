import numpy as np

from aew.thermo import saturation_vapor_pressure, theta_e


def test_saturation_vapor_pressure_reference_points():
    # standard reference values: es(0 C) = 6.112 hPa, es(25 C) ~ 31.7 hPa
    np.testing.assert_allclose(saturation_vapor_pressure(273.15), 6.112, rtol=1e-6)
    np.testing.assert_allclose(saturation_vapor_pressure(298.15), 31.7, atol=0.2)


def test_theta_e_textbook_case():
    # T = 25 C, Td = 15 C (RH ~ 53.8%), p = 1000 hPa. Hand evaluation of Bolton (1980):
    # e = es(15 C) = 17.04 hPa, r = 10.78 g/kg, T_L (eq. 21) = 285.9 K, giving
    # theta-e = 298.15 exp(0.1008) = 329.8 K; independent implementations agree to ~0.2 K.
    rh = 100.0 * saturation_vapor_pressure(288.15) / saturation_vapor_pressure(298.15)
    te = theta_e(298.15, rh, 1000.0)
    assert abs(te - 329.8) < 0.5


def test_theta_e_reduces_to_dry_theta_when_dry():
    # with almost no vapor, theta-e approaches the dry potential temperature
    T, p = 280.0, 700.0
    theta_dry = T * (1000.0 / p) ** 0.2854
    te = theta_e(T, 0.1, p)
    assert abs(te - theta_dry) < 1.0


def test_theta_e_monotone_in_moisture_and_temperature():
    te_dryish = theta_e(285.0, 30.0, 700.0)
    te_moist = theta_e(285.0, 90.0, 700.0)
    assert te_moist > te_dryish
    assert theta_e(290.0, 50.0, 700.0) > theta_e(285.0, 50.0, 700.0)


def test_theta_e_vectorizes():
    T = np.array([280.0, 285.0, 290.0])
    rh = np.array([40.0, 60.0, 80.0])
    out = theta_e(T, rh, 700.0)
    assert out.shape == (3,)
    assert np.all(np.diff(out) > 0)
