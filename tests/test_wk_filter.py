import numpy as np
from aew.filtering import wk_bandpass


def _wave(nt, nlon, spd, period_days, wavenumber, westward):
    """cos wave: phase = 2pi(k*x/nlon + s*f*t), s=+1 westward, -1 eastward (our convention:
    westward <=> freq & raw-wavenumber opposite sign -> here we build x-index/time-index)."""
    t = np.arange(nt)
    x = np.arange(nlon)
    f = 1.0 / (period_days * spd)  # cycles per sample
    k = wavenumber
    sign = 1.0 if westward else -1.0
    # westward: x decreases as t increases -> phase k*x + (sign)*omega*t with sign matching
    phase = 2 * np.pi * (k * x[None, :] / nlon + sign * f * t[:, None])
    return np.cos(phase)


def test_passes_westward_in_band_removes_eastward():
    nt, nlon, spd = 480, 144, 4  # 120 days, 6-hourly, global
    west = _wave(nt, nlon, spd, period_days=5.0, wavenumber=6, westward=True)
    east = _wave(nt, nlon, spd, period_days=5.0, wavenumber=6, westward=False)
    field = west + east
    out = wk_bandpass(field, spd, 2.5, 10.0, -20, 0, time_axis=0, lon_axis=-1)
    # westward component is retained, eastward removed
    assert np.corrcoef(out.ravel(), west.ravel())[0, 1] > 0.95
    assert abs(np.corrcoef(out.ravel(), east.ravel())[0, 1]) < 0.2
    # amplitude of recovered ~ westward amplitude (1), not 2
    assert 0.8 < out.std() / west.std() < 1.2


def test_removes_out_of_period_band():
    nt, nlon, spd = 480, 144, 4
    inb = _wave(nt, nlon, spd, period_days=5.0, wavenumber=4, westward=True)
    slow = _wave(nt, nlon, spd, period_days=30.0, wavenumber=4, westward=True)  # too slow
    out = wk_bandpass(inb + slow, spd, 2.5, 10.0, -20, 0)
    assert np.corrcoef(out.ravel(), inb.ravel())[0, 1] > 0.95
    assert abs(np.corrcoef(out.ravel(), slow.ravel())[0, 1]) < 0.2


def test_removes_out_of_wavenumber_band():
    nt, nlon, spd = 480, 144, 4
    inb = _wave(nt, nlon, spd, period_days=5.0, wavenumber=6, westward=True)
    bigk = _wave(nt, nlon, spd, period_days=5.0, wavenumber=30, westward=True)  # |k|>20
    out = wk_bandpass(inb + bigk, spd, 2.5, 10.0, -20, 0)
    assert np.corrcoef(out.ravel(), inb.ravel())[0, 1] > 0.95
    assert abs(np.corrcoef(out.ravel(), bigk.ravel())[0, 1]) < 0.2


def test_real_output_shape():
    nt, nlon, spd = 240, 120, 4
    f = np.random.default_rng(0).standard_normal((nt, 3, nlon))  # (time, lat, lon)
    out = wk_bandpass(f, spd, 2.5, 10.0, -20, 0, time_axis=0, lon_axis=-1)
    assert out.shape == f.shape
    assert np.isrealobj(out)


def test_regional_longitude_guard():
    import pytest
    nt, nlon, spd = 240, 60, 4
    f = np.zeros((nt, nlon))
    lon_regional = np.linspace(-45, 75, nlon)   # 120 deg window, not global
    with pytest.raises(ValueError):
        wk_bandpass(f, spd, 2.5, 10.0, -20, 0, lon=lon_regional)
    # global grid passes; allow_regional bypasses
    lon_global = np.arange(0, 360, 360 / nlon)
    wk_bandpass(f, spd, 2.5, 10.0, -20, 0, lon=lon_global)
    wk_bandpass(f, spd, 2.5, 10.0, -20, 0, lon=lon_regional, allow_regional=True)
