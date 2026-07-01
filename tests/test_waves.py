import numpy as np
import pandas as pd
from aew.waves import (curvature_vorticity, relative_vorticity, trough_points,
                       anomaly_from_timemean)


def test_curvature_zero_for_zonal_flow():
    lat = np.arange(0.0, 25.1, 1.0); lon = np.arange(-40.0, 40.1, 1.0)
    u = -6.0 + np.zeros((lat.size, lon.size))
    v = np.zeros((lat.size, lon.size))
    zc = curvature_vorticity(u, v, lat, lon)
    assert np.nanmax(np.abs(zc)) < 1e-12


def test_curvature_is_half_relative_for_solid_rotation():
    # solid-body rotation: curvature vorticity = Omega, relative vorticity = 2 Omega,
    # so curvature is half of relative (a clean physics check).
    lat = np.arange(5.0, 16.0, 0.5); lon = np.arange(-5.0, 5.1, 0.5)
    LON, LAT = np.meshgrid(lon, lat)
    R = 6.371e6
    x = np.deg2rad(LON) * R * np.cos(np.deg2rad(10.0))
    y = np.deg2rad(LAT - 10.0) * R
    om = 1e-5
    u = -om * y; v = om * x
    zc = curvature_vorticity(u, v, lat, lon)[3:-3, 3:-3]
    zr = relative_vorticity(u, v, lat, lon)[3:-3, 3:-3]
    assert abs(np.nanmedian(zc) / np.nanmedian(zr) - 0.5) < 0.1


def test_trough_points_find_the_trough_axis():
    # v = +sin(k lon) with easterly base flow: curvature vorticity is cyclonic (a trough)
    # at lon = 0. One wavelength across the window so there is a single trough.
    lat = np.arange(0.0, 25.1, 1.0); lon = np.arange(-15.0, 15.1, 1.0)
    t = pd.date_range("2000-07-01", periods=3, freq="6h").values
    v2 = np.sin(2 * np.pi * lon / 30.0)
    v = np.broadcast_to(v2, (t.size, lat.size, lon.size)).copy()
    u = -4.0 + np.zeros_like(v)
    zc = curvature_vorticity(u, v, lat, lon)
    tt, la, lo = trough_points(zc, u, t, lat, lon, u_max=2.5, lat_range=(5, 15),
                               min_anom=0.3 * np.nanmax(zc))
    assert lo.size > 0
    assert abs(np.median(lo)) <= 3.0


def test_trough_points_respect_umax():
    lat = np.arange(0.0, 25.1, 1.0); lon = np.arange(-15.0, 15.1, 1.0)
    t = pd.date_range("2000-07-01", periods=2, freq="6h").values
    v = np.broadcast_to(np.sin(2 * np.pi * lon / 30.0), (t.size, lat.size, lon.size)).copy()
    u = 10.0 + np.zeros_like(v)  # strong westerly -> no trough points
    zc = curvature_vorticity(u, v, lat, lon)
    tt, la, lo = trough_points(zc, u, t, lat, lon, u_max=2.5, min_anom=0.3 * np.nanmax(zc))
    assert lo.size == 0


def test_trough_axis_subgrid_recovers_offgrid_longitude():
    # peak placed between grid points at lon=2.5; sub-grid interpolation should recover it
    lat = np.arange(5.0, 16.0, 1.0); lon = np.arange(-20.0, 20.1, 1.0)
    t = np.array([np.datetime64("2000-07-01T00")])
    from aew.waves import trough_axis
    ca = np.exp(-((lon - 2.5) ** 2) / (2 * 6.0 ** 2))
    ca = np.broadcast_to(ca, (1, lat.size, lon.size)).copy()
    u = -4.0 + np.zeros_like(ca)
    tt, la, lo, amp = trough_axis(ca, u, t, lat, lon, min_anom=0.3, u_max=2.5,
                                  smooth_lat=1, smooth_lon=1, lat_range=(5, 15))
    assert lo.size > 0
    assert abs(np.median(lo) - 2.5) < 0.6  # sub-grid, closer than the 1 deg spacing


def test_coherence_filter_drops_isolated_trough():
    from aew.waves import coherence_filter
    t = pd.to_datetime(["2000-07-01T00", "2000-07-01T06", "2000-07-01T06"]).values
    # two matching troughs at consecutive times near lon 0, plus one isolated far away
    lons = np.array([0.0, 1.0, 100.0]); lats = np.array([10.0, 10.0, 10.0])
    keep = coherence_filter(t, lats, lons, max_dlon=8.0, max_dlat=3.0)
    assert keep.tolist() == [True, True, False]
