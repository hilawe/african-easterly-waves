"""The preparation audit's synthetic checks: the vorticity decomposition on solid-body
rotation and straight shear, the bridge between the two relative-vorticity definitions,
the production decimation's 2 by 2 footprint and offset, the southern sign, the support
carried through each footprint, and the signed shear categories."""
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "src"))
import pilot_eastern_preparation_audit as PA  # noqa: E402
import pilot_field_sequence as FS  # noqa: E402
from aew.v1port import climatology as clim  # noqa: E402
from aew.v1port import vorticity as VO  # noqa: E402
from aew.v1port.detection import _prepare  # noqa: E402

OMEGA = 1.0e-5
LAT = np.arange(-10.0, 10.5, 1.0)                                  # ascending, as the decomposition expects
LON = np.arange(20.0, 41.0, 1.0)
LATG, LONG = np.meshgrid(LAT, LON, indexing="ij")
X = VO.EARTH_RADIUS_M * np.cos(np.deg2rad(LATG)) * np.deg2rad(LONG - 30.0)
Y = VO.EARTH_RADIUS_M * np.deg2rad(LATG)
FAR = np.hypot(LATG, LONG - 30.0) >= 3.0


def test_solid_body_rotation_splits_evenly_into_curvature_and_shear_away_from_the_calm_center():
    r, s, c = PA.components_oriented(LATG, LONG, -OMEGA * Y, OMEGA * X)
    ok = FAR & np.isfinite(r)
    assert ok.sum() > 100
    np.testing.assert_allclose(r[ok], 2 * OMEGA, rtol=1e-9)
    np.testing.assert_allclose(c[ok], OMEGA, rtol=0.05)
    np.testing.assert_allclose(s[ok], OMEGA, rtol=0.05)


def test_straight_shear_is_all_shear_away_from_the_calm_line():
    alpha = 2.0e-5
    r, s, c = PA.components_oriented(LATG, LONG, -alpha * Y, np.zeros_like(Y))
    ok = (np.abs(LATG) >= 3.0) & np.isfinite(r)
    np.testing.assert_allclose(r[ok], alpha, rtol=1e-9)
    np.testing.assert_allclose(s[ok], alpha, rtol=0.05)
    assert np.all(np.abs(c[ok]) <= 0.05 * alpha)


def test_the_bridge_a_uniform_zonal_flow_gives_only_the_metric_term():
    lat, lon = np.arange(18.0, 11.5, -1.0), np.arange(30.0, 37.0, 1.0)
    latg, long_ = np.meshgrid(lat, lon, indexing="ij")
    u, v = np.full(latg.shape, -8.0), np.zeros(latg.shape)
    z0, cls = FS.relative_vorticity(u, v, lat, lon, np.full(latg.shape, "valid", dtype=object))
    r, _, _ = PA.components_oriented(latg, long_, u, v)
    i = int(np.flatnonzero(lat == 15.0)[0])
    assert math.isclose(z0[i, 3], -8.0 * math.tan(math.radians(15.0)) / FS.EARTH_RADIUS_M, rel_tol=1e-12)
    assert abs(r[i, 3]) < 1e-20


def test_the_decimation_puts_a_native_cell_into_one_coarse_cell_centered_half_a_cell_on():
    f = np.zeros((12, 14))
    for R, C in ((5, 7), (6, 8)):
        f[:] = 0.0
        f[R, C] = 1.0
        g = clim.gaussian_decimate(f, 1.0, 2.5)
        unit = clim.gaussian_decimate(f, 1.0, 1.0)
        assert np.array_equal(unit[::2, ::2], g)
        assert g[R // 2, C // 2] == 0.25 and np.count_nonzero(g) == 1
        rows, cols = np.nonzero(unit)
        assert sorted(set(rows)) == [R - 1, R] and sorted(set(cols)) == [C - 1, C]    # the value at (r, c) holds rows r, r+1 and columns c, c+1
    ok = np.ones((12, 14), bool); ok[5, 7] = False
    bad = ~PA.block_valid(ok)
    assert {(int(a), int(b)) for a, b in zip(*np.nonzero(bad[:-1, :-1]))} == {(4, 6), (4, 7), (5, 6), (5, 7)}
    assert bad[-1, :].all() and bad[:, -1].all()


def test_the_southern_sign_and_the_nine_point_footprint():
    lat = np.array([2.0, 1.0, 0.0, -1.0, -2.0])
    out = _prepare(np.full((5, 4), 3.0), lat)
    assert np.all(out[lat > 0] > 0) and np.all(out[lat < 0] < 0)
    ok = np.ones((6, 6), bool); ok[2, 3] = False
    v = PA.nine_valid(ok)
    assert not v[1:4, 2:5].any() and v[4, 1] and v[0, 3]                   # interior neighbors lose support, the outer ring keeps its own
    s = PA.stencil_valid(ok)
    assert not s[2, 3] and not s[1, 3] and not s[2, 2] and s[1, 2]


def test_signed_categories_keep_cancellation_and_weak_wind_apart():
    assert PA.shear_category(3.0, 1.0, 5.0) == "shear larger, same sign"
    assert PA.shear_category(1.0, 1.0, 5.0) == "curvature larger, same sign"
    assert PA.shear_category(3.0, -1.0, 5.0) == "cancelling, shear positive"
    assert PA.shear_category(-1.0, 3.0, 5.0) == "cancelling, curvature positive"
    assert PA.shear_category(3.0, 1.0, 0.5) == "weak wind"
    assert PA.shear_category(float("nan"), 1.0, 5.0) == "unresolved"


def test_a_north_first_grid_is_oriented_as_the_pipeline_orients_it():
    from aew.v1port import pipeline as PL
    u, v = -OMEGA * Y + 3.0 * np.sin(np.deg2rad(LONG)), OMEGA * X
    r_up, s_up, c_up = PA.components_oriented(LATG, LONG, u, v)
    r, s, c = PA.components_oriented(LATG[::-1], LONG[::-1], u[::-1], v[::-1])
    for got, want in ((r, r_up), (s, s_up), (c, c_up)):
        np.testing.assert_array_equal(got, want[::-1])
    np.testing.assert_array_equal(c, PL.curvature_from_winds(LATG[::-1], LONG[::-1], u[::-1][None], v[::-1][None])[0])
