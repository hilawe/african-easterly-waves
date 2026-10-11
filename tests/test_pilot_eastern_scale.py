"""The scale comparison's footprint, support and assessability rules. The footprint mean
is checked against the installed QTrack package itself on a synthetic field, at every
center the package computes, and beyond that boundary only where the whole footprint
fits."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import pilot_eastern_scale as SC  # noqa: E402


def _package_cv(u, v, lat, lon):
    """The package's curvature vorticity formula, used only as a test input field."""
    LON, LAT = np.meshgrid(lon, lat)
    lat_met = 6371 * 2 * np.pi * 1000 / 360
    dy = np.gradient(LAT, axis=0) * lat_met
    dx = np.gradient(LON, axis=1) * np.cos(np.deg2rad(LAT)) * lat_met
    V2 = u ** 2 + v ** 2
    return (1 / V2) * (u * u * np.gradient(v, axis=1) / dx - v * v * np.gradient(u, axis=0) / dy - v * u * np.gradient(u, axis=1) / dx + u * v * np.gradient(v, axis=0) / dy)


def test_the_footprint_mean_equals_the_installed_package_where_the_package_computes(tmp_path, monkeypatch):
    netCDF4 = pytest.importorskip("netCDF4")
    core = pytest.importorskip("qtrack.core")
    lat = np.arange(39.0, -1.0, -1.0)                     # 40 rows running south, as ERA5
    lon = np.arange(0.0, 60.0, 1.0)                       # 60 columns
    LON, LAT = np.meshgrid(lon, lat)
    u = -8.0 + 3.0 * np.sin(LON / 7.0) * np.cos(LAT / 9.0)
    v = 2.0 * np.sin((LON - 20.0) / 5.0) * np.exp(-((LAT - 15.0) / 8.0) ** 2) + 0.5
    src = tmp_path / "in.nc"
    d = netCDF4.Dataset(src, "w")
    d.createDimension("time", 1); d.createDimension("latitude", lat.size); d.createDimension("longitude", lon.size)
    t = d.createVariable("time", "f8", ("time",)); t.units = "hours since 1900-01-01 00:00:00"; t[:] = [0.0]
    d.createVariable("latitude", "f8", ("latitude",))[:] = lat
    d.createVariable("longitude", "f8", ("longitude",))[:] = lon
    d.createVariable("u", "f8", ("time", "latitude", "longitude"))[:] = u[None]
    d.createVariable("v", "f8", ("time", "latitude", "longitude"))[:] = v[None]
    d.close()
    monkeypatch.chdir(tmp_path)
    core.COMPUTE_CURV_VORT_NON_DIV_UPDATE(str(src), str(tmp_path / "out.nc"), 1, 600, 1, nondiv=False, SAVE_IMAGE=False, SAVE_OUTPUT=True)
    pkg = np.asarray(netCDF4.Dataset(tmp_path / "out.nc")["curv_vort"][0], float)
    cv = _package_cv(u, v, lat, lon)
    mine = SC.radial_mean(cv, np.ones(cv.shape, bool), lat, lon, 600.0)
    ib, jb = np.arange(7, lat.size - 6), np.arange(12, lon.size - 11)            # the package's loop boundary
    np.testing.assert_allclose(mine["complete_mean"][np.ix_(ib, jb)], pkg[np.ix_(ib, jb)], rtol=1e-12, atol=0)
    assert np.all(pkg[:, -11:] == 0.0)                                          # the package leaves these at zero
    assert np.isfinite(mine["complete_mean"][20, 52]) and np.isnan(mine["complete_mean"][20, 56])   # beyond its boundary only where the footprint fits


def test_complete_and_partial_means_are_kept_apart_and_uncomputed_values_are_missing():
    lat = np.arange(30.0, -1.0, -1.0); lon = np.arange(20.0, 61.0, 1.0)
    f = np.ones((lat.size, lon.size))
    valid = np.ones(f.shape, bool)
    valid[15, 20] = False                                                       # one invalid cell at 15 N 40 E
    c = SC.radial_mean(f, valid, lat, lon, 600.0)
    assert not c["complete"][15, 20] and c["valid_fraction"][15, 22] < 1.0
    assert np.isnan(c["complete_mean"][15, 22]) and c["partial_mean"][15, 22] == 1.0
    assert c["complete"][15, 30] and c["complete_mean"][15, 30] == 1.0 and np.isnan(c["partial_mean"][15, 30])
    assert not c["inside"][0, 20] and np.isnan(c["complete_mean"][0, 20]) and np.isnan(c["partial_mean"][0, 20])


def test_peak_support_needs_all_eight_neighbors_and_unsupported_points_are_not_counted():
    common = np.ones((6, 6), bool)
    common[2, 3] = False
    ok = SC.peak_support(common)
    assert not ok[2, 2] and not ok[3, 4] and bool(ok[4, 1])
    assert not ok[0, 0]                                                         # the edge has missing neighbors
    lat1, lon1 = np.array([10.0, 11.0, 12.0]), np.array([40.0, 41.0, 42.0])
    got, tot = SC.support_share([(10.0, 40.0), (11.0, 41.0), (50.0, 41.0)], np.ones((3, 3), bool), lat1, lon1)
    assert (got, tot) == (2, 3)                                                 # a point outside the subgrid is unsupported, not filled


def test_an_outcome_is_a_hit_a_miss_only_when_fully_assessable_and_otherwise_not_assessable():
    m = {"axis_lon": 45.0, "lat_range": [10.0, 20.0]}
    near = [{"lat": 15.0, "lon": 47.0, "value": 2.5}]
    far = [{"lat": 15.0, "lon": 55.0, "value": 3.0}]
    assert SC.outcome(near, m, 3.0, False)[0] == "hit" and SC.outcome(near, m, 3.0, False)[1]["distance_deg"] == 2.0
    assert SC.outcome(far, m, 3.0, True)[0] == "miss"
    assert SC.outcome(far, m, 3.0, False)[0] == "not assessable" and SC.outcome([], m, 3.0, False)[0] == "not assessable"


def test_gains_and_losses_count_only_maps_where_both_outcomes_are_determined():
    def row(d, b, c, f):
        return {"date": d, "segment": "S", "B": {"3.0": b, "5.0": b, "in_case_box": []}, "C": {"3.0": c, "5.0": c, "in_case_box": []},
                "F": {"3.0": f, "5.0": f, "in_case_box": 0}, "C_partial_in_case_box": []}
    rows = [row("d1", "miss", "hit", "miss"), row("d2", "hit", "miss", "hit"), row("d3", "hit", "not assessable", "hit"), row("d4", "hit", "hit", "miss")]
    s = SC.summarize(rows, [], [3.0])["3.0"]
    assert s["C_against_B"] == {"maps_determined": 3, "gained": [("d1", "S")], "lost": [("d2", "S")]}
    assert s["C_against_F"]["gained"] == [("d1", "S"), ("d4", "S")] and s["C_against_F"]["lost"] == [("d2", "S")]
    assert s["counts"]["C"] == {"hit": 2, "miss": 1, "not assessable": 1}
