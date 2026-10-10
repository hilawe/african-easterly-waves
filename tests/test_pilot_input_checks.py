"""The eastern pilot's two input checks: the shared-column comparison of a wider retrieval
with the production file, year by year, and the climatology comparison over a predeclared
longitude range with the boundary columns reported apart."""
import importlib.util
import os

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
netCDF4 = pytest.importorskip("netCDF4")


def _load(name):
    spec = importlib.util.spec_from_file_location(name + "_under_test", os.path.join(ROOT, "scripts", name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write(path, lon, values, times=(0, 6), expver=("0001",), lat=(2.0, 1.0, 0.0)):
    with netCDF4.Dataset(path, "w") as ds:
        ds.createDimension("valid_time", len(times)); ds.createDimension("latitude", len(lat)); ds.createDimension("longitude", len(lon))
        ds.createVariable("valid_time", "i8", ("valid_time",))[:] = np.array(times)
        ds.createVariable("latitude", "f8", ("latitude",))[:] = np.array(lat)
        ds.createVariable("longitude", "f8", ("longitude",))[:] = np.array(lon)
        ds.createVariable("expver", str, ("valid_time",))[:] = np.array([expver[0]] * len(times), dtype=object)
        ds.createVariable("u", "f4", ("valid_time", "latitude", "longitude"))[:] = values


def _pair(tmp_path, year, lon_a, lon_b, values_b, edit=None):
    a, b = tmp_path / "production", tmp_path / "wider"
    a.mkdir(parents=True, exist_ok=True); b.mkdir(parents=True, exist_ok=True)
    cols = [list(lon_b).index(x) for x in lon_a]
    kw = edit or {}
    for var in ("u700", "v700"):                                           # the check wants both variables of a year
        _write(str(a / f"era5_{var}_{year}_6h_region.nc"), lon_a, values_b[..., cols])
        _write(str(b / f"era5_{var}_{year}_6h_region.nc"), lon_b, kw.get("values", values_b), times=kw.get("times", (0, 6)),
               expver=kw.get("expver", ("0001",)), lat=kw.get("lat", (2.0, 1.0, 0.0)))
    return str(a), str(b)


def test_shared_columns_pass_only_when_everything_shared_is_identical(tmp_path):
    C = _load("check_shared_columns")
    lon_a, lon_b = np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    rng = np.random.default_rng(0)
    vals = rng.normal(size=(2, 3, 5)).astype("f4")
    a, b = _pair(tmp_path, 1990, lon_a, lon_b, vals)
    r = C.check(a, b, (1990, 1990))
    f = r["files"]["era5_u700_1990_6h_region.nc"]
    assert r["passed"] and f["problems"] == [] and f["shared_columns"] == 3 and f["new_columns"] == 2 and f["east_edge_wider"] == 4.0
    assert f["expver_production"] == ["0001"] == f["expver_wider"]
    changed = vals.copy(); changed[1, 2, 1] += 1e-6                       # one shared cell, one bit
    a, b = _pair(tmp_path / "v", 1990, lon_a, lon_b, changed)
    _write(os.path.join(a, "era5_u700_1990_6h_region.nc"), lon_a, vals[..., :3])
    r = C.check(a, b, (1990, 1990))
    assert not r["passed"] and "values differ on the shared columns, 1 cells" in r["files"]["era5_u700_1990_6h_region.nc"]["problems"][0]
    a, b = _pair(tmp_path / "t", 1990, lon_a, lon_b, vals, edit={"times": (0, 12)})
    assert "timestamps differ" in C.check(a, b, (1990, 1990))["files"]["era5_u700_1990_6h_region.nc"]["problems"][0]
    a, b = _pair(tmp_path / "e", 1990, lon_a, lon_b, vals, edit={"expver": ("0005",)})
    assert any("experiment versions differ" in p for p in C.check(a, b, (1990, 1990))["files"]["era5_u700_1990_6h_region.nc"]["problems"])
    a, b = _pair(tmp_path / "l", 1990, lon_a, lon_b, vals, edit={"lat": (2.5, 1.0, 0.0)})
    assert "latitudes differ" in C.check(a, b, (1990, 1990))["files"]["era5_u700_1990_6h_region.nc"]["problems"]
    r = C.check(a, b, (1990, 1991))                                        # 1991 is absent on both sides
    assert not r["passed"] and r["files"]["era5_u700_1991_6h_region.nc"]["problems"][0].startswith("missing")
    assert r["files"]["era5_v700_1991_6h_region.nc"]["problems"][0].startswith("missing")


def _cache(path, mean, counts=None):
    keys = np.array([[1, 1, 0], [1, 1, 6]])
    np.savez(path, years=np.array([1990, 1991]), fingerprint=np.asarray("f"), keys=keys,
             mean=mean, counts=counts if counts is not None else np.array([2, 2]))


def test_climatology_caches_are_gated_inside_the_range_and_reported_outside_it(tmp_path):
    M = _load("compare_climatology_caches")
    lon_a, lon_b = np.array([-3.0, -2.0, -1.0, 0.0, 1.0]), np.array([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
    rng = np.random.default_rng(1)
    mb = rng.normal(size=(2, 3, 7))
    ma = mb[..., :5].copy()
    ma[..., 4] = np.nan                                                    # the old grid's boundary column is masked
    _cache(tmp_path / "a.npz", ma); _cache(tmp_path / "b.npz", mb)
    r = M.compare(str(tmp_path / "a.npz"), lon_a, str(tmp_path / "b.npz"), lon_b, (-3.0, 0.0))
    assert r["passed"] and r["problems"] == [] and r["cells_compared"] == 4 * 6
    assert r["boundary"] == {"1": {"mask_cells_differing": 6, "value_cells_differing": 0}}   # reported, not gated
    # a changed value inside the range fails, and a changed mask inside the range fails
    mb2 = mb.copy(); mb2[0, 1, 2] += 1e-9
    _cache(tmp_path / "b2.npz", mb2)
    r = M.compare(str(tmp_path / "a.npz"), lon_a, str(tmp_path / "b2.npz"), lon_b, (-3.0, 0.0))
    assert not r["passed"] and "values differ inside the range on 1 cells" in r["problems"][0]
    mb3 = mb.copy(); mb3[1, 0, 0] = np.nan
    _cache(tmp_path / "b3.npz", mb3)
    r = M.compare(str(tmp_path / "a.npz"), lon_a, str(tmp_path / "b3.npz"), lon_b, (-3.0, 0.0))
    assert not r["passed"] and "masks differ inside the range on 1 cells" in r["problems"][0]
    # widening the declared range to the boundary column turns the expected difference into a failure
    r = M.compare(str(tmp_path / "a.npz"), lon_a, str(tmp_path / "b.npz"), lon_b, (-3.0, 1.0))
    assert not r["passed"] and "masks differ inside the range on 6 cells" in r["problems"][0]
    # different counts or keys fail regardless of the range
    _cache(tmp_path / "c.npz", mb, counts=np.array([2, 3]))
    assert "counts differ" in M.compare(str(tmp_path / "a.npz"), lon_a, str(tmp_path / "c.npz"), lon_b, (-3.0, 0.0))["problems"]


def test_the_longitudes_come_from_the_input_directory(tmp_path):
    M = _load("compare_climatology_caches")
    d = tmp_path / "inputs"; d.mkdir()
    _write(str(d / "era5_u700_1990_6h_region.nc"), np.array([5.0, 6.0]), np.zeros((2, 3, 2), "f4"))
    assert list(M.longitudes(str(d))) == [5.0, 6.0]
    with pytest.raises(SystemExit, match="no input file"):
        M.longitudes(str(tmp_path / "empty"))
