"""The entry listing: the I1 minimum is the deepest local minimum inside the declared
window with the others kept as alternatives and labeled, the edge is sought inside the
window only, the reversal is interpolated east of the minimum and interrupted by an
invalid cell, and a row outside the annotation is abstained."""
import importlib.util
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))


def test_the_native_longitude_grid_keeps_half_degree_columns_and_integer_rows(tmp_path):
    """Added for the second development package (2026-10-17): the files are on a
    half-degree grid and the frozen listing sampled integer degrees only, so a grid option
    must give the native longitude columns with integer latitude rows, and the default
    must stay the integer grid."""
    import netCDF4
    import numpy as np
    L = _load()
    p = tmp_path / "era5_v700_1999_6h_region.nc"
    d = netCDF4.Dataset(str(p), "w")
    d.createDimension("latitude", 5); d.createDimension("longitude", 7); d.createDimension("valid_time", 2)
    la = d.createVariable("latitude", "f8", ("latitude",)); la[:] = [12.0, 11.5, 11.0, 10.5, 10.0]
    lo = d.createVariable("longitude", "f8", ("longitude",)); lo[:] = [40.0, 40.5, 41.0, 41.5, 42.0, 42.5, 43.0]
    t = d.createVariable("valid_time", "f8", ("valid_time",)); t.units = "seconds since 1970-01-01"
    import datetime as dt
    t[:] = [(dt.datetime(1999, 7, 1) - dt.datetime(1970, 1, 1)).total_seconds(), (dt.datetime(1999, 7, 1, 6) - dt.datetime(1970, 1, 1)).total_seconds()]
    d.close()
    g_int = L.listing_grid(str(p), 1999, "integer")
    g_nat = L.listing_grid(str(p), 1999, "native-lon")
    assert list(g_int["longrid"][0, :]) == [40.0, 41.0, 42.0, 43.0] and list(g_int["latgrid"][:, 0]) == [12.0, 11.0, 10.0]
    assert list(g_nat["longrid"][0, :]) == [40.0, 40.5, 41.0, 41.5, 42.0, 42.5, 43.0] and list(g_nat["latgrid"][:, 0]) == [12.0, 11.0, 10.0]
    assert g_nat["lon_spacing_deg"] == 0.5 and g_int["lon_spacing_deg"] == 1.0


def _load():
    spec = importlib.util.spec_from_file_location("entry_listing_under_test", os.path.join(ROOT, "scripts", "pilot_i1_entry_listing.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ANN = {"i1": {"lon": [46, 56], "rows": [13, 28]}, "e": {"lon": [57, 63], "rows": [17, 23]}, "za": {"lon": [49, 62], "rows": [13, 28]}}
THR = {"primary": -8.0, "sensitivity": -6.0}


def test_the_i1_minimum_is_the_deepest_inside_the_window_and_the_stripe_is_labeled_w():
    L = _load()
    lons = np.arange(38.0, 66.0)
    v = np.full(lons.size, -2.0)
    v[lons == 41] = -12.0                       # the stripe, west of the window
    v[lons == 50] = -11.0; v[lons == 53] = -9.0 # two minima inside the window
    v[lons == 60] = -9.5                        # the second core
    v[lons >= 62] = 3.0                         # southerlies to the east
    valid = np.ones(lons.size, bool)
    zeta = np.zeros(lons.size); zeta[lons == 55] = 4e-5
    m = L.row_measurements(lons, v, valid, zeta, valid, ANN, 20, THR)
    assert m["status"] == "assigned" and m["i1"]["lon"] == 50.0 and m["i1"]["v"] == -11.0
    assert [x["lon"] for x in m["i1_alternatives"]] == [53.0]
    labels = {x["lon"]: x["label"] for x in m["wind_minima"]}
    assert labels[41.0] == "W" and labels[60.0] == "E"
    assert m["edge"]["primary"]["lon"] == 50.0 and m["edge"]["sensitivity"]["lon"] == 50.0, "the edge is sought inside the window, so the stripe at 41 E is not taken"
    assert m["za"]["lon"] == 55.0 and m["zs"] is None


def test_the_reversal_is_interpolated_east_of_the_minimum_and_interrupted_by_an_invalid_cell():
    L = _load()
    lons = np.arange(38.0, 66.0)
    v = np.full(lons.size, -4.0); v[lons == 50] = -10.0
    v[lons == 58] = -1.0; v[lons == 59] = 3.0  # crossing between 58 and 59 E at a quarter of the step
    valid = np.ones(lons.size, bool)
    zeta = np.zeros(lons.size)
    m = L.row_measurements(lons, v, valid, zeta, valid, ANN, 20, THR)
    assert m["reversal"]["status"] == "available" and abs(m["reversal"]["lon"] - 58.25) < 1e-9
    valid2 = valid.copy(); valid2[lons == 55] = False
    m2 = L.row_measurements(lons, v, valid2, zeta, valid2, ANN, 20, THR)
    assert m2["reversal"]["lon"] is None and m2["reversal"]["status"].startswith("interrupted")
    v3 = v.copy(); v3[lons >= 58] = -1.0
    m3 = L.row_measurements(lons, v3, valid, zeta, valid, ANN, 20, THR)
    assert m3["reversal"]["status"] == "none within the box"


def test_rows_outside_the_annotation_are_abstained_and_a_missing_minimum_is_reported():
    L = _load()
    lons = np.arange(38.0, 66.0)
    v = np.full(lons.size, -4.0); v[lons == 50] = -10.0
    valid = np.ones(lons.size, bool); zeta = np.zeros(lons.size)
    assert L.row_measurements(lons, v, valid, zeta, valid, ANN, 11, THR)["status"] == "abstained"
    assert L.row_measurements(lons, v, valid, zeta, valid, None, 20, THR)["status"] == "abstained"
    flat = np.full(lons.size, -4.0); flat[lons == 41] = -12.0
    assert L.row_measurements(lons, flat, valid, zeta, valid, ANN, 20, THR)["status"] == "no I1 minimum on the row"
