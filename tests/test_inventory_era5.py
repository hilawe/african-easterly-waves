"""The ERA5 on-disk inventory, bound on synthetic files with a planted NaN and a planted
masked cell.

MUTATION LIST, written before the assertions:
  E1 masked cells are not counted (only NaN);
  E2 NaN cells are not counted (only masked);
  E3 the time axis's unique steps are not recorded, so a gap in a file is invisible;
  E4 an unreadable file is skipped silently instead of recorded as an error.
Added after a review found a hand-typed coverage table wrong on extent, years and
variables, so the summary is now derived from the files:
  E5 the pressure level is not read, so u at 700 and u at 850 collapse into one field;
  E6 domain coverage is judged from the first file instead of the extent over all files;
  E7 missing years are counted against one target period only;
  E8 a tree lacking v while holding u is reported as having the needed fields.
Added after a review produced two trees the first summary called usable, one with two
observations per year and one with u on the northern half and v on the southern half:
  E9 a year counts as complete without four steps per calendar day;
  E10 the two components are not compared grid to grid, so split halves pass as one
      grid;
  E11 a readiness verdict is emitted.
Added after a review showed equal endpoints passing as one grid:
  E12 grids with the same endpoints and different resolution count as one grid;
  E13 grids that agree on endpoints, count and first spacing but differ at an interior
      coordinate count as one grid;
  E14 a reversed axis counts as a different grid.
"""
import importlib.util
import json
import os
import sys

import numpy as np
import pytest

nc = pytest.importorskip("netCDF4")

HERE = os.path.dirname(os.path.abspath(__file__))
NAN = np.nan


def _load():
    spec = importlib.util.spec_from_file_location(
        "inventory_era5", os.path.join(HERE, "..", "scripts", "inventory_era5.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["inventory_era5"] = mod
    spec.loader.exec_module(mod)
    return mod


def write_field(path, hours, lats, lons, values, fill=None, var="u", level=700.0):
    d = nc.Dataset(path, "w")
    d.createDimension("pressure_level", 1)
    d.createVariable("pressure_level", "f8", ("pressure_level",))[:] = [level]
    d.createDimension("time", len(hours))
    d.createDimension("latitude", len(lats))
    d.createDimension("longitude", len(lons))
    t = d.createVariable("time", "f8", ("time",))
    t.units = "hours since 1900-01-01"
    t[:] = hours
    d.createVariable("latitude", "f8", ("latitude",))[:] = lats
    d.createVariable("longitude", "f8", ("longitude",))[:] = lons
    v = d.createVariable(var, "f4", ("time", "latitude", "longitude"), fill_value=fill)
    v[:] = values
    d.close()


def test_inventory_records_grid_steps_missing_cells_and_unreadable_files(tmp_path):
    M = _load()
    root = tmp_path / "era5"
    (root / "tree").mkdir(parents=True)
    hours = [0.0, 6.0, 12.0, 24.0]                      # a six-hour gap after 12
    lats = [10.0, 9.5, 9.0]
    lons = [-20.0, -19.5]
    vals = np.ones((4, 3, 2), dtype=np.float32)
    vals[0, 0, 0] = NAN                                  # one NaN
    write_field(str(root / "tree" / "u_2000.nc"), hours, lats, lons, vals)
    masked = np.ma.masked_array(np.ones((4, 3, 2), dtype=np.float32))
    masked[1, 1, 1] = np.ma.masked                       # one masked cell
    write_field(str(root / "tree" / "u_2001.nc"), [0.0, 6.0], lats, lons, masked[:2],
                fill=-9999.0)
    (root / "tree" / "u_broken.nc").write_bytes(b"not a netcdf file")
    out = tmp_path / "inv.json"
    assert M.main(["--root", str(root), "--out", str(out)]) == 0
    v = json.load(open(out))["dirs"]["tree"]
    assert v["n_files"] == 3 and v["n_errors"] == 1
    assert "error" in v["files"]["u_broken.nc"]
    a = v["files"]["u_2000.nc"]
    assert a["time"] == {"n": 4, "first": "1900-01-01 00:00:00", "last": "1900-01-02 00:00:00",
                         "units": "hours since 1900-01-01", "calendar": "standard",
                         "unique_steps": [6.0, 12.0]}
    assert {k: a["lat"][k] for k in ("name", "n", "first", "last", "step")} == {
        "name": "latitude", "n": 3, "first": 10.0, "last": 9.0, "step": -0.5}
    assert {k: a["lon"][k] for k in ("name", "n", "first", "last", "step")} == {
        "name": "longitude", "n": 2, "first": -20.0, "last": -19.5, "step": 0.5}
    assert len(a["lat"]["fingerprint"]) == 64
    assert a["missing"]["u"]["nan"] == 1 and a["missing"]["u"]["masked"] == 0
    b = v["files"]["u_2001.nc"]
    assert b["missing"]["u"]["masked"] == 1 and b["missing"]["u"]["nan"] == 0
    assert v["masked_or_nan_cells"] == 2


def jan1(year):
    """Hours since 1900-01-01 at the start of a year, through the calendar, since a
    first version used 8760 hours per year and drifted a year by leap days."""
    import datetime as dt
    return float(nc.date2num(dt.datetime(year, 1, 1), "hours since 1900-01-01"))


def test_summary_is_descriptive_and_never_calls_a_tree_ready(tmp_path):
    """Two of the review's counterexamples and one positive control. "sparse" has both
    components on the whole buffered grid for every year of 1981 to 2010 but only two
    observations per year; "halves" has full six-hourly years with u on the northern
    half of the domain and v on the southern half, so the envelope reaches the target
    while neither component does; "complete" is a genuine full six-hourly year with
    both components on one grid. None may be called usable, and the first two must be
    visibly incomplete and split."""
    M = _load()
    root = tmp_path / "era5"
    lats = np.arange(50.0, -50.5, -1.0)
    lons = np.arange(-155.0, 55.5, 1.0)
    (root / "sparse").mkdir(parents=True)
    ones = np.ones((2, lats.size, lons.size), dtype=np.float32)
    for year in range(1981, 2011):
        h0 = jan1(year)
        write_field(str(root / "sparse" / f"u700_{year}.nc"), [h0, h0 + 6], lats, lons, ones)
        write_field(str(root / "sparse" / f"v700_{year}.nc"), [h0, h0 + 6], lats, lons, ones,
                    var="v")
    (root / "halves").mkdir()
    north, south = np.arange(50.0, -0.5, -1.0), np.arange(0.0, -50.5, -1.0)
    n1981 = 4 * 365
    hours = [jan1(1981) + 6.0 * k for k in range(n1981)]
    write_field(str(root / "halves" / "u700_1981.nc"), hours, north, lons,
                np.ones((n1981, north.size, lons.size), dtype=np.float32))
    write_field(str(root / "halves" / "v700_1981.nc"), hours, south, lons,
                np.ones((n1981, south.size, lons.size), dtype=np.float32), var="v")
    # "halfday": both components on one grid, first step 00Z on January 1, last step
    # 18Z on December 31, one uniform step, but three samples, so only the count of
    # four steps per calendar day can reject it
    (root / "halfday").mkdir()
    twelve = [jan1(1981), jan1(1981) + 4377.0, jan1(1981) + 8754.0]   # uniform, ends 18Z Dec 31
    for var in ("u", "v"):
        write_field(str(root / "halfday" / f"{var}700_1981.nc"), twelve, south, lons,
                    np.ones((len(twelve), south.size, lons.size), dtype=np.float32), var=var)
    (root / "complete").mkdir()
    write_field(str(root / "complete" / "u700_1981.nc"), hours, south, lons,
                np.ones((n1981, south.size, lons.size), dtype=np.float32))
    write_field(str(root / "complete" / "v700_1981.nc"), hours, south, lons,
                np.ones((n1981, south.size, lons.size), dtype=np.float32), var="v")
    out, rep = tmp_path / "inv.json", tmp_path / "INV.md"
    assert M.main(["--root", str(root), "--out", str(out), "--report", str(rep)]) == 0
    v = json.load(open(out))["dirs"]
    for name in ("sparse", "halves", "complete"):
        assert "usable_for_calibration" not in v[name]["summary"]
        assert v[name]["summary"]["calibration_readiness"] == "not assessed by this inventory"
    sp = v["sparse"]["summary"]
    assert sp["envelope_reaches_target_domain"] is True and sp["fine_enough"] is True
    assert len(sp["years_with_needed_fields"]) == 30
    assert sp["years_with_needed_fields_complete_six_hourly"] == []
    assert sp["years_with_needed_fields_on_one_grid"] == list(range(1981, 2011))
    assert any("not a complete six-hourly year: 30" in n for n in sp["notes"])
    ha = v["halves"]["summary"]
    assert ha["lat_extent_envelope"] == [-50.0, 50.0]
    assert ha["envelope_reaches_target_domain"] is True
    assert ha["years_with_needed_fields_complete_six_hourly"] == [1981]
    assert ha["years_with_needed_fields_on_one_grid"] == []
    assert any("do not share one grid: 1" in n for n in ha["notes"])
    hd = v["halfday"]["summary"]
    assert hd["years_with_needed_fields"] == [1981]
    assert hd["years_with_needed_fields_complete_six_hourly"] == []
    co = v["complete"]["summary"]
    assert co["years_with_needed_fields_complete_six_hourly"] == [1981]
    assert co["years_with_needed_fields_on_one_grid"] == [1981]
    assert co["envelope_reaches_target_domain"] is False       # the southern half only
    text = rep.read_text()
    assert "Usable" not in text and "usable" not in text
    assert "readiness is NOT assessed" in text
    assert "| sparse | 60 | 0 | 0 | 1.0 | -50.0 to 50.0 | -155.0 to 55.0 | 2 | 1981 to 2010 | 2 | 0 | 0 of 30 | 30 of 30 |" in text
    assert "| halves | 2 | 0 | 0 | 1.0 | -50.0 to 50.0 | -155.0 to 55.0 | 2 | 1981 to 1981 | 31 | 29 | 1 of 1 | 0 of 1 |" in text
    assert "| complete | 2 | 0 | 0 | 1.0 | -50.0 to 0.0 | -155.0 to 55.0 | 2 | 1981 to 1981 | 31 | 29 | 1 of 1 | 1 of 1 |" in text


def test_levels_years_and_fields_are_still_derived(tmp_path):
    M = _load()
    root = tmp_path / "era5"
    (root / "wide").mkdir(parents=True)
    lats = np.arange(50.0, -50.5, -1.0)
    lons = np.arange(-155.0, 55.5, 1.0)
    ones = np.ones((2, lats.size, lons.size), dtype=np.float32)
    for year in (1981, 1982):
        h0 = jan1(year)
        write_field(str(root / "wide" / f"u700_{year}.nc"), [h0, h0 + 6], lats, lons, ones)
        write_field(str(root / "wide" / f"v700_{year}.nc"), [h0, h0 + 6], lats, lons, ones,
                    var="v")
    write_field(str(root / "wide" / "u850_1981.nc"), [jan1(1981), jan1(1981) + 6], lats, lons,
                ones, level=850.0)
    (root / "uonly").mkdir()
    write_field(str(root / "uonly" / "u700_1981.nc"), [jan1(1981), jan1(1981) + 6], lats, lons,
                ones)
    out = tmp_path / "inv.json"
    assert M.main(["--root", str(root), "--out", str(out)]) == 0
    v = json.load(open(out))["dirs"]
    w = v["wide"]["summary"]
    assert w["field_level_combinations"] == {"u@700": 2, "v@700": 2, "u@850": 1}
    assert w["years_with_needed_fields"] == [1981, 1982]
    assert len(w["missing_years"]["1979-2010"]) == 30
    assert len(w["missing_years"]["1981-2010"]) == 28
    u = v["uonly"]["summary"]
    assert u["has_needed_fields"] is False
    assert "lacks ['v@700']" in u["notes"][0]


def test_one_grid_means_the_same_coordinate_arrays(tmp_path):
    M = _load()
    root = tmp_path / "era5"
    lons = [10.0, 11.0]
    n = 4 * 365
    hours = [jan1(1981) + 6.0 * k for k in range(n)]

    def tree(name, ulats, vlats):
        (root / name).mkdir(parents=True)
        write_field(str(root / name / "u700_1981.nc"), hours, ulats, lons,
                    np.ones((n, len(ulats), 2), dtype=np.float32))
        write_field(str(root / name / "v700_1981.nc"), hours, vlats, lons,
                    np.ones((n, len(vlats), 2), dtype=np.float32), var="v")
    tree("sameends", [1.0, 0.0, -1.0], [1.0, 0.5, 0.0, -0.5, -1.0])      # the review's case
    tree("interior", [2.0, 1.0, 0.0, -1.0, -2.0], [2.0, 1.0, 0.5, -1.0, -2.0])
    tree("reversed", [1.0, 0.0, -1.0], [-1.0, 0.0, 1.0])
    tree("same", [1.0, 0.0, -1.0], [1.0, 0.0, -1.0])
    out = tmp_path / "inv.json"
    assert M.main(["--root", str(root), "--out", str(out)]) == 0
    v = json.load(open(out))["dirs"]
    assert v["sameends"]["summary"]["years_with_needed_fields_on_one_grid"] == []
    assert v["interior"]["summary"]["years_with_needed_fields_on_one_grid"] == []
    assert v["reversed"]["summary"]["years_with_needed_fields_on_one_grid"] == [1981]
    assert v["same"]["summary"]["years_with_needed_fields_on_one_grid"] == [1981]
    # the interior case agrees on endpoints, count and the first spacing (2 to 1 on
    # both), so only the full array can tell it apart; the recorded `step` is the
    # smallest unique spacing, which does differ here, and is not what is compared
    ui = v["interior"]["files"]["u700_1981.nc"]["lat"]
    vi = v["interior"]["files"]["v700_1981.nc"]["lat"]
    assert (ui["first"], ui["last"], ui["n"]) == (vi["first"], vi["last"], vi["n"])
    assert ui["fingerprint"] != vi["fingerprint"]
