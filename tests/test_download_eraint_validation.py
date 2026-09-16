"""Tests for the downloader's file validation, added when it was hardened.

WHAT WAS WRONG BEFORE. `plan()` treated any existing path as a completed retrieval and
`client.retrieve` wrote straight to the final path, so an interrupted request left a
partial file that every later run skipped as done. A review named it before the 1979 and
1980 retrieval, which is exactly when a partial file would have poisoned a named case.

The retrieval itself is not tested here, since it is an outward network request. What is
tested is the validation that stands between a file on disk and the claim that a year is
complete, and the planner's refusal to trust bare existence.
"""
import calendar
import datetime
import importlib.util
import os
import sys

import numpy as np
import pytest

nc = pytest.importorskip("netCDF4")

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "download_eraint_v1port", os.path.join(_HERE, "..", "scripts",
                                           "download_eraint_v1port.py"))
dl = importlib.util.module_from_spec(_spec)
sys.modules["download_eraint_v1port"] = dl
_spec.loader.exec_module(dl)

AREA, GRID = "12/-3/10/0", "1.0/1.0"                 # 3 lats x 4 lons, tiny on purpose


def write_file(path, year, *, steps=None, lat_shift=0.0, garbage=False,
               level=700.0, level_units="hPa", omit_level_var=False,
               time_units="seconds since 1970-01-01", zero_times=False):
    """Genuine-shaped by default (the real files carry pressure_level=[700.] hPa),
    with one knob per counterfeit a review executed against the old validator."""
    if garbage:
        with open(path, "wb") as fh:
            fh.write(b"this is not a netcdf file")
        return
    days = 366 if calendar.isleap(year) else 365
    n = steps if steps is not None else days * 4
    lat = np.array([12.0, 11.0, 10.0]) + lat_shift
    lon = np.array([-3.0, -2.0, -1.0, 0.0])
    start = (datetime.date(year, 1, 1) - datetime.date(1970, 1, 1)).days * 86400
    seconds = [0] * n if zero_times else [start + 21600 * i for i in range(n)]
    with nc.Dataset(path, "w") as ds:
        ds.createDimension("valid_time", n)
        ds.createDimension("pressure_level", 1)
        ds.createDimension("latitude", lat.size)
        ds.createDimension("longitude", lon.size)
        t = ds.createVariable("valid_time", "i8", ("valid_time",))
        t.units = time_units
        t[:] = seconds
        if not omit_level_var:
            lv = ds.createVariable("pressure_level", "f8", ("pressure_level",))
            lv.units = level_units
            lv[:] = [level]
        ds.createVariable("latitude", "f8", ("latitude",))[:] = lat
        ds.createVariable("longitude", "f8", ("longitude",))[:] = lon
        ds.createVariable("u", "f4", ("valid_time", "pressure_level", "latitude",
                                      "longitude"))[:] = \
            np.zeros((n, 1, lat.size, lon.size))


def test_a_complete_year_validates(tmp_path):
    path = str(tmp_path / "eraint_u700_1981_6h_region.nc")
    write_file(path, 1981)
    assert dl.validate_file(path, 1981, "u700", AREA, GRID) is None


def test_a_truncated_year_is_named(tmp_path):
    """The interrupted-download case: a partial file must fail with the counts named."""
    path = str(tmp_path / "eraint_u700_1981_6h_region.nc")
    write_file(path, 1981, steps=900)
    reason = dl.validate_file(path, 1981, "u700", AREA, GRID)
    assert reason is not None and "900" in reason and "1460" in reason


def test_a_shifted_grid_is_named(tmp_path):
    path = str(tmp_path / "eraint_u700_1981_6h_region.nc")
    write_file(path, 1981, lat_shift=0.5)
    reason = dl.validate_file(path, 1981, "u700", AREA, GRID)
    assert reason is not None and "grid" in reason and "exactly" in reason


def test_an_unreadable_file_is_named_not_raised(tmp_path):
    path = str(tmp_path / "eraint_u700_1981_6h_region.nc")
    write_file(path, 1981, garbage=True)
    reason = dl.validate_file(path, 1981, "u700", AREA, GRID)
    assert reason is not None and "unreadable" in reason


def test_the_planner_validates_existing_files_instead_of_trusting_them(tmp_path,
                                                                       capsys):
    """A valid file counts as done; a truncated one is moved aside to .invalid and
    re-queued, so a broken earlier run cannot satisfy this one silently."""
    good = str(tmp_path / "eraint_u700_1981_6h_region.nc")
    bad = str(tmp_path / "eraint_v700_1981_6h_region.nc")
    write_file(good, 1981)
    write_file(bad, 1981, steps=700)
    todo, done = dl.plan([1981], ["u700", "v700"], str(tmp_path),
                         area=AREA, grid=GRID)
    assert [(y, v) for y, v, _ in done] == [(1981, "u700")]
    assert [(y, v) for y, v, _ in todo] == [(1981, "v700")]
    assert not os.path.exists(bad), "the failed file must not remain at the final path"
    assert os.path.exists(bad + ".invalid"), "it is kept aside for inspection"
    assert "FAILED VALIDATION" in capsys.readouterr().out


def test_repeated_zero_timestamps_are_refused(tmp_path):
    """The review's first counterfeit: 1,460 timestamps all equal to zero passed the old
    count-only check. Decoded through the loader's rules they are 1970, not the year."""
    path = str(tmp_path / "eraint_u700_1981_6h_region.nc")
    write_file(path, 1981, zero_times=True)
    reason = dl.validate_file(path, 1981, "u700", AREA, GRID)
    assert reason is not None and "not the requested year" in reason


def test_a_wrong_epoch_is_refused_not_decoded_as_luck(tmp_path):
    """The review's second counterfeit: hours since 1800 looked six-hourly in raw values.
    The loader refuses units it does not convert, and so must the validator."""
    path = str(tmp_path / "eraint_u700_1981_6h_region.nc")
    write_file(path, 1981, time_units="hours since 1800-01-01")
    reason = dl.validate_file(path, 1981, "u700", AREA, GRID)
    assert reason is not None and "unreadable" in reason


def test_a_wrong_year_with_valid_everything_else_is_refused(tmp_path):
    path = str(tmp_path / "eraint_u700_1981_6h_region.nc")
    write_file(path, 1982)                            # a genuine, complete 1982
    reason = dl.validate_file(path, 1981, "u700", AREA, GRID)
    assert reason is not None and "not the requested year" in reason


def test_the_wrong_pressure_level_is_refused(tmp_path):
    """The scientifically consequential one: the loader squeezes a singleton level axis
    without reading its coordinate, so an 850 hPa field named u700 would run a named
    case at the wrong level silently. The validator establishes the level instead."""
    path = str(tmp_path / "eraint_u700_1981_6h_region.nc")
    write_file(path, 1981, level=850.0)
    reason = dl.validate_file(path, 1981, "u700", AREA, GRID)
    assert reason is not None and "850" in reason and "700" in reason
    # Pascals are recognised and converted, so 70000 Pa is the RIGHT level
    write_file(path, 1981, level=70000.0, level_units="Pa")
    assert dl.validate_file(path, 1981, "u700", AREA, GRID) is None


def test_a_level_dimension_without_a_coordinate_is_refused(tmp_path):
    """Unestablished never means passed: a level axis whose value cannot be read is a
    refusal, not a benefit of the doubt."""
    path = str(tmp_path / "eraint_u700_1981_6h_region.nc")
    write_file(path, 1981, omit_level_var=True)
    reason = dl.validate_file(path, 1981, "u700", AREA, GRID)
    assert reason is not None and "cannot be established" in reason


def test_coordinates_are_compared_exactly_not_approximately(tmp_path):
    """allclose's default relative tolerance would bless a grid a millionth of a degree
    off; array_equal does not."""
    path = str(tmp_path / "eraint_u700_1981_6h_region.nc")
    write_file(path, 1981, lat_shift=1e-9)
    reason = dl.validate_file(path, 1981, "u700", AREA, GRID)
    assert reason is not None and "grid" in reason


def test_a_dry_run_plan_is_read_only(tmp_path, capsys):
    """A first version moved invalid files aside even under --dry-run, which is a write
    a print-the-plan flag must not perform."""
    bad = str(tmp_path / "eraint_u700_1981_6h_region.nc")
    write_file(bad, 1981, steps=700)
    todo, done = dl.plan([1981], ["u700"], str(tmp_path), area=AREA, grid=GRID,
                         repair=False)
    assert os.path.exists(bad), "the dry run must leave the file exactly where it was"
    assert not os.path.exists(bad + ".invalid")
    assert [(y, v) for y, v, _ in todo] == [(1981, "u700")]
    assert "would be moved aside" in capsys.readouterr().out


def test_a_duplicated_timestamp_with_the_right_count_is_refused(tmp_path):
    """Cadence in isolation: full count, correct start, one duplicated stamp mid-year.

    Every other timing counterfeit here also breaks the count or the start, so the
    cadence check alone stood between this file and acceptance, and a mutation dropping
    it survived until this test existed.
    """
    path = str(tmp_path / "eraint_u700_1981_6h_region.nc")
    write_file(path, 1981)
    with nc.Dataset(path, "a") as ds:
        times = ds["valid_time"][:]
        times[800] = times[799]                       # a duplicate, count unchanged
        ds["valid_time"][:] = times
    reason = dl.validate_file(path, 1981, "u700", AREA, GRID)
    assert reason is not None and "six-hourly" in reason
