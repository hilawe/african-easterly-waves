"""The shared-tail evidence script classifies pair geometry from masks, nothing more.

MUTATION LIST, written before the assertions: report a contained track as a shared
tail; drop the common-end requirement so a partial overlap reads as a shared tail;
treat a zero genesis value as a date; refuse nothing on an empty directory. (A first
list had "report a split as a shared tail", which the branch order makes equivalent to
the original, an inert mutant rather than a caught one.)
Added after a further review: drop the interior check, so tracks that diverge or lose
validity BETWEEN the first and last shared step still read as a shared tail; decode
genesis by dividing by one billion whatever the declared units say.
"""
import datetime as dt
import importlib.util
import os
import sys

import numpy as np
import pytest

pytest.importorskip("netCDF4")

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "diagnose_qtrack_shared_tails", os.path.join(_HERE, "..", "scripts",
                                                 "diagnose_qtrack_shared_tails.py"))
diag = importlib.util.module_from_spec(_spec)
sys.modules["diagnose_qtrack_shared_tails"] = diag
_spec.loader.exec_module(diag)

T, F = True, False


def test_geometry_classes_by_hand():
    # different heads, identical to the common end: the archive's merge shape
    va = np.array([T, T, T, T, T, F])
    vb = np.array([F, T, T, T, T, F])
    same = np.array([F, F, T, T, T, F])
    assert diag.geometry_class(va, vb, same).startswith("shared tail")
    # b lies entirely within the shared span: contained
    vb2 = np.array([F, F, T, T, T, F])
    assert diag.geometry_class(va, vb2, same) == "contained"
    # shared head, different tails: split
    va3 = np.array([T, T, T, T, F, F])
    vb3 = np.array([T, T, F, T, T, T])
    same3 = np.array([T, T, F, F, F, F])
    assert diag.geometry_class(va3, vb3, same3).startswith("split")
    # INTERIOR divergence: valid throughout, equal at steps 1, 2, 4, 5 but not 3, so
    # the positions are NOT identical to a common end (a first classifier said they
    # were, having looked only before the first match and after the last)
    va5 = np.array([T, T, T, T, T, T])
    same5 = np.array([F, T, T, F, T, T])
    assert diag.geometry_class(va5, va5, same5) == "partial overlap"
    # INTERIOR mask gap: b has a head of its own, then is missing one step inside the
    # span, so it is neither contained nor identical to a common end
    vb6 = np.array([T, T, T, F, T, T])
    same6 = np.array([F, T, T, F, T, T])
    assert diag.geometry_class(va5, vb6, same6) == "partial overlap"
    # a track whose every valid step is shared lies inside the other: contained
    vb7 = np.array([F, T, T, F, T, T])
    assert diag.geometry_class(va5, vb7, same6) == "contained"
    # b starts and ends inside a's span but has an UNSHARED step of its own in the
    # middle: not contained (an ends-only check said it was), not a shared tail
    vb8 = np.array([F, T, T, T, T, F])
    same8 = np.array([F, T, T, F, T, F])
    assert diag.geometry_class(va5, vb8, same8) == "partial overlap"
    # a continues after the shared span while b started before it: partial
    va4 = np.array([F, T, T, T, T, T])
    vb4 = np.array([T, T, T, T, F, F])
    same4 = np.array([F, F, T, T, F, F])
    assert diag.geometry_class(va4, vb4, same4) == "partial overlap"


def test_genesis_is_decoded_through_its_declared_units():
    import datetime as dt
    first = dt.datetime(2013, 10, 13, 12)
    NS = "nanoseconds since 1970-01-01"
    S = "seconds since 1970-01-01"
    for bad in (0.0, float("nan"), float("inf"), -5.0):
        assert diag.decode_genesis(bad, NS) is None
    # 14 October 2013 00Z, once in nanoseconds and once in seconds: the same instant
    assert diag.decode_genesis(1381708800e9, NS) == diag.decode_genesis(1381708800.0, S)
    assert diag.gen_date(diag.decode_genesis(1381708800.0, S)) == "2013-10-14 00Z"
    assert diag.relative_genesis(diag.decode_genesis(1381708800.0, S), first) == "at or after"
    assert diag.relative_genesis(diag.decode_genesis(1381600800.0, S), first) == "before"
    assert diag.relative_genesis(None, first) is None
    with pytest.raises(ValueError):
        diag.decode_genesis(5.0, "fortnights since 1970-01-01")


def test_the_command_decodes_a_seconds_genesis_end_to_end(tmp_path):
    """A file declaring genesis in seconds must yield 2013-10-14 00Z, not 1970."""
    import json
    import netCDF4 as nc
    NAN = float("nan")
    d = nc.Dataset(str(tmp_path / "ERA5_AEW_tracks_with_basins_2013.nc"), "w")
    d.createDimension("time", None)
    d.createDimension("system", 2)
    tv = d.createVariable("time", "f8", ("time",))
    tv.units = "hours since 1900-01-01"
    base = (dt.datetime(2013, 10, 13, 0) - dt.datetime(1900, 1, 1)).total_seconds() / 3600
    tv[:] = [base + 6 * k for k in range(6)]
    d.createVariable("system", "f8", ("system",))[:] = [1.0, 2.0]
    for name, rows in (("AEW_lon", [[-70, -71, -72, -73, -74, -75], [-60, -61, -72, -73, -74, -75]]),
                       ("AEW_lat", [[10, 10, 10, 10, 10, 10], [11, 11, 10, 10, 10, 10]]),
                       ("basin_des", [[7] * 6, [7] * 6])):
        v = d.createVariable(name, "f8", ("system", "time"), fill_value=NAN)
        v[:] = rows
    d.createVariable("first_basin_des", "f8", ("system",))[:] = [7.0, 7.0]
    tn = d.createVariable("TC_name", str, ("system",))
    tn[0], tn[1] = "PRISCILLA", "OCTAVE"
    g = d.createVariable("TC_gen_time", "f8", ("system",), fill_value=NAN)
    g.units = "seconds since 1970-01-01"
    g[:] = [1381708800.0, 1381600800.0]           # 10-14 00Z and 10-12 18Z
    d.close()
    out = str(tmp_path / "ev.json")
    assert diag.main(["--directory", str(tmp_path), "--out", out]) == 0
    with open(out) as fh:
        ev = json.load(fh)
    pair = ev["pairs_with_two_names_or_a_name_on_the_shorter_member_only"][0]
    assert {t["name"]: t["tc_genesis"] for t in pair["tracks"]} == \
        {"PRISCILLA": "2013-10-14 00Z", "OCTAVE": "2013-10-12 18Z"}
    assert pair["shared_span"]["first"] == "2013-10-13 12Z"
    assert pair["genesis_relative_to_shared_span"] == {"1": "at or after", "2": "before"}
    assert ev["time_units_by_year"]["2013"]["TC_gen_time"] == "seconds since 1970-01-01"


def test_an_empty_directory_is_refused(tmp_path):
    assert diag.main(["--directory", str(tmp_path)]) == 2


def test_a_file_without_genesis_units_is_refused(tmp_path):
    """No units on TC_gen_time means no decoding, not a guess of nanoseconds."""
    import netCDF4 as nc
    NAN = float("nan")
    d = nc.Dataset(str(tmp_path / "ERA5_AEW_tracks_with_basins_2001.nc"), "w")
    d.createDimension("time", None)
    d.createDimension("system", 1)
    tv = d.createVariable("time", "f8", ("time",))
    tv.units = "hours since 1900-01-01"
    tv[:] = [0.0, 6.0]
    d.createVariable("system", "f8", ("system",))[:] = [1.0]
    for name in ("AEW_lon", "AEW_lat", "basin_des"):
        d.createVariable(name, "f8", ("system", "time"), fill_value=NAN)[:] = [[7.0, 7.0]]
    d.createVariable("first_basin_des", "f8", ("system",))[:] = [7.0]
    d.createVariable("TC_name", str, ("system",))[0] = "N/A"
    d.createVariable("TC_gen_time", "f8", ("system",), fill_value=NAN)[:] = [NAN]
    d.close()
    assert diag.main(["--directory", str(tmp_path)]) == 2
