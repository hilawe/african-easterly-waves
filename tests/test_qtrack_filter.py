"""The QTrack Atlantic and repeat filters, bound on synthetic files the real schema shape.

MUTATION LIST, written before the assertions:
  A1 decide Atlantic by first_basin_des instead of any step;
  A2 treat a NaN basin as Atlantic;
  A3 include code 1 (land) among the Atlantic codes;
  D1 compare longitude only, ignoring latitude;
  D2 count shared steps with `>` instead of `>=` (three shared at threshold four
     must NOT cluster, four must);
  D3 drop the transitive union (a-b, b-c chain);
  D4 keep the shortest of a cluster instead of the longest;
  D5 run deduplication over all systems instead of Atlantic ones only;
  D6 drop the name retention, so a developer twin of an untagged longer track, or
     of a differently named storm, is removed (found on the real files: Bertha 1996,
     Harvey 2017, and four clusters of two distinct storms);
  W1 the subset writer forgets to subset a system-dimension variable;
  W2 the subset writer renumbers `system` instead of keeping the originals.
Added after a repository-access review (2026-09-17) found seven survivors on the first
fixture:
  A4 drop the Pacific-origin exclusion (a system beginning in code 5 that later clips
     code 6 or 7 counted as Atlantic: Rick 2009, Simon 2014, Darby 2016, Dora 1999);
  D7 keep EVERY bearer of a missing name instead of the longest one;
  D8 break ties toward the higher index;
  D9 take the retention survivor as group[0] rather than the longest;
  W3 drop the fill value when copying a variable;
  W4 write `time` as a fixed dimension when the source had it unlimited;
  W5 drop the global attributes.
"""
import os

import numpy as np
import pytest

nc = pytest.importorskip("netCDF4")

from aew import qtrack as Q  # noqa: E402

NAN = np.nan


def write_year(path, lon, lat, basin, names):
    """A file in the published schema's shape: `time` UNLIMITED as in the real files,
    a global attribute, `system` numbered from 1, and a fill value on every array."""
    lon, lat, basin = (np.asarray(a, dtype=float) for a in (lon, lat, basin))
    n_sys, n_t = lon.shape
    d = nc.Dataset(path, "w")
    d.setncattr("source", "synthetic")
    d.createDimension("time", None)
    d.createDimension("system", n_sys)
    d.createDimension("longitude", 4)
    d.createVariable("time", "f8", ("time",))[:] = np.arange(n_t) * 21600.0
    d.createVariable("system", "f8", ("system",))[:] = np.arange(1, n_sys + 1, dtype=float)
    for name, arr in (("AEW_lon", lon), ("AEW_lat", lat), ("basin_des", basin)):
        v = d.createVariable(name, "f8", ("system", "time"), fill_value=NAN)
        v.long_name = name
        v[:] = arr
    fb = d.createVariable("first_basin_des", "f8", ("system",), fill_value=NAN)
    fb[:] = [next((c for c in row if c == c), NAN) for row in basin]
    tn = d.createVariable("TC_name", str, ("system",))
    for i, nm in enumerate(names):
        tn[i] = nm
    cm = d.createVariable("curv_data_mean", "f8", ("time", "longitude"))
    cm[:] = np.arange(n_t * 4, dtype=float).reshape(n_t, 4)
    d.close()


@pytest.fixture
def year(tmp_path):
    # 0: Africa then Atlantic (Atlantic)
    # 1: land first, then Caribbean (Atlantic, though first_basin_des is 1)
    # 2: eastern Pacific only (not Atlantic)
    # 3: duplicate of 0 on four steps, shorter (removed as a repeat)
    # 4: shares only THREE positions with 0 (kept at threshold four)
    # 5: Pacific twin sharing positions with 2 (must not touch the Atlantic count)
    # 6: NaN basin throughout with positions (not Atlantic)
    # 7: over land in the Americas throughout, code 1 only (not Atlantic; a first
    #    version of this fixture had no land-only track and the mutation that adds
    #    land to the Atlantic codes survived)
    # 8: untagged Atlantic track of 5 steps; 9: its shorter twin, tagged BETA, shares
    #    4 positions, so 9 is RETAINED for its name and 8 survives as the longest
    # 10 and 11: two different storms (GAMMA, DELTA) sharing 4 positions, both kept
    # 12: tagged EPSILON, 5 steps; 13: its untagged shorter twin, REMOVED ("N/A" is
    #     not a name, so nothing is retained for it; a mutation treating it as one
    #     survived the fixture without this pair)
    # 14: begins in the eastern Pacific (5) and later clips the Caribbean (6): NOT
    #     Atlantic (the review's Rick/Simon/Darby/Dora class)
    # 15: untagged, 5 steps, survivor of a three-member cluster; 16 and 17: two twins
    #     both tagged ZETA, 4 steps each (a twin with three steps could not join a
    #     four-step cluster, which a first draft of this fixture got wrong): only 16,
    #     the lower index of the tied bearers, is retained
    # 18 and 19: an untagged TIE at 4 steps each: the lower index, 18, survives
    # 20: developer THETA at 4 steps, lower index; 21: untagged 5 steps, higher
    #     index, the longest: 21 survives and 20 is retained for its name
    lon = [[-10, -14, -18, -22, -26, -30],
           [-84, -82, -80, -78, NAN, NAN],
           [-100, -104, -108, -112, -116, NAN],
           [-10, -14, -18, -22, NAN, NAN],
           [-10, -14, -18, -40, NAN, NAN],
           [-100, -104, -108, -112, NAN, NAN],
           [-50, -52, -54, NAN, NAN, NAN],
           [-100, -101, -102, -103, NAN, NAN],
           [-30, -34, -38, -42, -46, NAN],
           [-30, -34, -38, -42, NAN, NAN],
           [-40, -44, -48, -52, -56, NAN],
           [-40, -44, -48, -52, NAN, NAN],
           [-20, -24, -28, -32, -36, NAN],
           [-20, -24, -28, -32, NAN, NAN],
           [-100, -96, -92, -88, -84, -80],
           [-70, -71, -72, -73, -74, NAN],
           [-70, -71, -72, -73, NAN, NAN],
           [-70, -71, -72, -73, NAN, NAN],
           [-45, -46, -47, -48, NAN, NAN],
           [-45, -46, -47, -48, NAN, NAN],
           [-33, -34, -35, -36, NAN, NAN],
           [-33, -34, -35, -36, -37, NAN]]
    lat = [[10, 10, 11, 11, 12, 12],
           [12, 13, 14, 15, NAN, NAN],
           [12, 12, 13, 13, 14, NAN],
           [10, 10, 11, 11, NAN, NAN],
           [10, 10, 11, 20, NAN, NAN],
           [12, 12, 13, 13, NAN, NAN],
           [12, 12, 12, NAN, NAN, NAN],
           [20, 20, 21, 21, NAN, NAN],
           [14, 14, 15, 15, 16, NAN],
           [14, 14, 15, 15, NAN, NAN],
           [13, 13, 14, 14, 15, NAN],
           [13, 13, 14, 14, NAN, NAN],
           [11, 11, 12, 12, 13, NAN],
           [11, 11, 12, 12, NAN, NAN],
           [12, 13, 14, 15, 16, 17],
           [18, 18, 18, 18, 18, NAN],
           [18, 18, 18, 18, NAN, NAN],
           [18, 18, 18, 18, NAN, NAN],
           [17, 17, 17, 17, NAN, NAN],
           [17, 17, 17, 17, NAN, NAN],
           [16, 16, 16, 16, NAN, NAN],
           [16, 16, 16, 16, 16, NAN]]
    basin = [[2, 2, 7, 7, 7, 7],
             [1, 1, 6, 6, NAN, NAN],
             [5, 5, 5, 5, 5, NAN],
             [2, 2, 7, 7, NAN, NAN],
             [2, 2, 7, 7, NAN, NAN],
             [5, 5, 5, 5, NAN, NAN],
             [NAN, NAN, NAN, NAN, NAN, NAN],
             [1, 1, 1, 1, NAN, NAN],
             [7, 7, 7, 7, 7, NAN],
             [7, 7, 7, 7, NAN, NAN],
             [7, 7, 7, 7, 7, NAN],
             [7, 7, 7, 7, NAN, NAN],
             [7, 7, 7, 7, 7, NAN],
             [7, 7, 7, 7, NAN, NAN],
             [5, 5, 5, 1, 6, 6],
             [7, 7, 7, 7, 7, NAN],
             [7, 7, 7, 7, NAN, NAN],
             [7, 7, 7, 7, NAN, NAN],
             [7, 7, 7, 7, NAN, NAN],
             [7, 7, 7, 7, NAN, NAN],
             [7, 7, 7, 7, NAN, NAN],
             [7, 7, 7, 7, 7, NAN]]
    names = ["ALPHA", "N/A", "N/A", "ALPHA", "N/A", "N/A", "N/A", "N/A",
             "N/A", "BETA", "GAMMA", "DELTA", "EPSILON", "N/A",
             "N/A", "N/A", "ZETA", "ZETA", "N/A", "N/A", "THETA", "N/A"]
    path = str(tmp_path / "y.nc")
    write_year(path, lon, lat, basin, names)
    return path


def test_atlantic_is_any_step_not_first_basin_and_not_land_or_nan(year):
    data = Q.read_year(year)
    atl = Q.atlantic_systems(data["basin"])
    assert atl.tolist() == [True, True, False, True, True, False, False, False,
                            True, True, True, True, True, True,
                            False, True, True, True, True, True, True, True]


def test_duplicates_need_both_coordinates_at_the_threshold_and_are_transitive():
    lon = np.array([[1.0, 2.0, 3.0, 4.0, 5.0],
                    [1.0, 2.0, 3.0, 4.0, NAN],       # shares 4 with 0
                    [1.0, 2.0, 3.0, 9.0, NAN],       # shares 3 with 0 and 1
                    [1.0, 2.0, 3.0, 4.0, NAN],       # same lon as 1, different lat
                    [NAN, 2.0, 3.0, 4.0, 5.0]])      # shares 4 with 0, 3 with 1
    lat = np.array([[0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, NAN],
                    [0.0, 0.0, 0.0, 0.0, NAN],
                    [1.0, 1.0, 1.0, 1.0, NAN],
                    [NAN, 0.0, 0.0, 0.0, 0.0]])
    assert Q.duplicate_clusters(lon, lat, min_shared=4) == [[0, 1, 4], [2], [3]]
    # at threshold three, system 2 joins through its three shared steps
    assert Q.duplicate_clusters(lon, lat, min_shared=3) == [[0, 1, 2, 4], [3]]
    # the chain: 4 shares 4 with 0 and only 3 with 1, yet all three are one cluster
    keep = Q.keep_longest([[0, 1, 4], [2], [3]], n_valid=[5, 4, 4, 4, 4])
    assert keep.tolist() == [True, False, True, True, False]


def test_filter_year_removes_the_shorter_repeat_among_atlantic_systems_only(year):
    data = Q.read_year(year)
    r = Q.filter_year(data, min_shared=4)
    T, F = True, False
    assert r["atlantic"].tolist() == [T, T, F, T, T, F, F, F, T, T, T, T, T, T,
                                      F, T, T, T, T, T, T, T]
    assert r["duplicate"].tolist() == [F, F, F, T, F, F, F, F, F, F, F, F, F, T,
                                       F, F, F, T, F, T, F, F]
    assert r["keep"].tolist() == [T, T, F, F, T, F, F, F, T, T, T, T, T, F,
                                  F, T, T, F, T, F, T, T]
    assert r["retained_for_name"].tolist() == [F, F, F, F, F, F, F, F, F, T, F, T, F, F,
                                               F, F, T, F, F, F, T, F]
    assert r["n_systems"] == 22 and r["n_atlantic"] == 17
    assert r["n_duplicates_removed"] == 4 and r["n_kept"] == 13
    assert r["n_developers_kept"] == 7 and r["n_developers_removed_as_duplicate"] == 2
    assert r["n_retained_for_name"] == 4
    assert [g for g in r["clusters"] if len(g) > 1] == [
        [0, 3], [8, 9], [10, 11], [12, 13], [15, 16, 17], [18, 19], [20, 21]]


def test_write_subset_keeps_original_system_numbers_and_all_variables(year, tmp_path):
    data = Q.read_year(year)
    r = Q.filter_year(data, min_shared=4)
    dst = str(tmp_path / "atl.nc")
    Q.write_subset(year, dst, r["keep"])
    d = nc.Dataset(dst)
    assert set(d.variables) == {"time", "system", "AEW_lon", "AEW_lat", "basin_des",
                                "first_basin_des", "TC_name", "curv_data_mean"}
    assert len(d.dimensions["system"]) == 13 and len(d.dimensions["time"]) == 6
    assert d.dimensions["time"].isunlimited(), "the source's unlimited time must survive"
    assert d.source == "synthetic", "global attributes are copied"
    assert d["system"][:].tolist() == [1.0, 2.0, 5.0, 9.0, 10.0, 11.0, 12.0, 13.0,
                                       16.0, 17.0, 19.0, 21.0, 22.0]
    assert np.asarray(d["TC_name"][:]).astype(str).tolist() == [
        "ALPHA", "N/A", "N/A", "N/A", "BETA", "GAMMA", "DELTA", "EPSILON",
        "N/A", "ZETA", "N/A", "THETA", "N/A"]
    assert np.ma.filled(d["AEW_lon"][:], NAN)[2, 3] == -40.0     # system 5's own track
    assert d["first_basin_des"][:].tolist() == [2.0, 1.0, 2.0, 7.0, 7.0, 7.0, 7.0, 7.0,
                                                 7.0, 7.0, 7.0, 7.0, 7.0]
    assert d["curv_data_mean"].shape == (6, 4) and d["AEW_lon"].long_name == "AEW_lon"
    for name in ("AEW_lon", "AEW_lat", "basin_des", "first_basin_des"):
        assert np.isnan(d[name]._FillValue), f"{name} lost its fill value"
    assert "positional repeats removed" in d.aew_filter
    d.close()


def test_guards():
    with pytest.raises(ValueError):
        Q.atlantic_systems(np.zeros(3))
    with pytest.raises(ValueError):
        Q.duplicate_clusters(np.zeros((2, 3)), np.zeros((2, 4)))
    with pytest.raises(ValueError):
        Q.duplicate_clusters(np.zeros((2, 3)), np.zeros((2, 3)), min_shared=0)
