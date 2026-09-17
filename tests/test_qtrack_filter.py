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
Added after a second independent review with folder access (2026-09-17) found five more
survivors, two of them (R3, R4) from the FIRST review's catalog that the author's list
had never included:
  R1 select the SHORTEST bearer of a missing name (the fixture's bearers tied);
  R2 remove code 4 from the Pacific-origin exclusion (only a code-5 origin was tested);
  R3 add code 8 to the Atlantic codes (no code-8-only system existed);
  R4 count valid steps from longitude alone (lon and lat masks always agreed);
  R5 write zeros for curv_data_mean (the writer test checked its shape only);
  R6 the writer ignores the threshold it is given and writes "4" regardless.
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
    # 15: untagged, 5 steps, survivor of a three-member cluster (ties with 16 on
    #     length, lower index); 16 and 17: two twins both tagged ZETA, 5 and 4 steps
    #     (UNEQUAL, so "longest bearer" is distinguishable from "any bearer" and from
    #     "shortest bearer"): only 16 is retained
    # 18 and 19: an untagged TIE at 4 steps each: the lower index, 18, survives
    # 20: developer THETA at 4 steps, lower index; 21: untagged 5 steps, higher
    #     index, the longest: 21 survives and 20 is retained for its name
    # 22: begins in the CENTRAL Pacific (4) and later touches 7: NOT Atlantic
    # 23: code 8 only (east of the African window): NOT Atlantic
    # 24: five longitudes but only four latitudes (a lon-without-lat step), untagged;
    #     25: five full pairs sharing four with 24: 25 is the longer track by VALID
    #     PAIRS and survives, which a longitude-only count would get wrong
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
           [-70, -71, -72, -73, -74, NAN],
           [-70, -71, -72, -73, NAN, NAN],
           [-45, -46, -47, -48, NAN, NAN],
           [-45, -46, -47, -48, NAN, NAN],
           [-33, -34, -35, -36, NAN, NAN],
           [-33, -34, -35, -36, -37, NAN],
           [-150, -145, -140, -60, -55, NAN],
           [46, 47, 48, 49, NAN, NAN],
           [-60, -61, -62, -63, -64, NAN],
           [-60, -61, -62, -63, -64, NAN]]
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
           [18, 18, 18, 18, 18, NAN],
           [18, 18, 18, 18, NAN, NAN],
           [17, 17, 17, 17, NAN, NAN],
           [17, 17, 17, 17, NAN, NAN],
           [16, 16, 16, 16, NAN, NAN],
           [16, 16, 16, 16, 16, NAN],
           [15, 15, 15, 15, 15, NAN],
           [11, 11, 11, 11, NAN, NAN],
           [10, 10, 10, 10, NAN, NAN],
           [10, 10, 10, 10, 10, NAN]]
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
             [7, 7, 7, 7, 7, NAN],
             [7, 7, 7, 7, NAN, NAN],
             [7, 7, 7, 7, NAN, NAN],
             [7, 7, 7, 7, NAN, NAN],
             [7, 7, 7, 7, NAN, NAN],
             [7, 7, 7, 7, 7, NAN],
             [4, 4, 4, 7, 7, NAN],
             [8, 8, 8, 8, NAN, NAN],
             [7, 7, 7, 7, 7, NAN],
             [7, 7, 7, 7, 7, NAN]]
    names = ["ALPHA", "N/A", "N/A", "ALPHA", "N/A", "N/A", "N/A", "N/A",
             "N/A", "BETA", "GAMMA", "DELTA", "EPSILON", "N/A",
             "N/A", "N/A", "ZETA", "ZETA", "N/A", "N/A", "THETA", "N/A",
             "N/A", "N/A", "N/A", "N/A"]
    path = str(tmp_path / "y.nc")
    write_year(path, lon, lat, basin, names)
    return path


def test_atlantic_is_any_step_not_first_basin_and_not_land_or_nan(year):
    data = Q.read_year(year)
    atl = Q.atlantic_systems(data["basin"])
    assert atl.tolist() == [True, True, False, True, True, False, False, False,
                            True, True, True, True, True, True,
                            False, True, True, True, True, True, True, True,
                            False, False, True, True]


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
                                      F, T, T, T, T, T, T, T, F, F, T, T]
    assert r["duplicate"].tolist() == [F, F, F, T, F, F, F, F, F, F, F, F, F, T,
                                       F, F, F, T, F, T, F, F, F, F, T, F]
    assert r["keep"].tolist() == [T, T, F, F, T, F, F, F, T, T, T, T, T, F,
                                  F, T, T, F, T, F, T, T, F, F, F, T]
    assert r["retained_for_name"].tolist() == [F, F, F, F, F, F, F, F, F, T, F, T, F, F,
                                               F, F, T, F, F, F, T, F, F, F, F, F]
    assert r["n_systems"] == 26 and r["n_atlantic"] == 19
    assert r["n_duplicates_removed"] == 5 and r["n_kept"] == 14
    assert r["n_developers_kept"] == 7 and r["n_developers_removed_as_duplicate"] == 2
    assert r["n_retained_for_name"] == 4
    assert [g for g in r["clusters"] if len(g) > 1] == [
        [0, 3], [8, 9], [10, 11], [12, 13], [15, 16, 17], [18, 19], [20, 21], [24, 25]]


def test_write_subset_equals_the_expected_subset_of_the_source_in_every_variable(
        year, tmp_path):
    """EVERY variable of the output is compared against the subset of the source it
    must equal: data and masks, dimensions, dtype, fill value and attributes. A first
    version sampled a few values and let a writer that zeroed curv_data_mean pass."""
    data = Q.read_year(year)
    r = Q.filter_year(data, min_shared=4)
    dst = str(tmp_path / "atl.nc")
    Q.write_subset(year, dst, r["keep"], min_shared=4)
    src, out = nc.Dataset(year), nc.Dataset(dst)
    keep = r["keep"]
    assert set(out.variables) == set(src.variables)
    assert set(out.dimensions) == set(src.dimensions)
    assert len(out.dimensions["system"]) == int(keep.sum()) == 14
    assert out.dimensions["time"].isunlimited(), "the source's unlimited time must survive"
    assert out.source == "synthetic", "global attributes are copied"
    for name, sv in src.variables.items():
        ov = out.variables[name]
        assert ov.dimensions == sv.dimensions and ov.dtype == sv.dtype, name
        sa = {k: sv.getncattr(k) for k in sv.ncattrs()}
        oa = {k: ov.getncattr(k) for k in ov.ncattrs()}
        assert set(sa) == set(oa), f"{name} attribute names differ"
        for k in sa:                       # NaN fill values are equal to themselves here
            same = (isinstance(sa[k], float) and isinstance(oa[k], float)
                    and np.isnan(sa[k]) and np.isnan(oa[k])) or sa[k] == oa[k]
            assert same, f"{name}.{k} differs"
        want = sv[:]
        if "system" in sv.dimensions:
            want = np.compress(keep, want, axis=sv.dimensions.index("system"))
        got = ov[:]
        if sv.dtype == str:
            assert np.asarray(got).astype(str).tolist() == \
                np.asarray(want).astype(str).tolist(), name
        else:
            assert np.array_equal(np.ma.getmaskarray(got), np.ma.getmaskarray(want)), name
            assert np.array_equal(np.ma.filled(got, 0.0), np.ma.filled(want, 0.0)), name
    # the kept systems are the originals, not renumbered
    assert out["system"][:].tolist() == [1.0, 2.0, 5.0, 9.0, 10.0, 11.0, 12.0, 13.0,
                                         16.0, 17.0, 19.0, 21.0, 22.0, 26.0]
    assert out.aew_filter_min_shared == 4 and "4 or more" in out.aew_filter
    src.close()
    out.close()


def test_the_driver_records_a_nondefault_threshold_everywhere(tmp_path):
    """Run the actual driver at --min-shared 3 on a file with NONCONSECUTIVE system
    numbers: membership, the summary and the file metadata must agree, and the
    summary must speak in published system numbers."""
    import importlib.util
    import json
    import sys

    here = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location(
        "filter_qtrack_atlantic", os.path.join(here, "..", "scripts",
                                               "filter_qtrack_atlantic.py"))
    drv = importlib.util.module_from_spec(spec)
    sys.modules["filter_qtrack_atlantic"] = drv
    spec.loader.exec_module(drv)

    src_dir = tmp_path / "in"
    src_dir.mkdir()
    lon = [[-30, -34, -38, -42, NAN], [-30, -34, -38, -50, NAN], [-100, -104, NAN, NAN, NAN]]
    lat = [[10, 10, 11, 11, NAN], [10, 10, 11, 20, NAN], [12, 12, NAN, NAN, NAN]]
    basin = [[7, 7, 7, 7, NAN], [7, 7, 7, 7, NAN], [5, 5, NAN, NAN, NAN]]
    path = str(src_dir / "ERA5_AEW_tracks_with_basins_1999.nc")
    write_year(path, lon, lat, basin, ["N/A", "N/A", "N/A"])
    d = nc.Dataset(path, "a")
    d["system"][:] = [3.0, 17.0, 40.0]           # nonconsecutive published numbers
    d.close()
    out_dir = tmp_path / "out"
    summary = str(tmp_path / "summary.json")
    assert drv.main(["--directory", str(src_dir), "--out-directory", str(out_dir),
                     "--min-shared", "3", "--summary", summary]) == 0
    with open(summary) as fh:
        s = json.load(fh)
    y = s["years"]["1999"]
    # systems 3 and 17 share three positions: a repeat at threshold three, not four
    assert y["n_duplicates_removed"] == 1 and y["repeat_clusters"] == [[3, 17]]
    assert y["duplicates_removed_at_threshold"]["4"]["systems_removed"] == 0
    assert "3" in s["rules"]["repeats"]
    o = nc.Dataset(str(out_dir / "ERA5_AEW_tracks_atlantic_1999.nc"))
    assert o["system"][:].tolist() == [3.0]
    assert o.aew_filter_min_shared == 3 and "3 or more" in o.aew_filter
    o.close()


def test_guards():
    with pytest.raises(ValueError):
        Q.atlantic_systems(np.zeros(3))
    with pytest.raises(ValueError):
        Q.duplicate_clusters(np.zeros((2, 3)), np.zeros((2, 4)))
    with pytest.raises(ValueError):
        Q.duplicate_clusters(np.zeros((2, 3)), np.zeros((2, 3)), min_shared=0)
