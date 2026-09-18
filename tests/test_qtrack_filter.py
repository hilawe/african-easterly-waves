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
Added when the policy became KEEP EVERYTHING (2026-09-17, after a review showed the
installed QTrack post-processing can impose every shared tail by copying positions, so
the shorter head is a separate detection, not a duplicate; the archive's producing
revision is unconfirmed):
  K1 the keep policy still drops the shorter head;
  K2 observation_counts ignores the time step (dedupes on position alone);
  K3 observation_counts counts a record on three tracks as two distinct records;
  K4 storm_keys uses the name alone ("UNNAMED" twice with different genesis times
     collapses to one storm);
  K5 storm_keys uses the genesis time alone;
  K6 shared_tail_pairs reports first_shared for both ends of the span;
  K7 the retired drop_shorter policy no longer removes anything;
  K8 read_year stops reading TC_gen_time.
Added after a further review of the keep policy:
  K9 observation identity drops the latitude (same time and longitude, latitudes 5
     and 6, became one distinct record and one copy);
  K10 a tagged track with missing genesis is keyed by its name, so two unrelated
      UNNAMED tracks with no genesis collapse to one storm.
Observation ownership (the view a density analysis reads):
  O1 a shared record is owned by the track with FEWER valid records;
  O2 a tie is broken toward the HIGHER system number;
  O3 a shared record is marked owned on every holder (owned no longer equals
     distinct);
  O4 a record held by one track only is not owned at all.
"""
import os

import numpy as np
import pytest

nc = pytest.importorskip("netCDF4")

from aew import qtrack as Q  # noqa: E402

NAN = np.nan


def write_year(path, lon, lat, basin, names, gen_time=None):
    """A file in the published schema's shape: `time` UNLIMITED as in the real files,
    a global attribute, `system` numbered from 1, a fill value on every array, and
    TC_gen_time in nanoseconds since 1970 as every real file stores it."""
    lon, lat, basin = (np.asarray(a, dtype=float) for a in (lon, lat, basin))
    n_sys, n_t = lon.shape
    if gen_time is None:
        gen_time = [NAN] * n_sys
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
    gt = d.createVariable("TC_gen_time", "f8", ("system",), fill_value=NAN)
    gt.units = "nanoseconds since 1970-01-01"
    gt[:] = np.asarray(gen_time, dtype=float)
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
    # 18 and 19: a TIE at 4 steps each, both tagged UNNAMED with DIFFERENT genesis
    #     times (two distinct storms, as HURDAT's unnamed storms are); under the
    #     retired rule the lower index, 18, survives
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
             "N/A", "N/A", "ZETA", "ZETA", "UNNAMED", "UNNAMED", "THETA", "N/A",
             "N/A", "N/A", "N/A", "N/A"]
    G = 1.0e18                                  # nanoseconds since 1970, arbitrary scale
    gen = [NAN] * 26
    gen[0] = gen[3] = 1 * G                     # ALPHA tagged on both heads: one storm
    gen[9] = 2 * G
    gen[10], gen[11] = 3 * G, 4 * G
    gen[12] = 5 * G
    gen[16] = gen[17] = 6 * G                   # ZETA on both twins: one storm
    gen[18], gen[19] = 7 * G, 8 * G             # two UNNAMED storms, distinct
    gen[20] = 9 * G
    path = str(tmp_path / "y.nc")
    write_year(path, lon, lat, basin, names, gen)
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


def test_the_keep_policy_removes_nothing_and_reports_the_shared_tails(year):
    data = Q.read_year(year)
    r = Q.filter_year(data, min_shared=4)
    assert r["repeat_policy"] == "keep"
    assert r["keep"].tolist() == r["atlantic"].tolist(), "nothing beyond the basin rule"
    assert not r["duplicate"].any() and r["n_removed"] == 0
    assert r["n_kept"] == r["n_atlantic"] == 19
    pairs = {(q["a"], q["b"]): (q["first_shared"], q["last_shared"], q["n_shared"])
             for q in r["shared_tail_pairs"]}
    assert set(pairs) == {(0, 3), (8, 9), (10, 11), (12, 13), (15, 16), (15, 17),
                          (16, 17), (18, 19), (20, 21), (24, 25)}
    assert pairs[(0, 3)] == (0, 3, 4) and pairs[(15, 16)] == (0, 4, 5)
    assert pairs[(24, 25)] == (0, 3, 4), "the lon-without-lat step is not shared"
    assert r["n_shared_tail_pairs"] == 10
    # storm identity: ALPHA on two heads is one storm, ZETA on two twins is one,
    # two UNNAMED with different genesis times are two: nine storms, eleven tags
    assert r["n_developer_tags_kept"] == 11
    assert r["n_distinct_storms_kept"] == 9
    assert r["n_unresolved_storm_identities_kept"] == 0
    keys = r["storm_keys"]
    assert keys[0] == keys[3] == ("ALPHA", int(1.0e18)) and keys[1] is None
    assert keys[18] != keys[19] and keys[18][0] == keys[19][0] == "UNNAMED"


def test_missing_genesis_never_collapses_repeated_names(tmp_path):
    """Two unrelated retained tracks tagged UNNAMED in a file WITHOUT TC_gen_time: two
    developer tags, zero resolved storms, two unresolved identities."""
    lon = [[-30, -34, -38, NAN], [-50, -54, -58, NAN]]
    lat = [[10, 10, 11, NAN], [12, 12, 13, NAN]]
    basin = [[7, 7, 7, NAN], [7, 7, 7, NAN]]
    path = str(tmp_path / "nogen.nc")
    write_year(path, lon, lat, basin, ["UNNAMED", "UNNAMED"])
    d = nc.Dataset(path, "a")
    d["TC_gen_time"][:] = [NAN, NAN]
    d.close()
    r = Q.filter_year(Q.read_year(path), min_shared=4)
    assert r["n_developer_tags_kept"] == 2
    assert r["n_distinct_storms_kept"] == 0
    assert r["n_unresolved_storm_identities_kept"] == 2


def test_the_retired_drop_shorter_policy_is_still_reportable(year):
    """The old rule is kept only so the summary can say what it would have removed."""
    data = Q.read_year(year)
    r = Q.filter_year(data, min_shared=4, repeat_policy="drop_shorter")
    T, F = True, False
    assert r["keep"].tolist() == [T, T, F, F, T, F, F, F, T, T, T, T, T, F,
                                  F, T, T, F, T, F, T, T, F, F, F, T]
    assert r["n_removed"] == 5
    with pytest.raises(ValueError):
        Q.filter_year(data, min_shared=4, repeat_policy="merge")


def test_observation_counts_are_distinct_per_time_step_and_handle_triples():
    """Hand-counted: A has five records, B copies four of them, C copies two of those
    and adds one of its own at a NEW time step, and D repeats A's first position at a
    DIFFERENT time step, which is a distinct record. valid 5 + 4 + 3 + 1 = 13,
    distinct 5 + 0 + 1 + 1 = 7, copied 6. The record shared by A, B and C at steps 2
    and 3 is one distinct record and two copies each."""
    lon = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, NAN],
                    [1.0, 2.0, 3.0, 4.0, NAN, NAN],
                    [NAN, NAN, 3.0, 4.0, NAN, 9.0],
                    [NAN, 1.0, NAN, NAN, NAN, NAN]])
    lat = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, NAN],
                    [0.0, 0.0, 0.0, 0.0, NAN, NAN],
                    [NAN, NAN, 0.0, 0.0, NAN, 0.0],
                    [NAN, 0.0, NAN, NAN, NAN, NAN]])
    assert Q.observation_counts(lon, lat, [True] * 4) == \
        {"valid_records": 13, "distinct_records": 7, "copied_records": 6}
    assert Q.observation_counts(lon, lat, [True, False, False, True]) == \
        {"valid_records": 6, "distinct_records": 6, "copied_records": 0}
    # same time step and longitude, DIFFERENT latitudes: two distinct records, no copy
    lon2 = np.array([[10.0, 11.0], [10.0, NAN]])
    lat2 = np.array([[5.0, 5.0], [6.0, NAN]])
    assert Q.observation_counts(lon2, lat2, [True, True]) == \
        {"valid_records": 3, "distinct_records": 3, "copied_records": 0}


def test_storm_keys_need_both_name_and_genesis_time():
    names = ["UNNAMED", "UNNAMED", "N/A", "IDA", "IDA", "KATE"]
    gen = [1.0, 2.0, NAN, 5.0, 5.0, 5.0]
    keys = Q.storm_keys(names, gen)
    assert keys[2] is None
    assert keys[0] != keys[1], "two unnamed storms with different genesis times"
    assert keys[3] == keys[4], "one storm tagged on two tracks"
    assert keys[4] != keys[5], "two storms with one genesis time are still two"
    assert len({k for k in keys if k is not None}) == 4
    with pytest.raises(ValueError):
        Q.storm_keys(names, gen[:3])
    # missing, zero or non-finite genesis on a TAGGED track is an unresolved identity,
    # never a shared one: two UNNAMED with no genesis are not one storm
    keys = Q.storm_keys(["UNNAMED", "UNNAMED", "IDA", "N/A"], [NAN, 0.0, np.inf, NAN])
    assert keys[:3] == ["unresolved"] * 3 and keys[3] is None


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
    assert len(out.dimensions["system"]) == int(keep.sum()) == 19
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
    assert out["system"][:].tolist() == [1.0, 2.0, 4.0, 5.0, 9.0, 10.0, 11.0, 12.0,
                                         13.0, 14.0, 16.0, 17.0, 18.0, 19.0, 20.0,
                                         21.0, 22.0, 25.0, 26.0]
    assert out.aew_filter_min_shared == 4 and "4 or more" in out.aew_filter
    assert "Nothing removed for shared positions" in out.aew_filter
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
    # systems 3 and 17 share three positions: a shared-tail pair at threshold three,
    # BOTH KEPT, listed by published number; the retired rule would have removed 17
    assert y["n_kept"] == 2 and y["shared_tail_groups"] == [[3, 17]]
    assert [q["systems"] for q in y["shared_tail_pairs"]] == [[3, 17]]
    assert y["shared_tail_pairs"][0]["n_shared"] == 3
    assert y["retired_drop_shorter_rule"]["would_remove"] == [17]
    assert y["observations_kept"] == {"valid_records": 8, "distinct_records": 5,
                                      "copied_records": 3}
    assert "3" in s["rules"]["shared_tails"]
    o = nc.Dataset(str(out_dir / "ERA5_AEW_tracks_atlantic_1999.nc"))
    assert o["system"][:].tolist() == [3.0, 17.0]
    assert o.aew_filter_min_shared == 3 and "3 or more" in o.aew_filter
    o.close()


def test_guards():
    with pytest.raises(ValueError):
        Q.atlantic_systems(np.zeros(3))
    with pytest.raises(ValueError):
        Q.duplicate_clusters(np.zeros((2, 3)), np.zeros((2, 4)))
    with pytest.raises(ValueError):
        Q.duplicate_clusters(np.zeros((2, 3)), np.zeros((2, 3)), min_shared=0)


def test_observation_owners_is_one_track_per_record_and_matches_distinct_counts():
    """Hand-built: A (5 records, system 2) and B (4, system 7) share steps 0..3; C (2
    records, system 1) shares step 2 with both and holds step 5 alone; D (4 records,
    system 9) shares nothing. Owner of steps 0..3 is A (most records), so A also owns
    the step-2 record C holds; C owns step 5; D owns its four. Ten distinct records."""
    lon = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, NAN],
                    [1.0, 2.0, 3.0, 4.0, NAN, NAN],
                    [NAN, NAN, 3.0, NAN, NAN, 9.0],
                    [NAN, NAN, 7.0, 7.0, 7.0, 7.0]])
    lat = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, NAN],
                    [0.0, 0.0, 0.0, 0.0, NAN, NAN],
                    [NAN, NAN, 0.0, NAN, NAN, 0.0],
                    [NAN, NAN, 1.0, 1.0, 1.0, 1.0]])
    system = np.array([2.0, 7.0, 1.0, 9.0])
    v = Q.observation_owners(lon, lat, [True] * 4, system)
    T, F = True, False
    assert v["owned"].tolist() == [[T, T, T, T, T, F],
                                   [F, F, F, F, F, F],
                                   [F, F, F, F, F, T],
                                   [F, F, T, T, T, T]]
    assert v["owner_system"][(2, 3.0, 0.0)] == 2 and v["owner_system"][(5, 9.0, 0.0)] == 1
    counts = Q.observation_counts(lon, lat, [True] * 4)
    assert int(v["owned"].sum()) == counts["distinct_records"] == 10
    # a TIE: two tracks with equal record counts sharing everything, lower system wins
    lon2 = np.array([[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]])
    lat2 = np.zeros((2, 4))
    v2 = Q.observation_owners(lon2, lat2, [True, True], np.array([12.0, 5.0]))
    assert v2["owned"].tolist() == [[F, F, F, F], [T, T, T, T]]
    # a track not kept owns nothing and its records are not counted
    v3 = Q.observation_owners(lon2, lat2, [True, False], np.array([12.0, 5.0]))
    assert v3["owned"].tolist() == [[T, T, T, T], [F, F, F, F]]


def test_observation_owners_binds_the_rule_against_the_lazy_choosers():
    """Five implementations that a review wrote survived the worked case above while
    matching its every aggregate. Each block below is the one hand-counted case that
    separates the declared rule from one of them, asserting the WHOLE mask and the
    owner mapping, never a count. Recorded before the assertions were written:
      L1 keep the first holder unless the counts tie (a longer holder met later);
      L2 read the identity from time and longitude only (same lon, different lat);
      L3 round coordinates before comparing (close but unequal coordinates);
      L4 decide validity from longitude alone (a longitude with a missing latitude);
      L5 count valid records over kept rows only but index the original rows (an
         excluded row before a kept one)."""
    T, F = True, False
    keep_all = [True, True]
    # L1: the shorter track comes FIRST and shares its only record with a longer one
    v = Q.observation_owners([[1.0, NAN], [1.0, 2.0]], [[0.0, NAN], [0.0, 0.0]],
                             keep_all, [1.0, 2.0])
    assert v["owned"].tolist() == [[F, F], [T, T]]
    assert v["owner_system"] == {(0, 1.0, 0.0): 2, (1, 2.0, 0.0): 2}
    # L2: same time and longitude, different latitude, are two records, each owned
    v = Q.observation_owners([[1.0, 2.0], [1.0, 2.0]], [[0.0, 0.0], [5.0, 5.0]],
                             keep_all, [1.0, 2.0])
    assert v["owned"].tolist() == [[T, T], [T, T]]
    assert v["owner_system"] == {(0, 1.0, 0.0): 1, (1, 2.0, 0.0): 1,
                                 (0, 1.0, 5.0): 2, (1, 2.0, 5.0): 2}
    # L3: coordinates 0.04 degrees apart are different records; exact equality only
    v = Q.observation_owners([[1.00, 2.0], [1.04, 2.0]], [[0.0, 0.0], [0.0, 0.0]],
                             keep_all, [1.0, 2.0])
    assert v["owned"].tolist() == [[T, T], [T, F]]
    assert v["owner_system"][(0, 1.04, 0.0)] == 2 and v["owner_system"][(1, 2.0, 0.0)] == 1
    # L4: a longitude beside a missing latitude is no record, so the track that holds
    # it has ONE valid record, fewer than its twin, and loses the shared record
    v = Q.observation_owners([[1.0, 2.0, 3.0], [1.0, 2.0, NAN]],
                             [[0.0, NAN, NAN], [0.0, 0.0, NAN]], keep_all, [1.0, 2.0])
    assert v["owned"].tolist() == [[F, F, F], [T, T, F]]
    assert v["owner_system"] == {(0, 1.0, 0.0): 2, (1, 2.0, 0.0): 2}
    # L5: an EXCLUDED long track sits before two kept ones; counts must be read at the
    # original row, so the kept row 2 (three records) beats kept row 1 (two records)
    v = Q.observation_owners([[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, NAN, NAN], [1.0, 2.0, 3.0, NAN]],
                             [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, NAN, NAN], [0.0, 0.0, 0.0, NAN]],
                             [False, True, True], [1.0, 2.0, 3.0])
    assert v["owned"].tolist() == [[F, F, F, F], [F, F, F, F], [T, T, T, F]]
    assert v["owner_system"] == {(0, 1.0, 0.0): 3, (1, 2.0, 0.0): 3, (2, 3.0, 0.0): 3}


def _load_observation_driver():
    import importlib.util
    import sys

    here = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location(
        "observation_view_qtrack", os.path.join(here, "..", "scripts",
                                                "observation_view_qtrack.py"))
    drv = importlib.util.module_from_spec(spec)
    sys.modules["observation_view_qtrack"] = drv
    spec.loader.exec_module(drv)
    return drv


def test_the_observation_driver_writes_a_companion_a_reader_can_align(tmp_path):
    """Run the actual driver over two synthetic years and reopen everything: the whole
    owned mask, the coordinate axes decoded to the SAME dates as the source (a 360_day
    calendar on one file, which a companion carrying units alone would shift by a day),
    the recorded hashes, the per-year and total counts, and the worked example."""
    import hashlib
    import json

    drv = _load_observation_driver()
    tracks = tmp_path / "tracks"
    tracks.mkdir()
    # 1999: A (3 records, system 4) and B (2, system 9) share steps 0 and 1
    p99 = str(tracks / "ERA5_AEW_tracks_atlantic_1999.nc")
    write_year(p99, [[1.0, 2.0, 3.0], [1.0, 2.0, NAN]], [[0.0, 0.0, 0.0], [0.0, 0.0, NAN]],
               [[7, 7, 7], [7, 7, NAN]], ["N/A", "N/A"])
    d = nc.Dataset(p99, "a")
    d["system"][:] = [4.0, 9.0]
    d["time"].units = "hours since 1900-01-01"
    d["time"].calendar = "360_day"
    d["time"][:] = [1440.0, 1446.0, 1452.0]
    d.close()
    # 2000: one track, nothing shared
    p00 = str(tracks / "ERA5_AEW_tracks_atlantic_2000.nc")
    write_year(p00, [[5.0, 6.0]], [[1.0, 1.0]], [[7, 7]], ["N/A"])
    summary = str(tmp_path / "view.json")
    assert drv.main(["--tracks", str(tracks), "--summary", summary]) == 0

    o = nc.Dataset(p99[:-3] + "_owned.nc")
    s = nc.Dataset(p99)
    assert o["owned"][:].tolist() == [[1, 1, 1], [0, 0, 0]]
    assert o["system"][:].tolist() == [4.0, 9.0]
    assert o["time"].calendar == "360_day" and o["time"].units == s["time"].units
    assert nc.num2date(o["time"][:], o["time"].units, o["time"].calendar).tolist() == \
        nc.num2date(s["time"][:], s["time"].units, s["time"].calendar).tolist()
    assert nc.num2date(o["time"][0], o["time"].units, o["time"].calendar).strftime(
        "%m-%d") == "03-01"                                      # 1440 h in a 360-day year
    assert "most valid records" in o.aew_observation_rule
    o.close()
    s.close()

    with open(summary) as fh:
        v = json.load(fh)
    assert v["years"]["1999"]["valid_records"] == 5
    assert v["years"]["1999"]["distinct_records"] == 3
    assert v["years"]["1999"]["owned_records"] == 3
    assert v["years"]["1999"]["records_with_several_holders"] == 2
    assert v["years"]["2000"]["owned_records"] == 2
    assert v["totals"] == {"valid": 7, "distinct": 5, "owned": 5, "multi_holder": 2}
    for year, path in (("1999", p99), ("2000", p00)):
        assert v["years"][year]["input_sha256"] == hashlib.sha256(open(path, "rb").read()).hexdigest()
        assert v["years"][year]["output_sha256"] == hashlib.sha256(
            open(path[:-3] + "_owned.nc", "rb").read()).hexdigest()
    assert v["worked_example"] == {"year": "1999", "time_index": 0, "lon": 1.0, "lat": 0.0,
                                   "holders": [{"system": 4, "valid_records": 3},
                                               {"system": 9, "valid_records": 2}],
                                   "owner_system": 4}


def test_the_observation_driver_refuses_a_stray_file_and_a_mismatch(tmp_path, monkeypatch):
    """A file that is not a filtered year is a refusal before anything is written (a
    review's planted backup once made a year count twice in the totals and once in the
    entries), and an owned count that differs from the distinct count is a refusal that
    leaves no companion and no summary behind."""
    drv = _load_observation_driver()
    tracks = tmp_path / "tracks"
    tracks.mkdir()
    p = str(tracks / "ERA5_AEW_tracks_atlantic_1999.nc")
    write_year(p, [[1.0, 2.0], [1.0, NAN]], [[0.0, 0.0], [0.0, NAN]], [[7, 7], [7, NAN]],
               ["N/A", "N/A"])
    summary = str(tmp_path / "view.json")
    # Two strays, one at a time: the same year (the review's planted backup) and a
    # DIFFERENT year, so the refusal is bound to the filename rule itself and not
    # only to a year appearing twice.
    for name in ("backup_1999.nc", "backup_2003.nc"):
        stray = tracks / name
        stray.write_bytes(open(p, "rb").read())
        assert drv.main(["--tracks", str(tracks), "--summary", summary]) == 2
        assert not os.path.exists(p[:-3] + "_owned.nc") and not os.path.exists(summary)
        stray.unlink()

    real = drv.Q.observation_owners

    def wrong(lon, lat, keep, system):
        v = real(lon, lat, keep, system)
        v["owned"][:] = True                 # every holder marked: owned exceeds distinct
        return v

    monkeypatch.setattr(drv.Q, "observation_owners", wrong)
    assert drv.main(["--tracks", str(tracks), "--summary", summary]) == 2
    assert not os.path.exists(p[:-3] + "_owned.nc") and not os.path.exists(summary)
    monkeypatch.setattr(drv.Q, "observation_owners", real)
    assert drv.main(["--tracks", str(tracks), "--summary", summary]) == 0
    assert os.path.exists(p[:-3] + "_owned.nc") and os.path.exists(summary)
