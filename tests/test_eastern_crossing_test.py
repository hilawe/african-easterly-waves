"""The crossing test must extend the band means, find minima around gaps, follow histories
across 40 E and count record meetings and precedence as section 2 of
`EASTERN_CROSSING_TEST_2026-10-19.md` declares, with the adequacy checks of section 3."""
import datetime as dt
import importlib.util
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))


def _load():
    spec = importlib.util.spec_from_file_location("crossing_test_under_test", os.path.join(ROOT, "scripts", "eastern_crossing_test.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_gap_rule_splits_a_profile_and_never_places_a_minimum_at_a_gap():
    M = _load()
    lons = np.arange(0.0, 6.0, 0.5)                               # 12 cells
    row = np.array([0, -1200, -2400, -1200, 0, 0, -3600, 0, -1600, -2800, -1600, 0], dtype=np.int64)
    gap = np.zeros(12, bool); gap[6] = True                         # the deepest cell is a gap
    got = M.segment_minima([row], [400], lons, gap, -4.0)[0]
    assert [m["lon"] for m in got] == [1.0, 4.5], got
    assert got[1]["prominence"] == 7.0, "4.5 E's run (-4, -7, -4) is bounded by its segment's zeros, the gap acting as an edge"
    assert M.segment_minima([row], [400], lons, None, -4.0)[0][1]["lon"] == 3.0, "without the gap the deepest cell is the minimum"
    assert M.segment_minima([None], [400], lons, gap, -4.0) == [None]
    exact2 = np.array([0, 0, -1400, -2200, -1400, 0, 0, 0, 0, 0, 0, 0], dtype=np.int64)       # -5.5 m/s between two -3.5 m/s cells, prominence exactly 2
    assert [(m["lon"], m["prominence"]) for m in M.segment_minima([exact2], [400], lons, None, -4.0)[0]] == [(1.5, 2.0)]
    short = np.zeros(12, bool); short[2] = True; short[5] = True     # a run of two cells between gaps holds no minimum
    assert all(m["lon"] not in (1.5, 2.0) for m in M.segment_minima([row], [400], lons, short, -4.0)[0])


def test_a_column_missing_any_sample_is_a_gap_for_the_season():
    M = _load()
    lons = [0.0, 0.5, 1.0]
    recs = {}
    for m, nd in ((6, 30), (7, 31), (8, 31), (9, 30)):
        times = [f"1990-{m:02d}-{d:02d}T{h}" for d in range(1, nd + 1) for h in ("00", "06", "12", "18")]
        vals = [[-1.0, -2.0, -3.0] for _ in times]
        if m == 8:
            vals[5][1] = None
        recs[m] = {"year": 1990, "month": m, "lons": lons, "times": times, "bands": {"b": [0, 1]}, "band_means_700": {"b": [[np.nan if v is None else v for v in r] for r in vals]}}
    dates, H, gap = M.season_hundredths_with_gaps(recs, "b")
    assert gap.tolist() == [False, True, False] and len(dates) == 122
    assert H[0].tolist() == [-400, 0, -1200] and H[100].tolist() == [-400, 0, -1200], "the gap column is zeroed in every month, the others are the exact daily sums"


def test_the_extended_band_means_follow_the_renderers_rule_and_are_checked():
    M = _load()
    lat = np.arange(35.0, 4.9, -0.5); lons = M.XLONS
    rng = np.random.default_rng(3)
    v = rng.normal(size=(8, len(lat), len(lons))); sp = np.full(v.shape, 101325.0); sp[:, 20:24, 10:12] = 60000.0
    times = [dt.datetime(1990, 6, 1) + dt.timedelta(hours=6 * k) for k in range(8)]
    Mo = {"year": 1990, "month": 6, "lat": lat, "lon": lons, "v700": v, "sp": sp, "times": times, "files": {}}
    rec = M.xband_record(Mo)
    rows = M.V.band_rows(lat); ref = M.V.band_means(v, sp, 700, rows)
    for b in M.V.BANDS:
        assert np.array_equal(np.array(rec["band_means_700"][b], float), np.round(ref[b]["series"], 2), equal_nan=True)
    assert rec["generated_by"]["sha256_at_load"] == M.SCRIPT_SHA256_AT_LOAD and len(rec["lons"]) == 171
    cl = np.arange(20.0, 75.01, 0.5); cols = np.flatnonzero(lons >= 20.0)
    crec = {"lons": cl.tolist(), "times": rec["times"], "bands": rec["bands"], "band_means_700": {b: np.array(rec["band_means_700"][b], float)[:, cols].tolist() for b in rec["bands"]}}
    assert M.check_against_climatology(rec, crec)
    bad = json.loads(json.dumps(crec)); bad["band_means_700"]["15 to 19.5 N"][3][7] += 0.01
    assert not M.check_against_climatology(rec, bad)


def _hist(year, band, members, first=None, last=None):
    return {"year": year, "band": band, "members": members, "first": first if first is not None else members[0][1], "last": last if last is not None else members[-1][1], "days": len(members)}


def test_meetings_count_positions_on_the_same_day_within_their_tolerances():
    M = _load()
    h = _hist(1990, "15 to 19.5 N", [("1990-07-10", 46.0), ("1990-07-11", 43.0), ("1990-07-12", 40.0)])
    by = {"1990-07-12": [(("c", 1), 17.0, 38.0), (("c", 2), 22.0, 40.0), (("c", 3), 22.01, 40.0)],
          "1990-07-10": [(("c", 4), 17.0, 46.0)], "1991-07-12": [(("c", 5), 15.0, 41.0)]}
    assert M.meets(h, by) == {("c", 1), ("c", 2)}, "2 degrees of longitude and the band's latitudes padded by 2.5 (12.5 to 22) are inclusive, and a member east of 42 E never meets"
    assert M.meets(h, by, year=1991) == {("c", 5)} and M.shift_date("1990-07-12", 1991) == "1991-07-12"
    by4 = {"1990-07-12": [(("c", 6), 17.0, 44.0), (("c", 7), 17.0, 44.01)]}
    assert M.meets(h, by4, lon_tol=4.0) == {("c", 6)}, "4 degrees inclusive under the sensitivity"
    assert M.lat_window("10 to 14.5 N") == (7.5, 17.0)


def test_precedence_needs_the_history_at_or_east_of_the_first_position_on_its_day_or_the_day_before():
    M = _load()
    idx = {(1990, "15 to 19.5 N", "07-11"): [41.0], (1990, "15 to 19.5 N", "07-14"): [39.5]}
    t = (dt.datetime(1990, 7, 12, 6), 16.0, 40.0, -20.0)
    assert M.preceded(t, idx) is True, "41 E on the day before is within 2 degrees, east of 40 E"
    assert M.preceded((dt.datetime(1990, 7, 13, 6), 16.0, 40.0, -20.0), idx) is False, "two days before is too early"
    assert M.preceded((dt.datetime(1990, 7, 14, 0), 16.0, 40.0, -20.0), idx) is False, "39.5 E is west of the first position"
    assert M.preceded((dt.datetime(1990, 7, 12, 6), 16.0, 39.0, -20.0), idx) is True and M.preceded((dt.datetime(1990, 7, 12, 6), 16.0, 38.9, -20.0), idx) is False, "2 degrees inclusive"
    assert M.preceded((dt.datetime(1990, 7, 12, 6), 30.0, 40.0, -20.0), idx) is False, "a first latitude outside every padded band"
    assert M.preceded((dt.datetime(1991, 7, 12, 6), 16.0, 40.0, -20.0), idx, year=1990) is True
    assert M.preceded((dt.datetime(1990, 7, 13, 6), 16.0, 40.0, -20.0), idx, back_days=2) is True, "two days before counts under the sensitivity"
    assert M.preceded((dt.datetime(1990, 7, 12, 6), 16.0, 36.0, -20.0), idx, lon_tol=5.0) is True and M.preceded((dt.datetime(1990, 7, 12, 6), 16.0, 35.9, -20.0), idx, lon_tol=5.0) is False
    assert M.SENSITIVITY == {"meet_lon_tol": 4.0, "precede_lon_tol": 5.0, "precede_back_days": 2}
    tracks = {"a": (dt.datetime(1990, 7, 1), 16.0, 31.0, 0.0), "b": (dt.datetime(1990, 7, 1), 16.0, 29.5, 0.0), "c": (dt.datetime(1990, 5, 31), 16.0, 35.0, 0.0),
              "d": (dt.datetime(2005, 7, 1), 16.0, 35.0, 0.0), "e": (dt.datetime(1990, 7, 1), 33.0, 35.0, 0.0)}
    assert sorted(M.eligible_tracks(tracks)) == ["a"]


def test_the_archive_shuffle_never_pairs_a_history_with_its_own_season():
    M = _load()
    dates = [f"1990-07-{d:02d}" for d in range(10, 16)]
    h1 = {"days": [0, 1, 2, 3, 4, 5], "lons": [52.0, 51.0, 50.0, 49.0, 40.0, 39.0], "depths": [-5.0] * 6}     # -1 per day east of 40 E, -1 at or west of it
    h2 = {"days": [0, 1, 2, 3, 4, 5], "lons": [50.0, 47.0, 44.0, 41.0, 38.0, 35.0], "depths": [-5.0] * 6}     # ends exactly at 35 E, in the next band
    h3 = {"days": [0, 1, 2, 3, 4, 5], "lons": [44.0, 41.0, 38.0, 35.0, 32.0, 29.0], "depths": [-5.0] * 6}     # starts at 44 E, not eastern-origin
    hist = {}
    for y in M.YEARS:
        own = (y == 1990)
        for b, hs in (("15 to 19.5 N", [h1, h3]), ("20 to 24.5 N", [h2])):
            per = {M.HI.combo(r, l): ([(h, "westward") for h in hs] if own else []) for r, l in M.HI.ORDER}
            hist[(y, b)] = ([d.replace("1990", str(y)) for d in dates], per)
    by = {"1990-07-14": [(("c", 1), 16.0, 40.0)]}
    tracks = {("c", 1): (dt.datetime(1990, 7, 14), 16.0, 40.0, -20.0)}
    tabs = M.archive_tables(hist, {"catalog": (by, tracks)})
    r = tabs["RB1|LN"]["records"]["catalog"]
    assert r["eligible_histories"] == 2 and r["meeting"] == 1 and r["shuffled_max"] == 0 and r["shuffled_mean"] == 0.0, "the 20 to 24.5 N history is outside the point's padded latitudes"
    assert r["by_band"]["20 to 24.5 N"] == {"eligible": 1, "meeting": 0}
    assert r["eastern_tracks"] == 1 and r["preceded"] == 1 and r["preceded_shuffled_max"] == 0 and r["met_tracks_reaching_15W"] == 1
    assert r["sensitivity"]["meeting"] == 1 and r["sensitivity"]["meeting_shuffled_max"] == 0 and r["sensitivity"]["preceded"] == 1 and r["eastern_track_first_lon_q"] == [40.0, 40.0, 40.0] and r["eastern_track_first_lat_q"] == [16.0, 16.0, 16.0]
    b1 = tabs["RB1|LN"]["bands"]["15 to 19.5 N"]; b2 = tabs["RB1|LN"]["bands"]["20 to 24.5 N"]
    assert b1["eastern_origin_westward"] == 1, "the history starting at 44 E is not eastern-origin"
    assert b1["reach"]["40"] == 1 and b1["reach"]["35"] == 0 and b2["reach"]["35"] == 1 and b2["reach"]["30"] == 0, "ending exactly at 35 E reaches 35 E"
    assert b1["crossing_with_both_sides"] == 1 and b1["median_speed_east_of_40"] == -1.0 and b1["median_speed_west_of_40"] == -1.0, "40 E itself is on the western side"
    assert b2["median_speed_east_of_40"] == -3.0 and b2["median_speed_west_of_40"] == -3.0



def _season_dates(y):
    d0 = dt.date(y, 6, 1)
    return [(d0 + dt.timedelta(days=i)).strftime("%Y-%m-%d") for i in range(122)]


def test_reach_counts_any_member_and_the_shift_totals_use_every_other_season():
    """Review fold: a history that crosses a longitude and returns east still reaches it, and
    the shuffle and the observability counts are checked with two source seasons, one of
    them 1983, against hand-summed totals for every shift."""
    M = _load()
    keys = [M.HI.combo(r, l) for r, l in M.HI.ORDER]
    def mk(y, md_lons):
        dates = _season_dates(y)
        return {"days": [dates.index(f"{y}-{md}") for md, _ in md_lons], "lons": [x for _, x in md_lons], "depths": [-5.0] * len(md_lons)}
    A_ = mk(1983, [("07-08", 47.0), ("07-09", 44.0), ("07-10", 41.0)])
    B_ = mk(1990, [("08-03", 46.0), ("08-04", 43.0), ("08-05", 40.0)])
    C_ = mk(1995, [("07-01", 52.0), ("07-02", 49.0), ("07-03", 46.0), ("07-04", 43.0), ("07-05", 39.5), ("07-06", 41.0)])     # reaches 40 E and returns east
    hist = {}
    for y, b, h in ((1983, "15 to 19.5 N", A_), (1990, "15 to 19.5 N", B_), (1995, "25 to 29.5 N", C_)):
        hist[(y, b)] = (_season_dates(y), {k: [(h, "westward")] for k in keys})
    pts = {"1983-07-10": ("c", 1), "1984-07-10": ("c", 2), "1990-07-10": ("c", 3), "1983-08-05": ("c", 4), "1991-08-05": ("c", 5)}
    by = {d: [(k, 16.0, 41.0 if d.endswith("07-10") else 40.0)] for d, k in pts.items()}
    tracks = {k: (dt.datetime(int(d[:4]), int(d[5:7]), int(d[8:])), 16.0, 20.0, -20.0) for d, k in pts.items()}     # west of 30 E, never eligible
    tracks[("c", 10)] = (dt.datetime(1984, 7, 10, 6), 16.0, 40.0, -20.0)     # preceded only with 1983's history
    tracks[("c", 11)] = (dt.datetime(1990, 8, 4, 6), 16.0, 39.0, -20.0)      # compatible in longitude with its own season's history, a day too early
    tracks[("c", 12)] = (dt.datetime(1990, 9, 15, 6), 16.0, 36.0, -20.0)     # compatible only within 5 degrees
    tracks[("c", 13)] = (dt.datetime(1995, 8, 1, 6), 16.0, 50.5, -20.0)      # only the 25 to 29.5 N history lies east within 2 degrees, outside its latitudes
    tracks[("c", 14)] = (dt.datetime(1990, 7, 20, 6), 16.0, 48.0, -20.0)     # its season's history lies 2 degrees west of it, never east
    tabs = M.archive_tables(hist, {"catalog": (by, tracks)})
    r = tabs["RB1|LN"]["records"]["catalog"]
    shift_meet = [0] * 23; shift_meet[0] = 2; shift_meet[6] = 1; shift_meet[16] = 1
    shift_prec = [0] * 23; shift_prec[22] = 1
    assert r["eligible_histories"] == 3 and r["meeting"] == 1 and r["shift_totals"]["meeting"] == shift_meet and r["shuffled_max"] == 2 and abs(r["shuffled_mean"] - 4 / 23) < 1e-12
    assert r["eastern_tracks"] == 5 and r["preceded"] == 0 and r["shift_totals"]["preceded"] == shift_prec
    assert r["sensitivity"]["meeting"] == 1 and r["shift_totals"]["sensitivity_meeting"] == shift_meet
    assert r["sensitivity"]["preceded"] == 1 and r["shift_totals"]["sensitivity_preceded"] == shift_prec, "43 E on 4 August is within 5 degrees of 39 E"
    assert r["longitude_compatible"] == {"same_season": 1, "any_season": 2, "same_season_sensitivity": 2, "any_season_sensitivity": 3}
    b15 = tabs["RB1|LN"]["bands"]["15 to 19.5 N"]; b25 = tabs["RB1|LN"]["bands"]["25 to 29.5 N"]
    assert b15["reach"]["42"] == 2 and b15["reach"]["40"] == 1
    assert b25["reach"]["42"] == 1 and b25["reach"]["40"] == 1 and b25["reach"]["35"] == 0, "the history reaches 39.5 E and ends at 41 E, so it reaches 40 E"
    text = M.archive_text(tabs, {}, {})
    assert "| RB1|LN | catalog | 1 | 3 | 0 | 23 | 1 | 3 | 1 | 1 |" in text, "observed counts with the shifts at or above them"
    assert "| RB1|LN | catalog | 5 | 1 | 2 | 2 | 3 |" in text and "any member lies at or west of it" in text


def test_qtrack_times_follow_the_files_units_and_the_western_coverage_is_complete():
    M = _load()
    w = M.decode_qtrack_times(np.array([0.0, 6.0]), "hours since 1990-07-01 00:00:00", "gregorian", 1990)
    assert w == [dt.datetime(1990, 7, 1, 0), dt.datetime(1990, 7, 1, 6)]
    assert M.decode_qtrack_times(np.array([1717200000.0]), "seconds since 1970-01-01", "gregorian", 2024) == [dt.datetime(2024, 6, 1)]
    for vals, units, year, msg in ((np.array([0.0, 6.0]), None, 1990, "no units"), (np.array([6.0, 6.0]), "hours since 1990-07-01", 1990, "strictly increase"),
                                   (np.array([0.0, 6.0]), "hours since 1990-07-01", 1991, "outside that season"), (np.ma.masked_array([0.0, 6.0], [False, True]), "hours since 1990-07-01", 1990, "missing value")):
        try:
            M.decode_qtrack_times(vals, units, "gregorian", year); assert False
        except SystemExit as ex:
            assert msg in str(ex), str(ex)
    paths = [f"/x/ERA5_AEW_tracks_atlantic_{y}.nc" for y in M.YEARS]
    M.check_western_coverage(paths, M.qtrack_year, "QTrack")
    assert M.catalog_year("/x/ERA-Int_ew_700hPa_1987_AFR.nc") == 1987
    for bad, msg in ((paths[1:], "missing [1983]"), (paths + [paths[3]], "duplicated [1986]"), (paths + ["/x/ERA5_AEW_tracks_atlantic_2005.nc"], "outside [2005]")):
        try:
            M.check_western_coverage(bad, M.qtrack_year, "QTrack"); assert False
        except SystemExit as ex:
            assert msg in str(ex), str(ex)


def test_the_western_files_are_one_per_retained_season(tmp_path):
    M = _load()
    cat, q = tmp_path / "cat", tmp_path / "q"
    cat.mkdir(); q.mkdir()
    for y in M.YEARS + [2005]:
        (cat / f"ERA-Int_ew_700hPa_{y}_AFR.nc").write_bytes(b"")
        (q / f"ERA5_AEW_tracks_atlantic_{y}.nc").write_bytes(b"")
    (q / "ERA5_AEW_tracks_atlantic_1990_owned.nc").write_bytes(b"")
    cp, qp = M.western_paths(str(cat / "ERA-Int_ew_700hPa_*_AFR.nc"), str(q))
    assert [M.catalog_year(p) for p in cp] == M.YEARS and [M.qtrack_year(p) for p in qp] == M.YEARS, "2005 and the owned copy are never read"
    for d, name, msg in ((cat, "ERA-Int_ew_700hPa_1990_AFR.nc", "catalog files"), (q, "ERA5_AEW_tracks_atlantic_1991.nc", "QTrack files")):
        (d / name).unlink()
        try:
            M.western_paths(str(cat / "ERA-Int_ew_700hPa_*_AFR.nc"), str(q)); assert False
        except SystemExit as ex:
            assert msg in str(ex) and "missing [" in str(ex), str(ex)
        (d / name).write_bytes(b"")

def test_the_construction_paths_and_scores():
    M = _load()
    p = M.path_positions(72.0, 35.0, M.speed_profile(6.0))
    xs = [p[k / 4.0] for k in range(140, 280)]
    steps = np.diff(xs)
    assert abs(steps[0] + 0.5) < 1e-9 and abs(min(steps) + 1.5) < 1e-6, "2 deg/day east of 45 E is 0.5 per six hours, 6 deg/day west of 35 E is 1.5"
    assert abs(M.speed_profile(6.0)(40.0) - 4.0) < 1e-12 and M.speed_profile(7.0)(30.0) == 7.0 and M.speed_profile(6.0)(50.0) == 2.0
    cases = {c[0]: c for c in M.construction_cases()}
    assert sorted(cases) == ["X1 constant 2 deg/day, depth 10", "X1 constant 2 deg/day, depth 5", "X2 rising to 6 deg/day, depth 10", "X2 rising to 6 deg/day, depth 5", "X3 rising to 7 deg/day, depth 10", "X4 hand-off at 42 and 38 E", "X5 hand-off on consecutive days, 41 E"]
    _, kind5, trans5, meta5 = cases["X5 hand-off on consecutive days, 41 E"]
    assert kind5 == "handoff" and meta5["added"] is True and min(trans5[1][0]) == 50.0 and trans5[1][0][50.0] == 41.0 and abs(trans5[1][0][50.25] - 39.5) < 1e-9, "the second appears at 41 E as the first vanishes and moves 6 degrees per day"
    _, kind, trans, meta = cases["X4 hand-off at 42 and 38 E"]
    (pa, da), (pb, db) = trans
    assert da(46.0, 0) == 10.0 and abs(da(44.0, 0) - 5.0) < 1e-12 and da(42.0, 0) == 0.0 and meta["vanish_day"] == 50.0 and min(pb) == 51.0 and pb[51.0] == 38.0
    center = np.full(122, np.nan); center[40:46] = [50.0, 46.0, 42.0, 36.0, 30.0, 24.0]
    mins = [[]] * 122
    hs = [{"days": [40, 41, 42, 43, 44, 45], "lons": [50.0, 46.0, 42.5, 36.0, 30.0, 24.0], "depths": [0] * 6}]
    r = M.score_passage(hs, ["westward"], center, mins)
    assert r == {"present_days": 6, "share": 1.0, "spanning": True}
    r2 = M.score_passage(hs, ["other"], center, mins)
    assert r2["share"] == 0.0 and r2["spanning"] is False
    ca = np.full(122, np.nan); ca[40:42] = [48.0, 46.0]; cb = np.full(122, np.nan); cb[43:45] = [32.0, 26.0]
    joined = {"days": [40, 41, 42, 43, 44], "lons": [48.0, 46.0, 40.0, 32.0, 26.0], "depths": [0] * 5}
    assert M.score_handoff([joined], ["westward"], ca, cb) == {"joined": True} and M.score_handoff([joined], ["other"], ca, cb) == {"joined": False}


def test_the_rule_and_the_refusals():
    M = _load()
    keys = [M.HI.combo(r, l) for r, l in M.HI.ORDER]
    res = {"X1 a": {"kind": "passage", **{k: {"present_days": 10, "share": 0.8, "spanning": True} for k in keys}},
           "X3 a": {"kind": "limit", **{k: {"present_days": 10, "share": 0.0, "spanning": False} for k in keys}},
           "X4 a": {"kind": "handoff", **{k: {"joined": True} for k in keys}}}
    assert all(v["passes"] for v in M.apply_rule(res).values()), "X3 and X4 are limits, not part of the rule"
    res["X1 a"]["RB9|LN"]["spanning"] = False; res["X1 a"]["RB1|LN"]["share"] = 0.79
    v = M.apply_rule(res)
    assert v["RB9|LN"]["passes"] is False and v["RB1|LN"]["passes"] is False and v["RB1|LP"]["passes"] is True
    recs = [{"year": y, "month": m, "lons": [0.0], "bands": {"b": [0, 1]}} for y in M.YEARS for m in (6, 7, 8, 9)]
    files = [f"/x/eval60_xband_{r['year']}_{r['month']:02d}.json.gz" for r in recs]
    M.check_xband_identity(files, recs)
    for bad_files, bad_recs, msg in ((files, recs[:-1] + [dict(recs[-1], year=1990)], "not the declared months"), (files[::-1], recs, "holds the record"), (files, recs[:5] + [dict(recs[5], lons=[0.5])] + recs[6:], "grid")):
        try:
            M.check_xband_identity(bad_files, bad_recs); assert False
        except SystemExit as ex:
            assert msg in str(ex), str(ex)
