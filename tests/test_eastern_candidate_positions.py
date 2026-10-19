"""The candidate-positions layer must list every interior local minimum at or below the
floor with its region and prominence as section 2 of
`EASTERN_CANDIDATE_POSITIONS_2026-10-18.md` declares, and the crosswalk must read
coincident, competing and deeper-outside from one day's longitudes only."""
import importlib.util
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))


def _load():
    spec = importlib.util.spec_from_file_location("candidate_positions_under_test", os.path.join(ROOT, "scripts", "eastern_candidate_positions.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_candidates_are_interior_minima_at_or_below_the_floor_with_region_and_prominence():
    M = _load()
    lons = np.arange(20.0, 75.01, 0.5)
    v = np.zeros(len(lons))
    v[lons == 50.0] = -10.0; v[lons == 49.5] = -8.0; v[lons == 50.5] = -8.0      # a candidate at 50 E
    v[lons == 52.0] = -6.0; v[lons == 51.5] = -5.0; v[lons == 52.5] = -5.0      # a second in the same region, the ridge between them at -5
    v[lons == 51.0] = -5.0
    v[lons == 60.0] = -3.9; v[lons == 59.5] = -2.0; v[lons == 60.5] = -2.0      # above the floor, not a candidate
    v[lons == 30.0] = -4.0; v[lons == 29.5] = -1.0; v[lons == 30.5] = -1.0      # exactly at the floor, a candidate
    v[0] = -20.0                                                                  # the box edge, never a candidate
    cs = M.candidates_on(v, lons)
    assert [c["lon"] for c in cs] == [30.0, 50.0, 52.0], cs
    c50 = cs[1]; c52 = cs[2]
    assert c50["region"] == {"lo": 49.5, "hi": 52.5, "width_deg": 3.0, "candidates_in_region": 2} and c52["region"]["candidates_in_region"] == 2
    assert abs(c50["prominence"] - 5.0) < 1e-12, "the shallower ridge is the one toward 52 E at -5, so 5 m/s below it"
    assert abs(c52["prominence"] - 1.0) < 1e-12, "from -6 up to the -5 ridge"
    assert abs(cs[0]["prominence"] - 3.0) < 1e-12, "no neighboring candidate to the west, so the run's own end at -1 bounds it"
    flat = np.zeros(len(lons)); flat[(lons >= 40.0) & (lons <= 41.0)] = -7.0
    assert [c["lon"] for c in M.candidates_on(flat, lons)] == [40.0], "a flat run counts once, at its first cell"
    for vals in ([np.nan, -5, -6, -5, 0], [0, -5, -6, -5, np.nan], [0, -6, -5, np.nan, -5, -8, 0], [np.nan] * 5):
        try:
            M.candidates_on(np.array(vals, float), np.arange(len(vals), dtype=float), context=" (test)"); assert False, vals
        except SystemExit as ex:
            assert "missing cells" in str(ex) and "(test)" in str(ex), "a daily series with any missing cell is refused, in either orientation, with the context named (review of 2026-10-18, F6)"
    mirrored = np.zeros(len(lons)); mirrored[lons == 50.0] = -10.0; mirrored[lons == 49.5] = -8.0; mirrored[lons == 50.5] = -8.0
    mirrored[lons == 52.0] = -6.0; mirrored[lons == 51.5] = -5.0; mirrored[lons == 52.5] = -5.0; mirrored[lons == 51.0] = -5.0
    rev = M.candidates_on(mirrored[::-1], lons)
    assert sorted(round(c["prominence"], 9) for c in rev) == sorted(round(c["prominence"], 9) for c in M.candidates_on(mirrored, lons)), "prominences are the same under a west-east reflection"


def test_crosswalk_reads_coincident_competing_and_deeper_outside_from_one_day():
    M = _load()
    def cand(lon, depth, prominence, width=3.0):
        return {"lon": lon, "depth": depth, "region": {"lo": lon - width / 2, "hi": lon + width / 2, "width_deg": width, "candidates_in_region": 1}, "prominence": prominence}
    lay = {(2000, 7): {"b": {"2000-07-01": [cand(50.0, -10.0, 6.0), cand(53.0, -6.0, 1.5), cand(62.0, -12.0, 8.0)],
                             "2000-07-02": [cand(49.0, -9.0, 5.0)]}}}
    entry = {"id": "T", "year": 2000, "month": 7, "band": "b", "days": [
        {"date": "2000-07-01", "anchored": True, "anchor_lon": 50.0, "v": -10.0, "lo": 45.0, "hi": 55.0, "edge": False},
        {"date": "2000-07-02", "anchored": True, "anchor_lon": 49.5, "v": -8.0, "lo": 44.5, "hi": 54.5, "edge": True},
        {"date": "2000-07-03", "anchored": False, "anchor_lon": None, "v": None, "lo": 44.0, "hi": 54.0, "edge": False}]}
    xw = M.crosswalk([entry], lay)
    r = xw[0]
    assert r["anchored_days"] == 2
    d1, d2 = r["days"]
    assert d1 == {"date": "2000-07-01", "anchor_lon": 50.0, "anchor_v": -10.0, "edge": False, "coincident": True, "competing": True, "deeper_outside": True, "candidates_inside": 2, "candidates_on_day": 3, "competing_at_prominence": {"2": False, "4": False}}, d1
    assert d1["competing"] and not d1["competing_at_prominence"]["2"], "the competitor at 53 E has prominence 1.5, below the sensitivity floor of 2" 
    assert d2["coincident"] is False and d2["competing"] is True and d2["deeper_outside"] is False and d2["candidates_inside"] == 1, "an anchor that is not a candidate, with the real candidate inside its corridor"
    assert r["coincident"] == 1 and r["competing"] == 2 and r["deeper_outside"] == 1 and not r["all_coincident_never_competing"] and r["any_deeper_outside"]
    assert r["competing_at_prominence"] == {"2": 1, "4": 1}, "on the second day the real candidate at 49 E (prominence 5) competes with the non-candidate anchor" 
    clean = dict(entry, days=[{"date": "2000-07-02", "anchored": True, "anchor_lon": 49.0, "v": -9.0, "lo": 44.0, "hi": 54.0, "edge": False}])
    assert M.crosswalk([clean], lay)[0]["all_coincident_never_competing"]
    s = M.summarize({(2000, 7): {"b": lay[(2000, 7)]["b"]}}, xw)
    v = s["by_month_band"]["2000-07|b"]
    assert v["band_days"] == 2 and v["candidates"] == 4 and v["max_crowdedness"] == 3 and v["share_two_or_more"] == 0.5 and v["share_none"] == 0.0
    assert v["crowdedness_at_prominence"]["2"] == {"mean": 1.5, "share_two_or_more": 0.5} and v["crowdedness_at_prominence"]["4"] == {"mean": 1.5, "share_two_or_more": 0.5}
    assert s["crosswalk"]["edge_days_deeper_outside"] == 0 and s["crosswalk"]["nonedge_days_deeper_outside"] == 1
    assert s["crosswalk"]["competing_days_at_prominence"] == {"2": 1, "4": 1} and s["crosswalk"]["entries_any_competing_at_prominence"] == {"2": 1, "4": 1}
