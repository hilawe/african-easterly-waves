"""The two-sided look-back's rules (EASTERN_TWO_SIDED_LOOKBACK_2026-10-19.md, sections 2 and 3)."""

import datetime as dt
import importlib.util
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load():
    spec = importlib.util.spec_from_file_location("two_sided_under_test", os.path.join(ROOT, "scripts", "eastern_two_sided_lookback.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_populations():
    M = _load()
    tr = {("q", "f", 1): (dt.datetime(1990, 7, 4, 18), 9.0, 20.0, 0.0), ("q", "f", 2): (dt.datetime(1990, 7, 4), 9.0, 29.99, 0.0), ("q", "f", 3): (dt.datetime(1990, 7, 4), 9.0, 30.0, 0.0),
          ("q", "f", 4): (dt.datetime(1990, 7, 4), 9.0, 19.99, 0.0), ("q", "f", 5): (dt.datetime(1990, 10, 1), 9.0, 25.0, 0.0), ("q", "f", 6): (dt.datetime(2005, 7, 4), 9.0, 25.0, 0.0),
          ("q", "f", 7): (dt.datetime(1990, 7, 4), 7.4, 25.0, 0.0), ("q", "f", 8): (dt.datetime(1990, 7, 4), 7.5, 25.0, 0.0)}
    new = M.population(tr, 20.0, 30.0)
    assert sorted(tuple(t["key"]) for t in new) == [("q", "f", 1), ("q", "f", 2), ("q", "f", 8)], "20 E inclusive, 30 E exclusive, June to September, retained seasons, 7.5 N inclusive"
    assert [tuple(t["key"]) for t in M.population(tr, 30.0)] == [("q", "f", 3)]
    t1 = [t for t in new if t["key"] == ["q", "f", 1]][0]
    assert t1["day"] == 33 and t1["x0"] == 20.0 and t1["band"] == "10 to 14.5 N"


def test_track_values_reach_35_degrees_east_of_the_start():
    M = _load()
    lons = M.X.XLONS
    F = np.tile(lons, (122, 1)).astype(float)
    v = M.track_values(F, 10, 40.0)
    assert v.shape == (8, 101) and M.DREL[-1] == 35.0
    assert np.isfinite(v[M.LAGS.index(0), M.DREL <= 35.0]).all() and v[M.LAGS.index(0), -1] == 75.0
    v2 = M.track_values(F, 10, 45.0)
    assert np.isnan(v2[M.LAGS.index(0), M.DREL > 30.0]).all() and np.isnan(M.track_values(F, 3, 20.0)[M.LAGS.index(-5)]).all()
    v0 = M.track_values(F, 10, 0.0)
    assert np.isnan(v0[M.LAGS.index(0), M.DREL < -10.0]).all() and v0[M.LAGS.index(0), int(np.flatnonzero(M.DREL == -10.0)[0])] == -10.0, "west of 10 W is missing, never wrapped"
    fields = {(y, "b"): {"RB1": np.full((122, len(lons)), float(i))} for i, y in enumerate(M.YEARS)}
    tr = [{"year": 1983, "day": 50, "x0": 20.0, "band": "b"}, {"year": 2007, "day": 50, "x0": 20.0, "band": "b"}]
    assert M.track_stack(fields, tr, "RB1", shift=1)[:, M.LAGS.index(0), 0].tolist() == [1.0, 0.0], "1983 moves forward to 1984 and 2007 wraps to 1983"


def _C(spec, M, base=0.0):
    C = np.full((len(M.LAGS), len(M.DREL)), base)
    for k, (d, v) in spec.items():
        C[M.LAGS.index(k), int(np.flatnonzero(M.DREL == d)[0])] = v
    return C


def test_the_reading_and_its_mirror():
    M = _load()
    E = M.exact_from_values
    path = {-5: 20.0, -4: 16.0, -3: 12.0, -2: 8.0, -1: 4.0, 0: 0.0}
    C = _C({k: (d, -3.0) for k, d in path.items()}, M)
    shifts = [_C({k: (0.0, -1.0) for k in M.LAGS}, M)] * 23
    r = M.reading(E(C), [E(c) for c in shifts])
    assert r["approach"] and r["registered"] and r["lags"][-5]["location"] == 20.0 and r["parts"]["west_at_least_7p5"]
    assert M.reading(E(_C({**{k: (d, -3.0) for k, d in path.items()}, -5: (7.5, -3.0), -4: (7.0, -3.0), -3: (6.0, -3.0), -2: (4.0, -3.0)}, M)), [E(c) for c in shifts])["approach"] is True, "exactly 7.5 degrees passes"
    assert M.reading(E(_C({**{k: (d, -3.0) for k, d in path.items()}, -5: (7.0, -3.0), -4: (6.5, -3.0), -3: (6.0, -3.0), -2: (4.0, -3.0)}, M)), [E(c) for c in shifts])["parts"]["west_at_least_7p5"] is False
    assert M.reading(E(_C({**{k: (d, -3.0) for k, d in path.items()}, -4: (21.0, -3.0)}, M)), [E(c) for c in shifts])["parts"]["not_moving_east"] is False
    assert M.reading(E(_C({**{k: (d, -3.0) for k, d in path.items()}, -3: (16.0, -3.0)}, M)), [E(c) for c in shifts])["approach"] is True, "a stall does not break it"
    deep = [_C({k: (0.0, -1.0) for k in M.LAGS}, M)] * 22 + [_C({**{k: (0.0, -1.0) for k in M.LAGS}, -5: (0.0, -3.0)}, M)]
    assert M.reading(E(C), [E(c) for c in deep])["lags"][-5]["rank"] == 1 and M.reading(E(C), [E(c) for c in deep])["approach"] is False, "a shift at the same depth counts against the observed"
    P = _C({k: (d, 3.0) for k, d in path.items()}, M)
    pshifts = [_C({k: (0.0, 1.0) for k in M.LAGS}, M)] * 22 + [_C({**{k: (0.0, 1.0) for k in M.LAGS}, -4: (5.0, 3.0)}, M)]
    rm = M.flank_values(M.reading(E(-P), [E(-c) for c in pshifts]), -1.0)
    assert rm["lags"][-5]["location"] == 20.0 and rm["lags"][-5]["value"] == 3.0 and rm["lags"][-4]["rank"] == 1 and rm["approach"] is False, "the maximum's rank counts shifts at or above it"
    assert rm["lags"][-4]["shift_values"][-1] == 3.0 and rm["lags"][-4]["shift_values"][0] == 1.0
    weak0 = _C({**{k: (d, -3.0) for k, d in path.items()}, 0: (0.0, -0.5)}, M)
    assert M.reading(E(weak0), [E(c) for c in shifts])["registered"] is False and M.reading(E(weak0), [E(c) for c in shifts])["lags"][0]["rank"] == 23, "a start day weaker than every shift is not registered"
    gap = [_C({k: (0.0, -1.0) for k in M.LAGS}, M)] * 22 + [np.full((len(M.LAGS), len(M.DREL)), np.nan)]
    rg = M.reading(E(C), [E(c) for c in gap])
    assert rg["lags"][0]["rank"] is None and rg["lags"][0]["valid_shifts"] == 22 and rg["registered"] is False and rg["approach"] is False, "a missing shift leaves the rank unavailable and failing"
    tie = _C({0: (5.0, -2.0), -1: (6.0, -1.0)}, M); tie[M.LAGS.index(0), int(np.flatnonzero(M.DREL == -3.0)[0])] = -2.0
    assert M.reading(E(tie), [E(c) for c in shifts])["lags"][0]["location"] == -3.0, "the westernmost location on a tie"


def test_contributions_by_flank():
    M = _load()
    st = np.zeros((3, len(M.LAGS), len(M.DREL)))
    i = M.LAGS.index(-2); j = lambda d: int(np.flatnonzero(M.DREL == d)[0])
    st[0, i, j(10.0)] = 2.0; st[1, i, j(11.0)] = 2.5; st[2, i, j(11.5)] = 9.0
    assert M.contribution_count(-st, 10.0, -2) == 2, "+2 inclusive in the maximum flank, 1 degree inclusive"
    assert M.contribution_count(st, 10.0, -2) == 0 and M.contribution_count(st, None, -2) is None


def test_look_back_sees_both_flanks_of_a_constructed_wave():
    M = _load()
    lons = M.X.XLONS
    fields = {(y, "10 to 14.5 N"): {"RB1": np.zeros((122, len(lons))), "RB9": np.zeros((122, len(lons)))} for y in M.YEARS}
    for d in range(122):
        trough = 25.0 + 4.0 * (60 - d)
        fields[(1990, "10 to 14.5 N")]["RB1"][d] = np.rint(M.two_sided(lons, trough) * 372000) / 372000
    tracks = [{"year": 1990, "day": 60, "x0": 25.0, "band": "10 to 14.5 N"}]
    r = M.look_back(fields, tracks, "RB1")
    mn, mx = r["flanks"]["minimum"]["reading"], r["flanks"]["maximum"]["reading"]
    assert mn["approach"] and mx["approach"] and mn["registered"] and mx["registered"]
    assert mn["lags"][0]["location"] < 0.0 < mx["lags"][0]["location"], "northerly west of the trough, southerly east of it"
    assert abs((mx["lags"][-5]["location"] - mx["lags"][0]["location"]) - 20.0) <= 0.5 and mx["lags"][0]["value"] > 0.0 > mn["lags"][0]["value"]
    assert r["flanks"]["maximum"]["contributions"][-3]["observed"] == 1 and r["flanks"]["minimum"]["contributions"][-3]["observed"] == 1
    assert r["tracks"] == 1 and len(r["shift_composites"]) == 23


def test_the_constructions_place_their_features_as_declared():
    M = _load()
    x = M.X.XLONS
    w = M.two_sided(x, 40.0)
    assert w[np.flatnonzero(x == 40.0)[0]] == 0.0 and w[np.flatnonzero(x == 36.0)[0]] < -3.0 and w[np.flatnonzero(x == 44.0)[0]] > 3.0, "northerly west of the trough, southerly east"
    assert abs(w[np.flatnonzero(x == 57.5)[0]]) < 0.1 and abs(w[np.flatnonzero(x == 22.5)[0]]) < 0.1, "a 6-degree envelope leaves almost nothing 17.5 degrees away"
    days, _ = M.construction_days(0, "T1")
    assert len(days) == 4 and all(30 <= d <= 110 for d in days) and days == M.construction_days(0, "T7")[0]
    for case, speed in (("T1", 4.5), ("T2", 2.0), ("T3", 6.0)):
        _, tracks, centers = M.construction_season(case, 2)
        d2 = M.construction_days(2, case)[0]
        assert len(centers) == 4 and all(t["x0"] == 25.0 for t in tracks)
        for c, d in zip(centers, d2):
            assert abs(c[d] - 25.0) < 0.02 and abs(c[d - 1] - (25.0 + speed)) < 0.05
    _, _, c5 = M.construction_season("T5", 3)
    d5 = M.construction_days(3, "T5")[0]
    assert all(np.isnan(c[d - 1]) and abs(c[d] - (25.0 - 4.5 * 0.375)) < 0.02 for c, d in zip(c5, d5)), "absent the day before, at 25 E as the start day begins"
    assert len(M.construction_season("T8", 1)[2]) == 2 and M.construction_season("T7", 1)[2] == [] and M.construction_season("T6", 1)[2] == []
    sp = M.track_speeds(4, "T9")
    assert len(sp) == 4 and all(2.0 <= s <= 6.0 for s in sp) and len(set(sp)) == 4 and sp != M.track_speeds(5, "T9")
    _, _, c9 = M.construction_season("T9", 4)
    d9 = M.construction_days(4, "T9")[0]
    assert all(abs(c[d] - 25.0) < 0.02 and abs(c[d - 1] - (25.0 + s)) < 0.05 for c, d, s in zip(c9, d9, sp))
    d6, ph6 = M.construction_days(6, "T6")
    assert len(d6) == 4 and all(2 * np.pi / 14.0 * 5.0 * np.cos(2 * np.pi * (d + 0.375) / 14.0 + ph6) < -2.1 for d in d6), "T6 starts at its fastest westward wobble"
    H1, _, _ = M.construction_season("T1", 5); H4, _, _ = M.construction_season("T4", 5); H6, _, _ = M.construction_season("T6", 5); H7, _, _ = M.construction_season("T7", 5)
    assert not np.array_equal(H1, H4) and not np.array_equal(H6, H7) and not np.array_equal(H7, M.construction_season("T7", 6)[0])


def test_the_rule_the_verdicts_the_rows_and_the_refusal(tmp_path):
    M = _load()
    def rd(appr, reg=True):
        return {"reading": {"approach": appr, "registered": reg}}
    res = {c: {rep: {"flanks": {f: rd(c in ("T1", "T2", "T3", "T8", "T9")) for f in M.FLANKS}} for rep in M.REPS} for c in M.CASES}
    assert M.apply_rule(res) == {"minimum": {"adequate": True, "failures": []}, "maximum": {"adequate": True, "failures": []}}
    res["T6"]["RB9"]["flanks"]["maximum"]["reading"]["approach"] = True; res["T2"]["RB1"]["flanks"]["minimum"]["reading"]["approach"] = False; res["T5"]["RB1"]["flanks"]["maximum"]["reading"]["registered"] = False
    v = M.apply_rule(res)
    assert v["minimum"] == {"adequate": False, "failures": ["T2 under RB1 does not approach"]}
    assert v["maximum"]["adequate"] is False and sorted(v["maximum"]["failures"]) == ["T5 under RB1 does not register its start", "T6 under RB9 approaches"]
    per = {rep: {"flanks": {"minimum": rd(True), "maximum": rd(rep == "RB1")}} for rep in M.REPS}
    assert M.flank_verdict(per, "minimum") == "SHOWN" and M.flank_verdict(per, "maximum") == "under RB1 only"
    assert M.RULE == {"shown": ("T1", "T2"), "not_shown": ("T5", "T6", "T7"), "start_registered": ("T1", "T2", "T5")}
    lags = {k: {"value": float(k), "location": abs(float(k)), "rank": 0} for k in M.LAGS}
    row = M.reading_row("T1", "RB1", "maximum", {"lags": lags, "registered": True, "approach": False})
    assert row == "| T1 | RB1 | maximum | 5 (0), 4 (0), 3 (0), 2 (0), 1 (0), 0 (0) | -3.00, -1.00, 0.00 | yes | no |"
    good = {c: {rep: {"flanks": {f: rd(c in ("T1", "T2", "T3", "T8", "T9")) for f in M.FLANKS}} for rep in M.REPS} for c in M.CASES}
    good["T2"]["RB1"]["flanks"]["minimum"]["reading"]["approach"] = False
    v = M.apply_rule(good)
    rerun = lambda: json.loads(json.dumps(good))
    p = tmp_path / "c.json"
    stale = json.loads(json.dumps(good)); stale["T6"]["RB1"]["flanks"]["maximum"]["reading"]["approach"] = True
    for rec, run, msg in (({"generated_by": {"sha256": "x"}, "results": good, "verdict": v}, rerun, "different version"),
                          ({"generated_by": {"sha256": M.SCRIPT_SHA256_AT_LOAD}, "results": good, "verdict": v}, lambda: stale, "differs from a fresh run"),
                          ({"generated_by": {"sha256": M.SCRIPT_SHA256_AT_LOAD}, "results": good, "verdict": {"minimum": {"adequate": True, "failures": []}, "maximum": v["maximum"]}}, rerun, "stored summary disagrees")):
        p.write_text(json.dumps(rec))
        try:
            M.check_constructions(str(p), rerun=run); assert False
        except SystemExit as ex:
            assert msg in str(ex), str(ex)
    p.write_text(json.dumps({"generated_by": {"sha256": M.SCRIPT_SHA256_AT_LOAD}, "results": good, "verdict": v}))
    assert M.check_constructions(str(p), rerun=rerun) == {"minimum": False, "maximum": True}, "adequacy per flank, recomputed from the stored cases"


def _tie_fields(M, sign=1):
    """The review's construction, three tracks whose composite has an exact tie at lag -5
    between D = 7 and 7.5, the declared westernmost location giving 7 degrees of motion."""
    H = np.zeros((122, len(M.X.XLONS)), dtype=np.int64)
    col = lambda d: int(np.flatnonzero(M.X.XLONS == 25 + d)[0])
    tracks = []
    for i, day in enumerate((10, 40, 70)):
        tracks.append(dict(year=1983, day=day, x0=25.0, band=M.LB.LOW_BAND))
        H[day - 5, col(7.0)] = (-120, -80, -40)[i]
        H[day - 5, col(7.5)] = (-40, -80, -120)[i]
        for lag, d in ((-4, 6.0), (-3, 4.5), (-2, 3.0), (-1, 1.5), (0, 0.0)):
            H[day + lag, col(d)] = -80
        H[day + 10] = -H[day - 5:day + 1].sum(axis=0)
    H = sign * H
    fields = {(y, M.LB.LOW_BAND): {"RB1": M.LB.anomaly_values(H if y == 1983 else np.zeros_like(H), M.A.month_index(), "B1")} for y in M.YEARS}
    return fields, tracks


def test_exact_ties_resolve_westward_whatever_the_track_order():
    """Second review fold: the extremum is chosen from exact integer sums, so the westernmost
    location of an exact tie does not depend on the order the tracks are summed in."""
    M = _load()
    for sign, flank in ((1, "minimum"), (-1, "maximum")):
        fields, tracks = _tie_fields(M, sign)
        for tr in (tracks, list(reversed(tracks)), [tracks[1], tracks[0], tracks[2]]):
            rd = M.look_back(fields, tr, "RB1")["flanks"][flank]["reading"]
            assert rd["lags"][-5]["location"] == 7.0 and rd["approach"] is False and rd["parts"]["west_at_least_7p5"] is False, (flank, [t["day"] for t in tr])
            assert abs(rd["lags"][-5]["value"] - (-0.2 * sign)) < 1e-15
            full = M.look_back(fields, tr, "RB1")
            assert abs(full["composite"][M.LAGS.index(-5), int(np.flatnonzero(M.DREL == 7.0)[0])] - (-0.2 * sign)) < 1e-15 and full["n"][M.LAGS.index(-5), 0] == 3, "the stored composite is the mean over three tracks"
    e = M.exact_from_values(np.array([[-0.2, -0.2, -0.1] + [0.0] * (len(M.DREL) - 3)] * len(M.LAGS)))
    assert M.lag_minima(e)[0][1] == M.DREL[0], "an exact tie goes to the westernmost location"
    st = np.full((2, len(M.LAGS), len(M.DREL)), np.nan)
    st[0, :, 0] = -2.0; st[0, :, 1] = -1.5; st[1, :, 1] = -1.5
    e2 = M.exact_composite(st, 1000)
    assert M.lag_minima(e2)[0] == (-2.0, M.DREL[0]) and e2.n[0, :2].tolist() == [1, 2], "the mean decides, not the sum (-3 over two tracks is -1.5)"
    try:
        M.exact_composite(np.array([[0.3333]]), 3600); assert False
    except SystemExit as ex:
        assert "not integers over the scale" in str(ex)
    ok = M.exact_composite(np.array([[1.0 / 12000.0, 1.0 / 12400.0]]), 372000)
    assert ok.S.tolist() == [31, 30] and M.SCALES == {"RB1": 372000, "RB9": 3600}


def test_verify_lookback_finds_a_changed_location():
    M = _load()
    lons = M.X.XLONS
    fields = {}
    for i, y in enumerate(M.YEARS):
        f = np.zeros((122, len(lons)))
        if y == 1990:
            for d in range(122):
                f[d] = np.rint(-4.0 * np.exp(-0.5 * ((lons - (35.0 + 3.0 * (60 - d))) / 2.0) ** 2) * 1200) / 1200     # a lattice both scales contain
        fields[(y, "10 to 14.5 N")] = {"RB1": f, "RB9": f}
    tracks = [{"year": 1990, "day": 60, "x0": 35.0, "band": "10 to 14.5 N", "first_lon": 35.2}, {"year": 1990, "day": 62, "x0": 35.0, "band": "10 to 14.5 N", "first_lon": 37.0}]
    archive = {"tracks": tracks, "results": {}, "subsets": {}}
    for rep in M.REPS:
        archive["results"][rep] = {"reading": json.loads(json.dumps(M.LB.look_back(fields, tracks, rep)["reading"]))}
    for name, tr in M.LB.subsets(tracks).items():
        if tr:
            archive["subsets"][name] = {rep: {"reading": json.loads(json.dumps(M.LB.look_back(fields, tr, rep)["reading"]))} for rep in M.REPS}
    checked, mism = M.verify_lookback(archive, fields)
    assert mism == [] and checked == (1 + 3) * 2 * len(M.LAGS) * 24, "all, and the three non-empty subsets (west and east of 36 E, the low band), under both representations"
    archive["results"]["RB1"]["reading"]["lags"]["-3"]["D_star"] += 0.5
    archive["subsets"]["band 10 to 14.5 N"]["RB9"]["reading"]["lags"]["0"]["shift_minima"][4] = -9.0
    assert M.verify_lookback(archive, fields)[1] == ["all, RB1, lag -3, observed", "band 10 to 14.5 N, RB9, lag 0, shift 5"]
