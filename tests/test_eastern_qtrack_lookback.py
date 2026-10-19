"""The QTrack look-back's rules (EASTERN_QTRACK_LOOKBACK_2026-10-19.md, sections 2 and 3)."""

import datetime as dt
import importlib.util
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load():
    spec = importlib.util.spec_from_file_location("qtrack_lookback_under_test", os.path.join(ROOT, "scripts", "eastern_qtrack_lookback.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_bands_rounding_and_the_track_table():
    M = _load()
    assert M.band_of(9.7) == "10 to 14.5 N" and M.band_of(14.74) == "10 to 14.5 N" and M.band_of(14.76) == "15 to 19.5 N"
    assert M.band_of(14.75) == "10 to 14.5 N", "a tie goes to the southern band"
    assert M.band_of(31.0) == "25 to 29.5 N"
    assert M.round_half_east(34.24) == 34.0 and M.round_half_east(34.25) == 34.5 and M.round_half_east(34.26) == 34.5 and M.round_half_east(34.74) == 34.5 and M.round_half_east(34.75) == 35.0
    tracks = {("q", "f", 1): (dt.datetime(1990, 7, 4, 18), 9.7, 34.26, -20.0), ("q", "f", 2): (dt.datetime(1990, 6, 1, 0), 17.0, 38.0, 10.0),
              ("q", "f", 3): (dt.datetime(1990, 7, 4, 18), 9.7, 29.9, -20.0), ("q", "f", 4): (dt.datetime(2005, 7, 4), 12.0, 35.0, 0.0)}
    t = M.track_table(tracks)
    assert [tuple(x["key"]) for x in t] == [("q", "f", 2), ("q", "f", 1)], "west of 30 E and 2005 are never eligible, sorted by start time"
    assert t[1]["day"] == 33 and t[1]["x0"] == 34.5 and t[1]["first_lon"] == 34.26 and t[1]["band"] == "10 to 14.5 N" and t[1]["year"] == 1990
    assert t[0]["day"] == 0 and t[0]["band"] == "15 to 19.5 N"


def _field(fn):
    F = np.full((122, len(_load().X.XLONS)), np.nan)
    for d in range(122):
        F[d] = fn(d)
    return F


def test_track_values_index_lags_and_relative_longitudes():
    M = _load()
    lons = M.X.XLONS
    F = _field(lambda d: d * 1000.0 + lons)
    v = M.track_values(F, 10, 35.0)
    assert v.shape == (8, 91)
    i0 = M.LAGS.index(0); im3 = M.LAGS.index(-3); j = int(np.flatnonzero(M.DREL == 5.5)[0])
    assert v[i0, j] == 10 * 1000.0 + 40.5 and v[im3, j] == 7 * 1000.0 + 40.5
    v2 = M.track_values(F, 2, 70.0)
    assert np.isnan(v2[M.LAGS.index(-3)]).all(), "before the season is missing"
    assert np.isnan(v2[i0, M.DREL > 5.0]).all() and np.isfinite(v2[i0, M.DREL <= 5.0]).all(), "east of 75 E is missing"
    v0 = M.track_values(F, 10, 0.0)
    assert np.isnan(v0[i0, M.DREL < -10.0]).all() and np.isfinite(v0[i0, M.DREL >= -10.0]).all() and v0[i0, int(np.flatnonzero(M.DREL == -10.0)[0])] == 10 * 1000.0 - 10.0, "west of 10 W is missing, never wrapped"
    v3 = M.track_values(F, 120, 35.0)
    assert np.isnan(v3[M.LAGS.index(2)]).all() and np.isfinite(v3[M.LAGS.index(1)]).all()


def test_the_composite_the_shift_and_the_counts():
    M = _load()
    years = M.YEARS
    fields = {}
    for i, y in enumerate(years):
        f = _field(lambda d: np.full(len(M.X.XLONS), float(i)))
        fields[(y, "10 to 14.5 N")] = {"RB1": f}
    f = fields[(1983, "10 to 14.5 N")]["RB1"].copy(); f[60, :] = np.nan; fields[(1983, "10 to 14.5 N")]["RB1"] = f
    tracks = [{"year": 1983, "day": 50, "x0": 35.0, "band": "10 to 14.5 N"}, {"year": 2007, "day": 60, "x0": 35.0, "band": "10 to 14.5 N"}]
    C, n = M.composite(M.track_stack(fields, tracks, "RB1"))
    assert np.allclose(C[M.LAGS.index(0)], (0 + 23) / 2.0) and (n[M.LAGS.index(0)] == 2).all()
    C1, n1 = M.composite(M.track_stack(fields, tracks, "RB1", shift=1))
    assert np.allclose(C1[M.LAGS.index(-1)], (1 + 0) / 2.0), "1983 moves to 1984 and 2007 wraps to 1983"
    assert np.allclose(C1[M.LAGS.index(0)], 1.0) and (n1[M.LAGS.index(0)] == 1).all(), "2007 moved to 1983 is missing on day 60 and is left out of the mean and the count"
    C22, _ = M.composite(M.track_stack(fields, tracks, "RB1", shift=22))
    assert np.allclose(C22[M.LAGS.index(0)], (22 + 21) / 2.0), "1983 moves 22 places to 2006 (index 22) and 2007 to 2004 (index 21)"


def _C(spec):
    """A composite with value -v at relative longitude d for each lag in spec, zero elsewhere."""
    M = _load()
    C = np.zeros((len(M.LAGS), len(M.DREL)))
    for k, (d, v) in spec.items():
        C[M.LAGS.index(k), int(np.flatnonzero(M.DREL == d)[0])] = -v
    return C


def test_the_reading():
    M = _load()
    good = {-3: (6.0, 3.0), -2: (4.0, 3.0), -1: (2.0, 3.0), 0: (1.5, 3.0)}
    shifts = [_C({k: (0.0, 1.0) for k in M.LAGS})] * 23
    r = M.reading(_C(good), shifts)
    assert r["approach"] and r["instrument"] and r["lags"][-3]["rank"] == 0 and r["lags"][-3]["D_star"] == 6.0 and r["lags"][0]["min"] == -3.0
    assert r["parts"] == {"ranks_zero_at_minus3_to_minus1": True, "not_moving_east": True, "west_at_least_4p5": True}, "6 to 1.5 is exactly 4.5 degrees west, which passes"
    r4 = M.reading(_C({**good, -3: (5.5, 3.0)}), shifts)
    assert r4["approach"] is False and r4["parts"]["west_at_least_4p5"] is False and r4["parts"]["not_moving_east"] is True, "4.0 degrees west fails"
    r5 = M.reading(_C({**good, -3: (5.5, 3.0), 0: (1.5 + 0.5, 3.0), -1: (2.0, 3.0)}), shifts)
    assert r5["approach"] is False and r5["parts"]["west_at_least_4p5"] is False, "3.5 degrees is too little"
    r6 = M.reading(_C({**good, -1: (4.5, 3.0)}), shifts)
    assert r6["approach"] is False and r6["parts"]["not_moving_east"] is False, "a step east breaks it"
    r7 = M.reading(_C({**good, -2: (4.0, 3.0), -1: (4.0, 3.0)}), shifts)
    assert r7["approach"] is True, "a stall does not break it"
    one_deep = [_C({k: (0.0, 1.0) for k in M.LAGS})] * 22 + [_C({**{k: (0.0, 1.0) for k in M.LAGS}, -2: (10.0, 3.0)})]
    r8 = M.reading(_C(good), one_deep)
    assert r8["lags"][-2]["rank"] == 1 and r8["approach"] is False and r8["parts"]["ranks_zero_at_minus3_to_minus1"] is False, "a shift at the same depth counts against the observed"
    r9 = M.reading(_C({**good, 0: (1.5, 0.5)}), shifts)
    assert r9["instrument"] is False, "the start day weaker than every shift's minimum is not registered"
    tie = _C({-3: (6.0, 3.0), -2: (4.0, 3.0), -1: (2.0, 3.0), 0: (1.5, 3.0)}); tie[M.LAGS.index(0), int(np.flatnonzero(M.DREL == -4.0)[0])] = -3.0
    assert M.reading(tie, shifts)["lags"][0]["D_star"] == -4.0, "the westernmost location on a tie"
    db = M.reading(_C({-3: (6.0, 0.5), -2: (4.0, 0.5), -1: (2.0, 3.0), 0: (1.5, 3.0)}), shifts)
    assert db["approach"] is False and db["day_before_only"] is True
    db2 = M.reading(_C({-3: (6.0, 0.5), -2: (4.0, 0.5), -1: (1.5, 3.0), 0: (1.5, 3.0)}), shifts)
    assert db2["day_before_only"] is False, "the day before must lie east of the start day's minimum"


def test_the_repaired_five_day_reading():
    M = _load()
    shifts = [_C({k: (0.0, 1.0) for k in M.LAGS})] * 23
    five = {-5: (10.0, 3.0), -4: (8.0, 3.0), -3: (6.0, 3.0), -2: (4.0, 3.0), -1: (2.0, 3.0), 0: (0.0, 3.0)}
    r = M.reading(_C(five), shifts)
    assert r["approach_r1"] is True and r["parts_r1"] == {"ranks_zero_at_minus5_to_minus1": True, "not_moving_east_from_minus5": True, "west_at_least_7p5": True}
    assert M.reading(_C({**five, -5: (7.5, 3.0), -4: (7.0, 3.0)}), shifts)["approach_r1"] is True, "exactly 7.5 degrees passes"
    assert M.reading(_C({**five, -3: (8.0, 3.0)}), shifts)["approach_r1"] is True, "a stall between lags -4 and -3 does not break it"
    r2 = M.reading(_C({**five, -5: (7.0, 3.0), -4: (7.0, 3.0)}), shifts)
    assert r2["approach_r1"] is False and r2["parts_r1"]["west_at_least_7p5"] is False and r2["approach"] is True, "7 degrees fails the repair, the declared three-day reading still passes"
    r3 = M.reading(_C({**five, -5: (9.0, 3.0), -4: (9.5, 3.0), -3: (9.5, 3.0)}), shifts)
    assert r3["approach_r1"] is False and r3["parts_r1"]["not_moving_east_from_minus5"] is False, "a turn at lag -4, the wobble's shape"
    deep5 = [_C({k: (0.0, 1.0) for k in M.LAGS})] * 22 + [_C({**{k: (0.0, 1.0) for k in M.LAGS}, -5: (0.0, 3.0)})]
    r4 = M.reading(_C(five), deep5)
    assert r4["approach_r1"] is False and r4["parts_r1"]["ranks_zero_at_minus5_to_minus1"] is False and r4["approach"] is True, "a rank of 1 at lag -5 fails the repair only"
    deep4 = [_C({k: (0.0, 1.0) for k in M.LAGS})] * 22 + [_C({**{k: (0.0, 1.0) for k in M.LAGS}, -4: (0.0, 3.0)})]
    assert M.reading(_C(five), deep4)["approach_r1"] is False


def test_contribution_counts():
    M = _load()
    st = np.zeros((4, len(M.LAGS), len(M.DREL)))
    i = M.LAGS.index(-2); j = lambda d: int(np.flatnonzero(M.DREL == d)[0])
    st[0, i, j(5.0)] = -2.0; st[1, i, j(6.0)] = -3.0; st[2, i, j(6.5)] = -5.0; st[3, i, j(5.0)] = -1.99
    assert M.contribution_count(st, 5.0, -2) == 2, "-2 inclusive, 1 degree inclusive, 1.5 degrees away excluded"
    st[3, i, :] = np.nan
    assert M.contribution_count(st, 5.0, -2) == 2 and M.contribution_count(st, None, -2) is None


def test_the_constructions_place_their_features_as_declared():
    M = _load()
    days, phase = M.construction_days(0, "K1")
    assert len(days) == 4 and all(30 <= d <= 110 for d in days) and days == sorted(days) and days == M.construction_days(0, "K6")[0]
    H, tracks, centers = M.construction_season("K1", 0)
    assert H.shape == (122, len(M.X.XLONS)) and [t["day"] for t in tracks] == days and all(t["x0"] == 35.0 and t["band"] == "10 to 14.5 N" for t in tracks)
    for c, d in zip(centers, days):
        assert abs(c[d] - 35.0) < 0.02 and abs(c[d - 3] - 41.0) < 0.05, "35 E as a daily mean on the start day, 2 degrees per day"
    _, _, c2 = M.construction_season("K2", 3)
    d2 = M.construction_days(3, "K2")[0]
    assert all(abs(c[d] - 35.0) < 0.02 and abs(c[d - 1] - 40.0) < 0.05 for c, d in zip(c2, d2))
    _, _, c4 = M.construction_season("K4", 5)
    d4 = M.construction_days(5, "K4")[0]
    assert all(np.isnan(c[d - 1]) and abs(c[d] - 33.5) < 0.02 for c, d in zip(c4, d4)), "absent the day before, at 35 E as the start day begins"
    _, _, c7 = M.construction_season("K7", 2)
    assert len(c7) == 2
    _, tr6, c6 = M.construction_season("K6", 1)
    assert c6 == [] and len(tr6) == 4
    d5, ph5 = M.construction_days(4, "K5")
    _, tr5, _ = M.construction_season("K5", 4)
    assert len(d5) == 4 and all(t["x0"] == 38.0 for t in tr5) and M.construction_days(4, "K8") == (d5, ph5)
    for d in d5:
        tt = d + 0.375
        speed = 2 * np.pi / 14.0 * 3.0 * np.cos(2 * np.pi * tt / 14.0 + ph5)
        assert speed < -1.3, "each K5 start is within half a day of the fastest westward wobble (at most 1.35 degrees per day)"
    t0 = np.arange(122 * 4) / 4.0
    assert abs(np.ptp(M.wobble_center(t0, ph5, M.WOBBLE_AMPLITUDE["K8"])) - 10.0) < 0.05 and abs(np.ptp(M.wobble_center(t0, ph5, M.WOBBLE_AMPLITUDE["K5"])) - 6.0) < 0.05
    d9, ph9 = M.construction_days(4, "K9")
    assert ph9 == ph5 and len(d9) == 4 and d9 != d5
    for d in d9:
        tt = d + 0.375
        assert 2 * np.pi / 20.0 * 5.0 * np.cos(2 * np.pi * tt / 20.0 + ph9) < -1.5, "K9 starts at its fastest westward wobble, 20-day period"
    assert abs(np.ptp(M.wobble_center(t0, ph9, 5.0, 20.0)) - 10.0) < 0.05
    H5, _, _ = M.construction_season("K5", 4); H8, _, _ = M.construction_season("K8", 4); H6, _, _ = M.construction_season("K6", 4)
    H9, _, _ = M.construction_season("K9", 4)
    assert not np.array_equal(H8, H9), "K9's period differs from K8's"
    assert not np.array_equal(H5, H8) and not np.array_equal(H5, H6), "the wobble enters the field and its amplitude differs between K5 and K8"


def test_the_rule_the_verdicts_and_the_refusal(tmp_path):
    M = _load()
    def res(appr, inst=True):
        return {"reading": {"approach": appr, "instrument": inst, "day_before_only": False}}
    assert M.CASES == ("K1", "K2", "K3", "K4", "K5", "K6", "K7", "K8", "K9") and M.RULE == {"shown": ("K1", "K2"), "not_shown": ("K4", "K5", "K6"), "start_registered": ("K1", "K2", "K4")}
    assert M.RULE_R1 == {"shown": ("K1", "K2"), "not_shown": ("K4", "K5", "K6", "K8"), "start_registered": ("K1", "K2", "K4")}
    ok = {c: {rep: res(c in ("K1", "K2", "K3", "K7", "K8")) for rep in M.REPS} for c in M.CASES}
    assert M.apply_rule(ok) == {"adequate": True, "failures": []}
    for c in M.CASES:
        for rep in M.REPS:
            ok[c][rep]["reading"]["approach_r1"] = c in ("K1", "K2", "K9")
    assert M.apply_rule(ok, repaired=True) == {"adequate": True, "failures": []}
    ok["K8"]["RB9"]["reading"]["approach_r1"] = True
    assert M.apply_rule(ok, repaired=True)["failures"] == ["K8 under RB9 approaches"] and M.apply_rule(ok)["adequate"] is True
    ok["K8"]["RB9"]["reading"]["approach_r1"] = False; ok["K1"]["RB1"]["reading"]["approach_r1"] = False
    assert M.apply_rule(ok, repaired=True)["failures"] == ["K1 under RB1 does not approach"]
    bad = json.loads(json.dumps(ok)); bad["K5"]["RB1"]["reading"]["approach"] = True; bad["K2"]["RB9"]["reading"]["approach"] = False; bad["K4"]["RB1"]["reading"]["instrument"] = False
    v = M.apply_rule(bad)
    assert v["adequate"] is False and sorted(v["failures"]) == ["K2 under RB9 does not approach", "K4 under RB1 does not register its start", "K5 under RB1 approaches"]
    assert M.verdict_of({"RB1": res(True), "RB9": res(True)}) == "SHOWN" and M.verdict_of({"RB1": res(True), "RB9": res(False)}) == "under RB1 only"
    assert M.verdict_of({"RB1": res(False), "RB9": res(False)}) == "not shown"
    db = {rep: {"reading": {"approach": False, "approach_r1": False, "day_before_only": True}} for rep in ("RB1", "RB9")}
    assert M.verdict_of(db).startswith("a signal on the day before only") and M.verdict_of(db, "approach_r1") == "not shown", "the repaired reading never reports the day-before pattern"
    mix = {"RB1": {"reading": {"approach": False, "approach_r1": True, "day_before_only": False}}, "RB9": {"reading": {"approach": True, "approach_r1": True, "day_before_only": False}}}
    assert M.verdict_of(mix, "approach_r1") == "SHOWN" and M.verdict_of(mix) == "under RB9 only"
    for rep in mix:
        mix[rep]["reading"]["instrument"] = True
    assert M.archive_verdicts(mix) == ("SHOWN", "under RB9 only")
    mix["RB9"]["reading"]["instrument"] = False
    assert M.archive_verdicts(mix) == ("stopped, the start day is not registered under both representations",) * 2
    good = {c: {rep: {"reading": {"approach": c in ("K1", "K2", "K3", "K7", "K8", "K9"), "approach_r1": c in ("K1", "K2", "K3", "K7"), "instrument": True, "day_before_only": False}} for rep in M.REPS} for c in M.CASES}
    rerun = lambda: json.loads(json.dumps(good))
    v_ok = M.apply_rule(good, repaired=True)
    assert v_ok["adequate"] is True
    p = tmp_path / "c.json"
    stale = json.loads(json.dumps(good)); stale["K8"]["RB1"]["reading"]["approach_r1"] = True
    for rec, run, msg in (({"generated_by": {"sha256": "x"}, "results": good, "verdict_r1": v_ok}, rerun, "different version"),
                          ({"generated_by": {"sha256": M.SCRIPT_SHA256_AT_LOAD}, "results": good, "verdict_r1": v_ok}, lambda: stale, "differs from a fresh run"),
                          ({"generated_by": {"sha256": M.SCRIPT_SHA256_AT_LOAD}, "results": stale, "verdict_r1": v_ok}, lambda: json.loads(json.dumps(stale)), "do not pass the repaired rule"),
                          ({"generated_by": {"sha256": M.SCRIPT_SHA256_AT_LOAD}, "results": good, "verdict_r1": {"adequate": True, "failures": ["x"]}}, rerun, "stored summary disagrees")):
        p.write_text(json.dumps(rec))
        try:
            M.check_constructions(str(p), rerun=run); assert False
        except SystemExit as ex:
            assert msg in str(ex), str(ex)
    p.write_text(json.dumps({"generated_by": {"sha256": M.SCRIPT_SHA256_AT_LOAD}, "results": good, "verdict": {"adequate": False, "failures": ["K5"]}, "verdict_r1": v_ok}))
    assert M.check_constructions(str(p), rerun=rerun)["verdict_r1"]["adequate"] is True, "the repaired rule governs the archive, recomputed from the stored cases"


def test_anomaly_values_use_the_exact_path():
    M = _load()
    month_of = M.A.month_index()
    H = np.zeros((122, 3), dtype=np.int64); H[month_of == 6, 0] = 400 * 3; H[0, 1] = 400 * 30
    v = M.anomaly_values(H, month_of, "B1", gap=np.array([False, False, True]))
    assert v[0, 0] == 0.0 and v[0, 1] == 29.0 and v[1, 1] == -1.0 and np.isnan(v[:, 2]).all()
    v9 = M.anomaly_values(H, month_of, "B2_9")
    assert np.isnan(v9[:4]).all() and np.isnan(v9[-4:]).all() and np.isfinite(v9[4:-4]).all()


def test_subsets_rows_and_per_season_noise():
    M = _load()
    tr = [{"first_lon": 35.99, "band": "10 to 14.5 N"}, {"first_lon": 36.0, "band": "15 to 19.5 N"}, {"first_lon": 40.0, "band": "10 to 14.5 N"}]
    sub = M.subsets(tr)
    assert [len(sub[k]) for k in ("first position west of 36 E", "first position at or east of 36 E", "band 10 to 14.5 N", "other bands")] == [1, 2, 2, 1]
    lags = {k: {"min": -1.0 * (k + 10), "D_star": float(-k), "rank": 0} for k in M.LAGS}
    rd = {"lags": lags, "instrument": True, "approach": False, "day_before_only": True, "approach_r1": True}
    row = M.reading_rows("K9", "RB1", rd, {k: {"observed": 3, "shift_mean": 0.5, "shift_max": 2} for k in M.APPROACH_LAGS})
    assert row == "| K9 | RB1 | 5 (0), 4 (0) | -7.00 at 3 (rank 0) | -8.00 at 2 (rank 0) | -9.00 at 1 (rank 0) | -10.00 at 0 (rank 0) | yes | no | yes | yes | 3 (0.50, 2) | 3 (0.50, 2) | 3 (0.50, 2) |"
    assert row.count("|") - 1 == M.HEADER.count("|") - 1 + 3
    H0, _, _ = M.construction_season("K6", 0); H1, _, _ = M.construction_season("K6", 1)
    assert not np.array_equal(H0, H1), "each synthetic season has its own noise"


def test_missing_shifts_never_pass_and_the_qtrack_files_alone(tmp_path):
    """Review fold: a lag with any shifted season unavailable has no rank, so it fails the
    start-day check and both approaches, and the archive reads QTrack's files without the
    catalog."""
    M = _load()
    lons = M.X.XLONS
    fields = {(y, "10 to 14.5 N"): {"RB1": np.full((122, len(lons)), np.nan)} for y in M.YEARS}
    f = np.zeros((122, len(lons)))
    for d in range(122):
        f[d] = -5.0 * np.exp(-0.5 * ((lons - (35.0 + 3.0 * (60 - d))) / 2.0) ** 2)
    fields[(1990, "10 to 14.5 N")]["RB1"] = f
    r = M.look_back(fields, [{"year": 1990, "day": 60, "x0": 35.0, "band": "10 to 14.5 N"}], "RB1")
    rd = r["reading"]
    assert rd["lags"][0]["rank"] is None and rd["lags"][0]["valid_shifts"] == 0 and rd["instrument"] is False and rd["approach"] is False and rd["approach_r1"] is False
    assert np.isnan(r["pointwise_rank"]).all()
    fields[(1984, "10 to 14.5 N")]["RB1"] = np.zeros((122, len(lons)))
    r2 = M.look_back(fields, [{"year": 1990, "day": 60, "x0": 35.0, "band": "10 to 14.5 N"}], "RB1")
    assert r2["reading"]["lags"][0]["valid_shifts"] == 1 and r2["reading"]["lags"][0]["rank"] is None, "one shift of 23 is not the declared comparison"
    for y in M.YEARS:
        if y != 1990:
            fields[(y, "10 to 14.5 N")]["RB1"] = np.zeros((122, len(lons)))
    r3 = M.look_back(fields, [{"year": 1990, "day": 60, "x0": 35.0, "band": "10 to 14.5 N"}], "RB1")
    assert r3["reading"]["lags"][0]["rank"] == 0 and r3["reading"]["approach_r1"] is True and r3["reading"]["lags"][0]["valid_shifts"] == 23
    q = tmp_path / "q"; q.mkdir()
    for y in M.YEARS + [2005]:
        (q / f"ERA5_AEW_tracks_atlantic_{y}.nc").write_bytes(b"")
    assert [M.X.qtrack_year(p) for p in M.qtrack_paths(str(q))] == M.YEARS, "no catalog is needed and 2005 is never read"
    (q / "ERA5_AEW_tracks_atlantic_1999.nc").unlink()
    try:
        M.qtrack_paths(str(q)); assert False
    except SystemExit as ex:
        assert "missing [1999]" in str(ex)
