"""The wobble climatology's rules (EASTERN_WOBBLE_CLIMATOLOGY_2026-10-19.md, sections 2 and 3)."""

import importlib.util
import json
import os
from fractions import Fraction

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load():
    spec = importlib.util.spec_from_file_location("wobble_under_test", os.path.join(ROOT, "scripts", "eastern_wobble_climatology.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_daily_measure():
    M = _load()
    lons = M.LONS
    cols = M.window_cols((44.5, 57.0))
    assert len(cols) == 26 and lons[cols[0]] == 44.5 and lons[cols[-1]] == 57.0
    row = np.zeros(len(lons), dtype=np.int64)
    j = lambda x: int(np.flatnonzero(lons == x)[0])
    row[j(50.0)] = -2000; row[j(50.5)] = -1600; row[j(51.0)] = -1599; row[j(54.0)] = -1700
    d = M.day_measure(row, cols)
    assert d["core"] == Fraction(100 * 2000 + 101 * 1600, 2 * 3600), "-4 m/s exactly qualifies, -3.9975 does not, weights are the negated sums, a cell outside the strongest cell's run does not count"
    assert d["strongest"] == 50.0 and d["strongest_value"] == -5.0 and d["edge"] is False and d["region_edge"] is False
    row6 = np.zeros(len(lons), dtype=np.int64)
    row6[j(48.0)] = -1600; row6[j(48.5)] = -1700; row6[j(49.0)] = -2400; row6[j(49.5)] = -1650; row6[j(50.0)] = -1650
    row6[j(47.0)] = -1800; row6[j(51.0)] = -1700
    d6 = M.day_measure(row6, cols)
    assert d6["core"] == Fraction(96 * 1600 + 97 * 1700 + 98 * 2400 + 99 * 1650 + 100 * 1650, 2 * (1600 + 1700 + 2400 + 1650 + 1650)), "the run extends both ways to the first cell that does not qualify"
    row2 = np.zeros(len(lons), dtype=np.int64); row2[j(44.5)] = -2000; row2[j(47.0)] = -2000; row2[j(56.0)] = -1700
    d2 = M.day_measure(row2, cols)
    assert d2["strongest"] == 44.5 and d2["core"] == Fraction(89, 2) and d2["edge"] is True and d2["region_edge"] is True, "the westernmost on an exact tie, on the window's edge"
    row3 = np.zeros(len(lons), dtype=np.int64); row3[j(57.0)] = -1700; row3[j(50.0)] = -1800
    d3 = M.day_measure(row3, cols)
    assert d3["edge"] is False and d3["region_edge"] is False and d3["core"] == 50, "a qualifying cell on the edge outside the strongest cell's run flags nothing"
    east = np.zeros(len(lons), dtype=np.int64); east[j(56.0)] = -1800; east[j(56.5)] = -1700; east[j(57.0)] = -1650
    west = np.zeros(len(lons), dtype=np.int64); west[j(44.5)] = -1650; west[j(45.0)] = -1800
    assert all(M.day_measure(r, cols)["edge"] is False and M.day_measure(r, cols)["region_edge"] is True for r in (east, west)), "the strongest cell's run reaching either edge flags the region"
    row5 = np.zeros(len(lons), dtype=np.int64); row5[j(57.0)] = -2400; row5[j(50.0)] = -1800
    assert M.day_measure(row5, cols)["edge"] is True and M.day_measure(row5, cols)["strongest"] == 57.0, "the strongest cell on the eastern edge"
    row4 = np.zeros(len(lons), dtype=np.int64); row4[j(43.0)] = -4000; row4[j(58.0)] = -4000
    assert M.day_measure(row4, cols) is None, "cells outside the window do not count"
    assert M.day_measure(row, cols, threshold=-5.0)["core"] == 50 and M.day_measure(row, cols, threshold=-6.0) is None


def _table_blocks(text):
    """The cell-boundary counts of each Markdown table's lines, one list per table."""
    blocks, cur = [], []
    for line in text.splitlines():
        if line.startswith("|"):
            cur.append(line.count("|"))
        elif cur:
            blocks.append(cur); cur = []
    return blocks + ([cur] if cur else [])


def _days(cores):
    return [None if c is None else {"core": Fraction(c).limit_denominator(1000), "edge": False, "region_edge": False} for c in cores]


def test_swing_timescale_and_excursions():
    M = _load()
    month_of = M.A.month_index()
    t = np.arange(122)
    s14 = 50.0 + 3.0 * np.sin(2 * np.pi * t / 14.0)
    sw = M.monthly_swing(_days(s14), month_of)
    assert sorted(sw) == [6, 7, 8, 9] and all(abs(a - 3.0) < 0.2 and abs(h - 3.0) < 0.1 for a, h in sw.values()), "the amplitude estimate of a sinusoid is its amplitude"
    gappy = list(s14); gappy[0:11] = [None] * 11
    assert 6 not in M.monthly_swing(_days(gappy), month_of) and 7 in M.monthly_swing(_days(gappy), month_of), "19 present June days are too few"
    twenty = list(s14); twenty[0:10] = [None] * 10
    assert 6 in M.monthly_swing(_days(twenty), month_of), "20 present June days are enough"
    alt = [49.0 if i % 2 else 51.0 for i in range(122)]
    a6, h6 = M.monthly_swing(_days(alt), month_of)[6]
    assert abs(a6 - np.sqrt(2.0)) < 1e-12 and h6 == 1.0, "the population standard deviation, so an alternation of plus and minus 1 gives the square root of 2"
    assert M.season_timescale(_days(s14), month_of) == 14
    assert M.season_timescale(_days(50.0 + 3.0 * np.sin(2 * np.pi * t / 28.0)), month_of) == 28
    few = [None] * 43 + list(s14[43:])
    assert M.season_timescale(_days(few), month_of) is None, "79 present days are too few"
    assert M.season_timescale(_days(list(s14[:80]) + [None] * 42), month_of) == 14, "80 present days are enough"
    steps = [50.0, 48.5, 48.5, 45.5, 45.5, 50.0, 49.0, 50.0, 46.0, 45.0]
    assert M.excursions(_days(steps), 3, Fraction(9, 2)) == (1, 7), "exactly 4.5 degrees west with a stall passes, a step east fails"
    assert M.excursions(_days([50.0, 51.0, 47.0, 45.5]), 3, Fraction(9, 2)) == (0, 1), "4.5 degrees west with one step east does not pass"
    assert M.excursions(_days([50.0, 48.0, None, 46.0, 44.0, 42.0, 41.5]), 3, Fraction(9, 2)) == (1, 1), "a window with an absent day is not complete"
    assert M.largest_westward(_days([50.0, 49.0, 48.0, 47.0, 46.0, 51.0, 40.0])) == 9.0 and M.largest_westward(_days([50.0, None] * 4)) is None


def test_summarize_counts_and_denominators():
    M = _load()
    month_of = M.A.month_index()
    t = np.arange(122)
    a = _days(50.0 + 3.0 * np.sin(2 * np.pi * t / 14.0))
    b = list(a); b[0:30] = [None] * 30; b[40] = dict(b[40], edge=True, region_edge=True)
    s = M.summarize([a, b], month_of)
    assert s["presence"][6] == {"present": 30, "days": 60} and s["presence"][7] == {"present": 62, "days": 62}
    assert s["swing"]["season_months"] == 7 and s["timescale"] == {"qualifying_seasons": 2, "with_minimum": 2, "period_q": [14.0, 14.0, 14.0]}
    assert s["excursions"]["5_day"]["west_deg"] == 7.5 and s["edges"] == {"present_days": 214, "edge": 1, "region_edge": 1}
    e3 = s["excursions"]["3_day"]
    assert e3["windows"] == (122 - 3) + (92 - 3) and e3["meeting"] == sum(M.excursions(x, 3, Fraction(9, 2))[0] for x in (a, b))


def test_proximity_counts_positions_only():
    M = _load()
    cores = [None] * 122
    cores[10] = {"core": Fraction(50)}; cores[11] = {"core": Fraction(50)}; cores[12] = {"core": Fraction(50)}
    hist = {(1990, "20 to 24.5 N"): [{"days": [10, 11, 12, 13], "lons": [51.5, 52.0, 58.0, 49.0]}],
            (1990, "10 to 14.5 N"): [{"days": [10], "lons": [50.0]}]}
    p = M.proximity(hist, {("S2", 1990, "20 to 24.5 N"): cores, ("S2", 1990, "10 to 14.5 N"): cores})
    assert p == {"member_days_in_window": 2, "near_core": 1, "histories_in_window": 1, "histories_half_near": 1}, "1.5 degrees inclusive, outside the window or on an absent day not counted, other bands left out"


def _fake_records(M, value=0.0):
    import datetime as dt
    recs = []
    for y in M.YEARS:
        for m in M.MONTHS:
            n = M.A.MONTH_DAYS[m]
            times = [f"{(dt.date(y, m, 1) + dt.timedelta(days=d)).isoformat()}T{h:02d}" for d in range(n) for h in (0, 6, 12, 18)]
            bands = {b: (np.full((len(times), len(M.LONS)), value)).tolist() for b in M.SC.V.BANDS}
            recs.append({"year": y, "month": m, "times": times, "lons": M.LONS.tolist(), "bands": list(M.SC.V.BANDS), "band_means_700": bands})
    by = {}
    for r in recs:
        by.setdefault(r["year"], {})[r["month"]] = r
    return by


def test_recomputed_histories_must_match_the_stored_count():
    M = _load()
    by = _fake_records(M)
    assert M.recompute_histories(by, 0) == {(y, b): [] for y in M.YEARS for b in M.SC.V.BANDS}
    try:
        M.recompute_histories(by, 104); assert False
    except SystemExit as ex:
        assert "not the stored 104" in str(ex)
    assert M.stored_history_count({"tables": {"RB9|LN": {"westward_histories": [1, 2, 3]}}}) == 3
    res, cores_by = M.archive_results(by)
    assert sorted(res) == ["S1|10 to 14.5 N", "S2|15 to 19.5 N", "S2|20 to 24.5 N", "S2|25 to 29.5 N", "S3|20 to 24.5 N", "S3|25 to 29.5 N"]
    assert res["S2|15 to 19.5 N"]["declared"]["presence"][6] == {"present": 0, "days": 720} and set(res["S1|10 to 14.5 N"]["sensitivity"]) == {"-3", "-5"}
    assert M.proximity({}, cores_by) == {"member_days_in_window": 0, "near_core": 0, "histories_in_window": 0, "histories_half_near": 0}
    res5, cb5 = M.archive_results(_fake_records(M, -5.0))
    assert cb5[("S2", 1983, "20 to 24.5 N")][3]["core"] == Fraction(203, 4) and cb5[("S3", 1983, "20 to 24.5 N")][3]["core"] == Fraction(255, 4)
    assert M.proximity({(1983, "20 to 24.5 N"): [{"days": [3], "lons": [50.5]}]}, cb5) == {"member_days_in_window": 1, "near_core": 1, "histories_in_window": 1, "histories_half_near": 1}, "S3 shares the band and must not replace S2's cores"
    prox = M.proximity({}, cores_by); s1 = res["S1|10 to 14.5 N"]["declared"]
    assert "stated floor" not in M.archive_text(res, prox)
    text = M.archive_text(res, prox, floor=s1)
    assert "stated floor" in text and M.summary_rows("W0, constructions (the floor)", s1) in text.splitlines(), "the floor row is printed beside the archive's rows"
    assert "unmatched for S1, S3 and the other bands" in text, "the floor's limits travel with the table"
    row = M.summary_rows("S1|10 to 14.5 N", s1)
    assert M.HEADER.count("|") == 12 and row.count("|") == 12 and row.startswith("| S1, 10 to 14.5 N |"), "eleven cells in the header and in an element and band row"
    assert all(len(set(b)) == 1 for b in _table_blocks(text)) and len(_table_blocks(text)) == 3, "every row of each table has its header's cell count"


def test_constructions_rule_and_gate(tmp_path):
    M = _load()
    H1, s1, _ = M.construction_season("W1", 3)
    assert H1.shape == (122, len(M.LONS)) and abs(np.nanmax(s1) - 53.0) < 0.2 and abs(np.nanmin(s1) - 47.0) < 0.2
    _, s5, _ = M.construction_season("W5", 0); _, s4, t4 = M.construction_season("W4", 0)
    assert np.allclose(s5, 56.0) and np.allclose(s4, 50.0) and abs(t4[40] - (72.0 - 2.0 * (40 + 0.375 - 35.0))) < 1e-9
    assert not np.array_equal(M.construction_season("W0", 0)[0], M.construction_season("W0", 1)[0]), "each season has its own noise"
    assert not np.allclose(M.construction_season("W1", 3)[1], M.construction_season("W1", 4)[1]), "each season has its own phase"
    def res(a1, p1, a2, p2, a3, p3, a0):
        mk = lambda a, p: {"swing": {"amplitude_q": [0, a, 0]}, "timescale": {"period_q": [0, p, 0]}}
        return {"W1": mk(a1, p1), "W2": mk(a2, p2), "W3": mk(a3, p3), "W0": mk(a0, None)}
    assert M.apply_rule(res(2.25, 10.5, 6.25, 17.5, 3.75, 35, 0.99)) == {"adequate": True, "failures": []}, "the tolerance is inclusive"
    v = M.apply_rule(res(2.2, 14, 5, 18, 3, 28, 1.0))
    assert v["adequate"] is False and len(v["failures"]) == 3 and any(f.startswith("W0") for f in v["failures"])
    good = {"W0": {"swing": {"amplitude_q": [0, 0.5, 0]}, "timescale": {"period_q": None}}}
    for c, a, p in (("W1", 3, 14), ("W2", 5, 14), ("W3", 3, 28)):
        good[c] = {"swing": {"amplitude_q": [0, a, 0]}, "timescale": {"period_q": [0, p, 0]}}
    rerun = lambda: json.loads(json.dumps(good))
    p = tmp_path / "c.json"
    stale = json.loads(json.dumps(good)); stale["W2"]["swing"]["amplitude_q"] = [0, 9, 0]
    for rec, run, msg in (({"generated_by": {"sha256": "x"}, "results": good, "verdict": M.apply_rule(good)}, rerun, "different version"),
                          ({"generated_by": {"sha256": M.SCRIPT_SHA256_AT_LOAD}, "results": good, "verdict": M.apply_rule(good)}, lambda: stale, "differs from a fresh run"),
                          ({"generated_by": {"sha256": M.SCRIPT_SHA256_AT_LOAD}, "results": stale, "verdict": M.apply_rule(good)}, lambda: json.loads(json.dumps(stale)), "do not pass the rule"),
                          ({"generated_by": {"sha256": M.SCRIPT_SHA256_AT_LOAD}, "results": good, "verdict": {"adequate": True, "failures": ["x"]}}, rerun, "stored summary disagrees")):
        p.write_text(json.dumps(rec))
        try:
            M.check_constructions(str(p), rerun=run); assert False
        except SystemExit as ex:
            assert msg in str(ex), str(ex)
    p.write_text(json.dumps({"generated_by": {"sha256": M.SCRIPT_SHA256_AT_LOAD}, "results": good, "verdict": M.apply_rule(good)}))
    assert M.check_constructions(str(p), rerun=rerun)["verdict"]["adequate"] is True
    w0 = json.loads(json.dumps(good)); w0["W0"]["swing"]["amplitude_q"] = [0, 1.04, 0]
    both = json.loads(json.dumps(w0)); both["W1"]["swing"]["amplitude_q"] = [0, 9, 0]
    w1 = json.loads(json.dumps(good)); w1["W1"]["swing"]["amplitude_q"] = [0, 9, 0]
    for res_, flag, msg in ((w0, False, "do not pass the rule"), (both, True, "beyond W0's bar"), (w1, True, "beyond W0's bar")):
        p.write_text(json.dumps({"generated_by": {"sha256": M.SCRIPT_SHA256_AT_LOAD}, "results": res_, "verdict": M.apply_rule(res_)}))
        try:
            M.check_constructions(str(p), rerun=(lambda r=res_: json.loads(json.dumps(r))), floor=flag); assert False
        except SystemExit as ex:
            assert msg in str(ex), str(ex)
    p.write_text(json.dumps({"generated_by": {"sha256": M.SCRIPT_SHA256_AT_LOAD}, "results": w0, "verdict": M.apply_rule(w0)}))
    assert M.check_constructions(str(p), rerun=lambda: json.loads(json.dumps(w0)), floor=True)["verdict"]["adequate"] is False, "section 8 accepts W0's bar alone"
    p.write_text(json.dumps({"generated_by": {"sha256": M.SCRIPT_SHA256_AT_LOAD}, "results": w0, "verdict": M.apply_rule(good)}))
    try:
        M.check_constructions(str(p), rerun=lambda: json.loads(json.dumps(w0)), floor=True); assert False
    except SystemExit as ex:
        assert "stored summary disagrees" in str(ex), "the floor does not excuse a stored summary that disagrees"


def test_post_hoc_runs_steps_and_orders():
    M = _load()
    lons = M.LONS
    cols = M.window_cols((44.5, 57.0))
    j = lambda x: int(np.flatnonzero(lons == x)[0])
    r = np.zeros(len(lons), dtype=np.int64)
    r[j(44.5)] = -1600; r[j(45.0)] = -2000; r[j(45.5)] = -1599; r[j(47.0)] = -1700; r[j(56.5)] = -1600; r[j(57.0)] = -1800
    assert M.runs(r, cols) == [(0, 1), (5, 5), (24, 25)], "-4 m/s qualifies and -3.9975 breaks a run"
    d0 = np.zeros(len(lons), dtype=np.int64); d0[j(49.5)] = -1700; d0[j(50.0)] = -2000; d0[j(50.5)] = -1700
    d1 = np.zeros(len(lons), dtype=np.int64); d1[j(44.5)] = -2400; d1[j(45.0)] = -1600; d1[j(52.0)] = -1700
    d2 = np.zeros(len(lons), dtype=np.int64)
    d3 = np.zeros(len(lons), dtype=np.int64); d3[cols] = -1700; d3[j(50.0)] = -2000
    d4 = np.zeros(len(lons), dtype=np.int64); d4[j(56.0)] = -1700; d4[j(56.5)] = -2400; d4[j(57.0)] = -1700
    out = []
    s = M.run_summary([np.array([d0, d1, d2, d3, d4])], cols, daily_out=out)
    c1 = Fraction(89 * 2400 + 90 * 1600, 2 * 4000); c3 = Fraction(1700 * sum(range(89, 115)) + 100 * 300, 2 * (26 * 1700 + 300))
    steps = sorted([float(50 - c1), float(Fraction(113, 2) - c3)])
    assert (s["present_days"], s["two_or_more_runs"], s["run_reaches_west_only"], s["run_reaches_east_only"], s["run_spans_window"]) == (4, 1, 1, 1, 1)
    assert s["steps"] == 2 and s["step_q"] == [float(np.percentile(steps, q)) for q in (25, 50, 75, 90)], "an absent day breaks the chain of steps"
    assert s["run_span_q"] == [float(np.percentile([1.0, 0.5, 12.5, 1.0], q)) for q in (25, 50, 75, 90)], "spans in degrees between the run's first and last cells"
    assert out == [[50.0, 44.7, None, round(float(c3), 2), 56.5]], "the daily cores as the archive stores them"
    mk = lambda amp, e5: {"swing": {"amplitude_q": [0, amp, 0]}, "excursions": {"5_day": {"meeting": e5, "windows": 100}}}
    res = {"A": {"declared": mk(4.0, 5), "sensitivity": {"-3": mk(3.0, 4), "-5": mk(4.5, 6)}},
           "B": {"declared": mk(3.5, 1), "sensitivity": {"-3": mk(3.5, 1), "-5": mk(3.6, 2)}}}
    o = M.threshold_orders(res)
    assert o["median_amplitude"]["orders"] == {"-3": ["B", "A"], "-4": ["A", "B"], "-5": ["A", "B"]} and o["median_amplitude"]["threshold_dependent"] is True
    assert o["five_day_share"]["orders"]["-4"] == ["A", "B"] and o["five_day_share"]["threshold_dependent"] is False, "the largest first"
    one = {"A": {"present_days": 30, "run_reaches_west_only": 1, "run_reaches_east_only": 2, "run_spans_window": 0}}
    dailies = {"A": [[50.0, None] for _ in M.YEARS]}
    stored = lambda present=20, edge=3, daily=50.0: {"A": {"declared": {"presence": {6: {"present": 10}, 7: {"present": present}}, "edges": {"region_edge": edge}},
                                                         "daily": {str(y): [daily if y == M.YEARS[-1] else 50.0, None] for y in M.YEARS}}}
    M.check_against_archive(one, dailies, stored())
    for kw, msg in (({"present": 21}, "present days"), ({"edge": 4}, "region-edge days"), ({"daily": 50.01}, "daily cores")):
        try:
            M.check_against_archive(one, dailies, stored(**kw)); assert False
        except SystemExit as ex:
            assert msg in str(ex), str(ex)
    M.check_record_hashes({"a.json.gz": "x"}, {"records": {"a.json.gz": "x"}})
    for gen in ({"records": {"a.json.gz": "y"}}, {}):
        try:
            M.check_record_hashes({"a.json.gz": "x"}, gen); assert False
        except SystemExit as ex:
            assert "month records differ" in str(ex)
    arch = {"A": {"present_days": 30}}
    text = M.posthoc_text({"S2|15 to 19.5 N": s}, {"W0": s}, o)
    assert "| W0 | 4 | 1 | 1, 1, 1 |" in text and "median_amplitude, threshold-dependent yes" in text
    assert "| S2, 15 to 19.5 N | 4 |" in text and _table_blocks(text) == [[7] * 4], "six cells in every reader row"


def test_season_shift_baseline():
    M = _load()
    Y, b = M.YEARS, "20 to 24.5 N"
    assert M.exact_median([Fraction(3), Fraction(1), Fraction(2)]) == 2 and M.exact_median([Fraction(v) for v in (1, 2, 3, 4)]) == Fraction(5, 2) and M.exact_median([]) is None
    own = lambda i: Fraction(90 + i, 2)
    cores_by = {("S2", y, b): [None] * 10 + [{"core": own(i)}] * 3 + [None] * 109 for i, y in enumerate(Y)}
    cores_by[("S3", Y[0], b)] = [None] * 122
    sh = M.shifted_cores(cores_by, 1)
    assert sh[("S2", Y[-1], b)] is cores_by[("S2", Y[0], b)] and sh[("S2", Y[0], b)] is cores_by[("S2", Y[1], b)] and ("S3", Y[0], b) not in sh, "cyclic over the seasons, S2 only"
    hist = {(y, b): [{"days": [10, 11, 12], "lons": [float(own(i))] * 3}] for i, y in enumerate(Y)}
    assert M.member_distances({(Y[0], b): [{"days": [10, 11, 12, 13], "lons": [45.5, 47.0, 58.0, 45.0]}]}, cores_by) == [Fraction(1, 2), Fraction(2)], "outside the window or on an absent day not counted"
    base = M.season_shift_baseline(hist, cores_by)
    o = base["pooled"]["observed"]
    assert (o["member_days"], o["near"], o["histories"], o["half_near"], o["share"], o["median_distance"]) == (72, 72, 24, 24, 1.0, 0.0)
    assert base["pooled"]["shifts"][0]["near"] == 69 and base["pooled"]["shifts"][0]["member_days"] == 72, "one shift moves 23 seasons by half a degree and wraps the last"
    sm = base["pooled"]["summary"]
    assert sm["share"]["rank"] == 0 and sm["median_distance"]["rank"] == 0 and base["reading"].startswith("the members lie nearer")
    assert base[b]["observed"] == o and base["15 to 19.5 N"]["summary"]["share"]["rank"] is None, "each band separately, a band with no members unavailable"
    same = {key: [None] * 10 + [{"core": Fraction(45)}] * 3 + [None] * 109 for key in cores_by if key[0] == "S2"}
    flat = M.season_shift_baseline({(y, b): [{"days": [10, 11, 12], "lons": [45.0] * 3}] for y in Y}, same)
    assert flat["pooled"]["summary"]["share"]["rank"] == 23 and flat["pooled"]["summary"]["median_distance"]["rank"] == 23, "ties count as at or beyond"
    assert flat["reading"].startswith("the proximity is matched")
    lone = {key: ([None] * 10 + [{"core": Fraction(45)}] * 3 + [None] * 109 if key[1] == Y[0] else [None] * 122) for key in same}
    gap = M.season_shift_baseline({(Y[0], b): [{"days": [10], "lons": [45.0]}]}, lone)
    assert gap["pooled"]["summary"]["share"]["rank"] is None and gap["reading"] is None, "a shift with no member-days leaves the rank unavailable"
    mixed = M.season_shift_baseline(hist, {k: (v if k[1] != Y[5] else [None] * 10 + [{"core": Fraction(60)}] * 3 + [None] * 109) for k, v in cores_by.items()})
    assert M.baseline_reading({"share": {"rank": 0, "inside_range": False}, "median_distance": {"rank": 2, "inside_range": True}}) is None, "mixed ranks carry no summary"
    assert mixed["pooled"]["observed"]["near"] == 69
    M.check_observed(base, {"member_days_in_window": 72, "near_core": 72, "histories_in_window": 24, "histories_half_near": 24})
    try:
        M.check_observed(base, {"member_days_in_window": 72, "near_core": 71, "histories_in_window": 24, "histories_half_near": 24}); assert False
    except SystemExit as ex:
        assert "does not reproduce" in str(ex)
    text = M.baseline_text(base)
    assert _table_blocks(text) == [[9] * 6] and "| pooled | 72 of 72 | 24 of 24 |" in text and "reading under section 12, the members lie nearer" in text


def test_baseline_does_not_separate_year_from_day():
    """Section 12's qualification. Each year's core fixed at its own longitude, with one weaker
    westward-moving contribution per year read by the real anomaly and linker readers, gives
    rank 0 on both statistics, and moving that contribution 40 days later changes nothing, so
    the own-year pairing's ranks do not isolate a day-specific association."""
    M = _load()
    centers = [47.5, 48, 47, 46.5, 56, 53.5, 55, 51, 56.5, 52, 49.5, 49, 53, 48.5, 45.5, 45, 51.5, 52.5, 46, 50, 54, 50.5, 55.5, 54.5]
    offsets = [5, 2, 1.5, 1, 0.5, -0.5, -1, -1.5, -4.5]
    band = "15 to 19.5 N"
    col = lambda x: list(M.LONS).index(x)

    def build(start):
        histories, cores = {}, {}
        for year, center in zip(M.YEARS, centers):
            H = np.zeros((122, len(M.LONS)), dtype=np.int64)
            H[:, col(center)] = -2000
            path = [center + d for d in offsets]
            for day, x in enumerate(path, start):
                H[day, col(x)] = -1200
            minima = M.HI.rep_minima("RB9", H, M.A.month_index(), M.LONS)
            histories[(year, band)] = [h for h in M.HI.link(minima, "LN") if M.HI.classify(h) == "westward"]
            assert len(histories[(year, band)]) == 1 and histories[(year, band)][0]["lons"] == path
            cores[("S2", year, band)] = M.season_measures(H, M.window_cols((44.5, 57.0)))
            assert all(d["core"] == center for d in cores[("S2", year, band)]), "the core never moves"
        return M.season_shift_baseline(histories, cores)

    base = build(50)
    assert build(90) == base, "the moving contribution's timing does not enter the result"
    o, sm = base["pooled"]["observed"], base["pooled"]["summary"]
    assert (o["near"], o["member_days"], o["median_distance"]) == (138, 190, 1.5)
    assert sm["share"]["rank"] == 0 and sm["median_distance"]["rank"] == 0 and base["reading"].startswith("the members lie nearer")


def test_within_season_lag_control(tmp_path):
    M = _load()
    Y, b = M.YEARS, "20 to 24.5 N"
    assert M.LAGS == tuple(range(-20, -9)) + tuple(range(10, 21)) and len(M.LAGS) == 22
    days = [{"core": Fraction(d)} for d in range(122)]
    cores_by = {("S2", Y[0], b): days, ("S3", Y[0], b): days}
    lag = M.lagged_cores(cores_by, 10)
    assert list(lag) == [("S2", Y[0], b)] and lag[("S2", Y[0], b)][0]["core"] == 10 and lag[("S2", Y[0], b)][111]["core"] == 121 and lag[("S2", Y[0], b)][112] is None
    back = M.lagged_cores(cores_by, -20)[("S2", Y[0], b)]
    assert back[19] is None and back[20]["core"] == 0, "days outside the season are absent at that lag"
    for case, expect, ranks, own in (("C1", "matched", 22, (138, 190)), ("C2", "nearer", 0, (190, 190))):
        h, cores = M.lag_construction(case)
        assert all(len(v) == 1 and v[0]["lons"] == [M.LAG_CENTERS[Y.index(y)] + o for o in M.LAG_OFFSETS] for (y, _), v in h.items()), "the real readers find one westward history on its path"
        r = M.lag_control(h, cores)
        sm, o = r["pooled"]["summary"], r["pooled"]["observed"]
        assert r["reading"] == M.LAG_READINGS[expect] and sm["share"]["rank"] == ranks and sm["median_distance"]["rank"] == ranks and (o["near"], o["member_days"]) == own
        assert [x["lag"] for x in r["pooled"]["lags"]] == list(M.LAGS)
    cons = M.check_lag_constructions()
    assert cons["C1"]["reading"] == M.LAG_READINGS["matched"] and cons["C2"]["share_rank"] == 0
    M.LAG_CASES["C1"] = -2400
    try:
        M.check_lag_constructions(); assert False
    except SystemExit as ex:
        assert "construction C1" in str(ex) and "expected matched reading of section 15" in str(ex), "the declared rule's check refuses first, before the matched rule's"
    try:
        M.main(["lagcontrol", "--records", str(tmp_path), "--histories", str(tmp_path / "absent.json"), "--archive", str(tmp_path / "absent.json"), "--out-dir", str(tmp_path)]); assert False
    except SystemExit as ex:
        assert "construction C1" in str(ex), "the command runs the constructions before it opens any input"
    assert list(tmp_path.iterdir()) == [], "and writes nothing"
    M.LAG_CASES["C1"] = -1200
    edge = {(Y[0], b): [{"days": [115, 116], "lons": [50.0, 50.0]}]}
    flat = {("S2", Y[0], b): [{"core": Fraction(50)}] * 122}
    r = M.lag_control(edge, flat)
    assert [x["member_days"] for x in r["pooled"]["lags"]] == [2] * 11 + [2 if 116 + L < 122 else (1 if 115 + L < 122 else 0) for L in range(10, 21)]
    assert r["pooled"]["summary"]["share"]["rank"] is None and r["reading"] is None, "a lag with no member-days leaves the rank unavailable"
    text = M.lag_text(M.lag_control(*M.lag_construction("C2")), cons)
    assert _table_blocks(text) == [[9] * 6] and "| pooled | 190 of 190 |" in text and "of 22 |" in text
    assert "reading under section 15, the members lie nearer" in text and text.count("\n- C") == 4


def test_matched_lag_comparison():
    M = _load()
    Y, b = M.YEARS, "20 to 24.5 N"
    cons = M.check_lag_constructions()
    assert set(cons) == {"C1", "C2", "C3", "C4"}
    assert cons["C3"]["reading"] == M.LAG_READINGS["nearer"] and cons["C3"]["share_rank"] == 0 and cons["C3"]["distance_rank"] == 0, "the declared rule's failure, kept on record"
    assert cons["C3"]["matched_reading"] == M.MATCHED_READINGS["matched"] and cons["C3"]["matched_summary"] == {"share_at_or_beyond": 22, "distance_at_or_beyond": 22}
    assert cons["C4"]["reading"] == M.LAG_READINGS["matched"] and cons["C4"]["matched_reading"] == M.MATCHED_READINGS["matched"], "the same paths inside the season"
    assert cons["C1"]["matched_reading"] == M.MATCHED_READINGS["matched"] and cons["C2"]["matched_reading"] == M.MATCHED_READINGS["nearer"]
    h3, c3 = M.edge_construction(M.EDGE_STARTS["C3"])
    assert [x["lons"] for x in h3[(Y[0], "15 to 19.5 N")]] == [list(M.EDGE_NEAR), list(M.EDGE_FAR), list(M.EDGE_NEAR)] and all(d["core"] == 51 for d in c3[("S2", Y[0], "15 to 19.5 N")])
    days = [{"core": Fraction(50)}] * 122
    days = days[:30] + [None] + days[31:]
    hist = {(Y[0], b): [{"days": [30, 60, 115], "lons": [50.0, 52.0, 50.0]}]}
    m = M.matched_lag_comparison(hist, {("S2", Y[0], b): days})
    row = {r["lag"]: r for r in m["pooled"]["lags"]}
    assert (row[10]["common"], row[10]["own_only"], row[10]["lag_only"]) == (1, 1, 1), "day 60 on both, day 115 lagged out of the season, day 30 absent on its own day"
    assert (row[-10]["common"], row[-10]["own_only"], row[-10]["lag_only"]) == (2, 0, 1)
    assert row[-10]["own_median"] == 1.0 and row[-10]["own_near"] == 1 and row[10]["own_median"] == 2.0 and row[10]["own_near"] == 0
    assert m["pooled"]["summary"] == {"share_at_or_beyond": 22, "distance_at_or_beyond": 22} and m["reading"] == M.MATCHED_READINGS["matched"], "a flat core is matched at every lag, ties counting"
    lone = M.matched_lag_comparison({(Y[0], b): [{"days": [115], "lons": [50.0]}]}, {("S2", Y[0], b): [{"core": Fraction(50)}] * 122})
    assert lone["pooled"]["summary"] == {"share_at_or_beyond": None, "distance_at_or_beyond": None} and lone["reading"] is None, "a lag with no common member-days leaves the counts unavailable"
    assert M.matched_reading({"share_at_or_beyond": 0, "distance_at_or_beyond": 3}) is None and M.matched_reading({"share_at_or_beyond": 22, "distance_at_or_beyond": 21}) is None
    M.MATCHED_EXPECTED["C3"] = "nearer"
    try:
        M.check_lag_constructions(); assert False
    except SystemExit as ex:
        assert "construction C3" in str(ex) and "section 17" in str(ex)
    M.MATCHED_EXPECTED["C3"] = "matched"
    hc, cc = M.lag_construction("C2")
    text = M.lag_text(M.lag_control(hc, cc), cons, M.matched_lag_comparison(hc, cc))
    assert _table_blocks(text) == [[9] * 6, [9] * 6] and "reading under section 17, on the same member-days, the members lie nearer" in text
    assert "| pooled | 190 to 190 | 0 to 0, 0 to 0 | 1.000 to 1.000 |" in text
