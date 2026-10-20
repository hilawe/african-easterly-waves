"""The strip-following measure's rules (EASTERN_STRIP_FOLLOWING_2026-10-20.md, sections 2 to 4)."""

import datetime as dt
import importlib.util
import json
import os
from fractions import Fraction

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load():
    spec = importlib.util.spec_from_file_location("strips_under_test", os.path.join(ROOT, "scripts", "eastern_strip_following.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _row(S, cells):
    r = np.zeros(len(S.LONS), dtype=np.int64)
    for x, v in cells.items():
        r[S.col_of(x)] = v
    return r


def test_runs_and_linking():
    S = _load()
    runs = S.day_runs(_row(S, {50.0: -2000, 50.5: -2400, 51.0: -2400, 51.5: -1599, 60.0: -1600}))
    assert [(S.LONS[r["first"]], S.LONS[r["last"]]) for r in runs] == [(50.0, 51.0), (60.0, 60.0)], "-4 m/s qualifies and -3.9975 breaks a run"
    assert runs[0]["centroid"] == Fraction(100 * 2000 + 101 * 2400 + 102 * 2400, 2 * 6800) and S.LONS[runs[0]["strongest"]] == 50.5, "exact centroid, westernmost strongest on a tie"
    H = np.array([
        _row(S, {50.0: -2000, 50.5: -2000, 60.0: -2000}),
        _row(S, {50.5: -2000, 51.0: -2000, 65.5: -2000}),
        _row(S, {51.0: -2000, 52.0: -2000, 71.0: -2000}),
        _row(S, {}),
        _row(S, {51.0: -2000}),
    ])
    strips, by_day = S.link_strips(H)
    a = by_day[0][0][2]
    assert by_day[1][0][2] == a and by_day[2][0][2] == a, "a shared column continues a strip"
    b = by_day[0][1][2]
    assert by_day[1][1][2] == b and strips[b]["links"][1] == "fallback", "60 to 65.5 has no shared column and 5.5 degrees continues it"
    assert by_day[2][2][2] == b and strips[b]["links"][2] == "fallback", "65.5 to 71 is 5.5 degrees, inside the fallback"
    assert by_day[2][1][2] not in (a, b) and strips[by_day[2][1][2]]["born"] == "new", "52.0 overlaps nothing on day 1 and its nearest strip is already continued"
    assert by_day[3] == [] and by_day[4][0][2] not in (a, b) and strips[a]["ended"] == "end" and strips[b]["ended"] == "end", "a day with no run ends every strip"
    assert strips[-1]["ended"] == "season"


def test_fallback_limit_and_split_merge_events():
    S = _load()
    H = np.array([_row(S, {60.0: -2000}), _row(S, {66.0: -2000}), _row(S, {72.5: -2000})])
    strips, by_day = S.link_strips(H)
    assert by_day[1][0][2] == by_day[0][0][2] and by_day[2][0][2] != by_day[1][0][2], "exactly 6 degrees links and 6.5 does not"
    st, bd = S.link_strips(np.array([_row(S, {50.0: -2000, 50.5: -2000}), _row(S, {51.0: -2000, 51.5: -2000})]))
    assert bd[1][0][2] == bd[0][0][2] and st[0]["links"] == [None, "fallback"], "adjacent runs share no cell, so they link by the fallback, not by overlap"
    H = np.array([
        _row(S, {48.0: -2000, 48.5: -2000, 49.0: -2000, 49.5: -2000, 50.0: -2000}),
        _row(S, {48.0: -2400, 48.5: -2000, 49.0: -2000, 50.0: -2000}),
        _row(S, {48.0: -2000, 48.5: -2000, 49.0: -2000, 49.5: -2000, 50.0: -2000}),
    ])
    strips, by_day = S.link_strips(H)
    parent = by_day[0][0][2]
    assert by_day[1][0][2] == parent and strips[by_day[1][1][2]]["born"] == "split" and strips[by_day[1][1][2]]["split_from"] == parent, "the larger overlap continues, the other run is born by a split"
    assert strips[parent]["events"][1] == [("split_off", by_day[1][1][2])]
    piece = by_day[1][1][2]
    assert strips[piece]["ended"] == "merge" and strips[piece]["merged_into"] == parent and strips[parent]["events"][2] == [("absorbed", piece)]
    H = np.array([_row(S, {50.0: -2000, 50.5: -2000, 51.0: -2000}), _row(S, {49.0: -2000, 49.5: -2000, 50.0: -2000, 50.5: -2000}), _row(S, {})])
    strips, _ = S.link_strips(H)
    assert len(strips) == 1 and strips[0]["links"] == [None, "overlap"] and S.strip_at(S.link_strips(H)[1], 1, S.col_of(49.0)) == 0 and S.strip_at(S.link_strips(H)[1], 1, S.col_of(51.0)) is None


def test_overlap_order():
    S = _load()
    H = np.array([
        _row(S, {50.0: -2000, 50.5: -2000, 51.0: -2000, 51.5: -2000}),
        _row(S, {49.0: -2000, 49.5: -2000, 50.0: -2000, 51.0: -2000, 51.5: -2000, 52.0: -2000, 52.5: -2000}),
    ])
    strips, by_day = S.link_strips(H)
    left, right = by_day[1]
    assert (S.LONS[left[0]], S.LONS[right[0]]) == (49.0, 51.0)
    assert right[2] == by_day[0][0][2] and strips[left[2]]["born"] == "split", "two cells of overlap beat one"


def _steps(S, rows):
    H = np.array(rows)
    return S.s2_steps(H, S.link_strips(H)[1])


def test_q1_steps_and_summary():
    S = _load()
    st = _steps(S, [_row(S, {50.0: -2000}), _row(S, {50.5: -2000}), _row(S, {50.5: -1000, 55.0: -2000}), _row(S, {}), _row(S, {54.0: -2000}), _row(S, {54.0: -2000, 46.0: -3000})])
    assert [(s["day"], s["step"], s["continuation"]) for s in st] == [(1, Fraction(1, 2), True), (2, Fraction(9, 2), True), (5, Fraction(8), False)], "an absent day breaks the chain, 4.5 degrees is inside the fallback and 8 is not"
    st2 = _steps(S, [_row(S, {50.0: -2000}), _row(S, {56.5: -2000})])
    assert st2[0]["continuation"] is False, "6.5 degrees is beyond the fallback"
    steps = [{"step": Fraction(k), "continuation": k < 18} for k in range(20)] + [{"step": Fraction(17), "continuation": True}]
    sm = S.switch_summary(steps)
    assert (sm["pairs"], sm["continuations"], sm["changes"]) == (21, 19, 2)
    assert sm["large_cutoff"] == 17.0 and sm["large"] == 4 and sm["large_changes"] == 2, "ceil(21/10) = 3 largest, and the tie at 17 joins them"
    assert S.switch_summary([])["large"] == 0


def _hist(days, lons):
    return {"days": list(days), "lons": list(lons)}


def test_q2_paths_segments_and_classes():
    S = _load()
    standing = {x / 2: -2000 for x in range(98, 103)}
    H = np.array([
        _row(S, {66.0: -2000, 66.5: -2000, **standing}),
        _row(S, {62.0: -2000, 62.5: -2000, **standing}),
        _row(S, {58.0: -2000, 58.5: -2000, **standing}),
        _row(S, {x / 2: -2000 for x in range(98, 118)}),
        _row(S, {x / 2: -2000 for x in range(98, 111)}),
        _row(S, {49.0: -2000, 49.5: -2000, 51.0: -2000, 51.5: -2000, 52.0: -2000}),
        _row(S, {44.0: -2000, 51.0: -2000}),
    ])
    strips, by_day = S.link_strips(H)
    h = _hist(range(7), [66.0, 62.0, 58.0, 54.0, 50.0, 49.0, 44.0])
    path = S.history_path(h, strips, by_day)
    assert [m["into"] for m in path] == ["start", "same", "same", "merge", "same", "split", "same"]
    segs = S.segments(path, strips)
    assert segs == [(0, 2), (3, 4), (5, 6)]
    mc = S.member_classes(path, strips)
    assert mc == ["with_new"] * 3 + ["through"] * 2 + ["with_new"] * 2, "carried in from the east, through the merged strip, out on a strip not yet resident five days"
    assert S.history_path(_hist([0, 1], [50.0, 60.0]), strips, by_day)[1]["into"] == "off"
    assert [m["into"] for m in S.history_path(_hist([2, 3], [50.0, 58.5]), strips, by_day)] == ["start", "same"], "the wider standing strip keeps its identity at the merge"
    across = S.history_path(_hist([2, 3, 4], [50.0, 50.0, 50.0]), strips, by_day)
    assert [m["into"] for m in across] == ["start", "same", "same"] and S.segments(across, strips) == [(0, 0), (1, 2)], "the day the strip absorbs another starts a new segment"
    assert S.member_classes(S.history_path(_hist([0, 1], [50.0, 50.0]), strips, by_day), strips) == ["undetermined"] * 2, "a member that stays put is undetermined, not carried"
    HJ = np.array([_row(S, {50.0: -2000, 50.5: -2000, 51.0: -2000, 54.0: -2000, 54.5: -2000, 55.0: -2000, 55.5: -2000}),
                   _row(S, {50.0: -2000, 50.5: -2000, 51.0: -2000, 53.5: -2000, 54.0: -2000, 54.5: -2000, 55.0: -2000, 55.5: -2000}),
                   _row(S, {x / 2: -2000 for x in range(100, 112)})])
    stj, bdj = S.link_strips(HJ)
    assert stj[bdj[0][0][2]]["merged_into"] == bdj[0][1][2], "the narrower strip merges into the wider on day 2"
    assert [m["into"] for m in S.history_path(_hist([0, 1], [50.0, 54.0]), stj, bdj)] == ["start", "jump"], "a jump to the strip the old one merges into a day later is a jump, not a merge"
    HS = np.array([_row(S, {x / 2: -2000 for x in range(98, 105)})] * 2 + [_row(S, {49.0: -2000, 49.5: -2000, 51.0: -2000, 51.5: -2000, 52.0: -2000})] * 2)
    sts, bds = S.link_strips(HS)
    assert [m["into"] for m in S.history_path(_hist([1, 2, 3], [51.0, 51.0, 49.0]), sts, bds)] == ["start", "same", "jump"], "a step onto a strip split off the day before is a jump, not a split"
    ends = S.history_path(_hist([0, 1, 2], [52.0, 51.0, 49.0]), sts, bds)
    assert [m["into"] for m in ends] == ["start", "same", "split"] and S.member_classes(ends, sts) == ["through", "through", "undetermined"], "a segment that only ends at a split is a passage through"
    strips2, by_day2 = S.link_strips(np.array([_row(S, {50.0: -2000, 50.5: -2000, 51.0: -2000, 51.5: -2000})] * 3))
    within = S.member_classes(S.history_path(_hist([0, 1, 2], [51.5, 51.0, 50.0]), strips2, by_day2), strips2)
    assert within == ["within"] * 3, "a minimum moving west inside a strip that stays put"
    assert S.member_classes(S.history_path(_hist([0], [50.0]), strips2, by_day2), strips2) == ["undetermined"]
    assert S.member_classes(S.history_path(_hist([0, 1], [50.0, 51.0]), strips2, by_day2), strips2) == ["undetermined"] * 2, "no westward displacement"
    H5 = np.array([_row(S, {50.0: -2000, 50.5: -2000, 51.0: -2000})] * 6 + [_row(S, {49.5: -2000, 50.0: -2000, 50.5: -2000})])
    st5, bd5 = S.link_strips(H5)
    assert S.member_classes(S.history_path(_hist([5, 6], [51.0, 50.0]), st5, bd5), st5) == ["with_resident"] * 2, "five prior days inside the window make the strip resident"
    H4 = np.array([_row(S, {50.0: -2000, 50.5: -2000, 51.0: -2000})] * 5 + [_row(S, {49.5: -2000, 50.0: -2000, 50.5: -2000})])
    st4b, bd4b = S.link_strips(H4)
    assert S.member_classes(S.history_path(_hist([4, 5], [51.0, 50.0]), st4b, bd4b), st4b) == ["with_new"] * 2, "four prior days are too few"
    H6 = np.array([_row(S, {57.5: -2000, 58.0: -2000, 58.5: -2000})] * 6 + [_row(S, {57.0: -2000, 57.5: -2000, 58.0: -2000})])
    st6, bd6 = S.link_strips(H6)
    assert S.member_classes(S.history_path(_hist([5, 6], [58.5, 57.5]), st6, bd6), st6) == ["with_new"] * 2, "resident means inside S2's window, and 58 E is outside"
    st3, bd3 = S.link_strips(np.array([_row(S, {50.0: -2000, 50.5: -2000, 51.0: -2000}), _row(S, {49.5: -2000, 50.0: -2000, 50.5: -2000})]))
    assert S.member_classes(S.history_path(_hist([0, 1], [51.0, 50.0]), st3, bd3), st3) == ["with_new"] * 2, "the strip moves half a degree, exactly half the member's one degree"
    st4, bd4 = S.link_strips(np.array([_row(S, {50.0: -2000, 50.5: -2000, 51.0: -2000}), _row(S, {49.5: -2000, 50.0: -2000, 50.5: -2000, 51.0: -1000})]))
    assert S.member_classes(S.history_path(_hist([0, 1], [51.0, 49.5]), st4, bd4), st4) == ["within"] * 2, "half a degree is less than half of one and a half"


def test_history_strips_counts():
    S = _load()
    Y = S.YEARS
    H = np.array([_row(S, {50.0: -2000, 50.5: -2000})] * 3)
    seasons = {(Y[0], "20 to 24.5 N"): S.link_strips(H), (Y[0], "10 to 14.5 N"): S.link_strips(H)}
    hist = {(Y[0], "20 to 24.5 N"): [_hist([0, 1, 2], [60.0, 50.5, 50.0])], (Y[0], "10 to 14.5 N"): [_hist([0, 1], [50.5, 50.0])]}
    q = S.history_strips(hist, seasons, s2_only=True)
    assert q["member_days"] == 2 and q["by_class"]["none"] == 0 and q["histories"] == 1, "outside the window and outside S2's bands not counted"
    assert q["transitions"] == {"same": 1, "merge": 0, "split": 0, "jump": 0, "on": 1, "off": 0}
    allq = S.history_strips(hist, seasons)
    assert allq["member_days"] == 5 and allq["histories"] == 2 and allq["history_class"]["mixed"] == 1


def test_rule_gate_and_readings(tmp_path):
    S = _load()
    res = json.loads(json.dumps(S.run_constructions(), default=float))
    v = S.apply_rule(res)
    assert v == {"adequate": True, "failures": []}
    bad = json.loads(json.dumps(res)); bad["K4"]["histories_s2"]["by_class"]["none"] -= 1; bad["K4"]["histories_s2"]["by_class"]["within"] += 1
    assert S.apply_rule(bad)["failures"] == ["K4 none_at_least 1 not met (0.9947)"], "the bars are inclusive and exact"
    edge = json.loads(json.dumps(res)); edge["K1"]["histories_s2"]["by_class"] = {"with_resident": 50, "with_new": 45, "through": 0, "within": 5, "undetermined": 0, "none": 0}; edge["K1"]["histories_s2"]["member_days"] = 100
    assert S.apply_rule(edge)["adequate"] is True, "exactly on the K1 bars passes"
    edge["K1"]["histories_s2"]["by_class"]["with_resident"] = 49; edge["K1"]["histories_s2"]["by_class"]["within"] = 6
    assert "K1 resident_at_least 1/2 not met (0.49)" in S.apply_rule(edge)["failures"]
    empty = json.loads(json.dumps(res)); empty["K6"]["histories_s2"]["member_days"] = 0
    assert "K6 resident_at_least unavailable" in S.apply_rule(empty)["failures"]
    p = tmp_path / "c.json"
    stale = json.loads(json.dumps(res)); stale["K2"]["switch"]["large_changes"] = 0
    for rec, run, msg in (({"generated_by": {"sha256": "x"}, "results": res, "verdict": v}, lambda: res, "different version"),
                          ({"generated_by": {"sha256": S.SCRIPT_SHA256_AT_LOAD}, "results": res, "verdict": v}, lambda: stale, "differs from a fresh run"),
                          ({"generated_by": {"sha256": S.SCRIPT_SHA256_AT_LOAD}, "results": stale, "verdict": v}, lambda: stale, "do not pass the rule"),
                          ({"generated_by": {"sha256": S.SCRIPT_SHA256_AT_LOAD}, "results": res, "verdict": {"adequate": True, "failures": ["x"]}}, lambda: res, "stored summary disagrees")):
        p.write_text(json.dumps(rec))
        try:
            S.check_constructions(str(p), rerun=run); assert False
        except SystemExit as ex:
            assert msg in str(ex), str(ex)
    p.write_text(json.dumps({"generated_by": {"sha256": S.SCRIPT_SHA256_AT_LOAD}, "results": res, "verdict": v}))
    assert S.check_constructions(str(p), rerun=lambda: res)["verdict"]["adequate"] is True
    q1 = {"large": 10, "large_changes": 5}
    mk = lambda r, a, t, n: {"member_days": 10, "by_class": {"with_resident": r, "with_new": a, "through": t, "within": 10 - r - a - t - n, "undetermined": 0, "none": n}}
    assert S.readings(q1, mk(5, 0, 0, 0)) == {"q1": "the largest steps of S2's core are mostly changes of strip", "q2": "most in-window member-days of the westward histories lie on a strip resident in S2's window that moves with them"}
    assert S.readings({"large": 10, "large_changes": 4}, mk(4, 2, 3, 0))["q1"].endswith("continuations of one strip")
    assert S.readings(q1, mk(4, 2, 3, 0))["q2"].startswith("most in-window member-days of the westward histories lie on strips new")
    assert S.readings(q1, mk(4, 2, 2, 0))["q2"] is None and S.readings({"large": 0, "large_changes": 0}, mk(0, 0, 0, 10)) == {"q1": None, "q2": None}


def _records(S, values):
    recs = {}
    for y in S.YEARS[:2]:
        for m in S.MONTHS:
            n = S.A.MONTH_DAYS[m]
            times = [f"{(dt.date(y, m, 1) + dt.timedelta(days=d)).isoformat()}T{h:02d}" for d in range(n) for h in (0, 6, 12, 18)]
            bands = {b: np.full((len(times), len(S.LONS)), values.get(b, 0.0)).tolist() for b in S.SC.V.BANDS}
            recs.setdefault(y, {})[m] = {"year": y, "month": m, "times": times, "lons": S.LONS.tolist(), "bands": list(S.SC.V.BANDS), "band_means_700": bands}
    return recs


def test_archive_keeps_bands_apart(monkeypatch):
    S = _load()
    monkeypatch.setattr(S, "YEARS", S.YEARS[:2])
    by = _records(S, {"20 to 24.5 N": -5.0, "25 to 29.5 N": 0.0, "15 to 19.5 N": -5.0})
    hist = {(S.YEARS[0], "20 to 24.5 N"): [_hist([10, 11, 12], [51.0, 50.5, 50.0])], (S.YEARS[0], "25 to 29.5 N"): [_hist([10, 11, 12], [51.0, 50.5, 50.0])]}
    res = S.archive_results(by, hist)
    assert res["q2"]["20 to 24.5 N"]["by_class"]["within"] == 3 and res["q2"]["25 to 29.5 N"]["by_class"]["none"] == 3, "each band's members on that band's strips"
    assert res["q2"]["20 to 24.5 N"]["member_days"] == 3 and res["q2"]["25 to 29.5 N"]["member_days"] == 3, "and only that band's histories"
    assert res["strips"]["20 to 24.5 N"]["strips"] == 2 and res["strips"]["25 to 29.5 N"]["strips"] == 0
    assert res["q1"]["25 to 29.5 N"]["pairs"] == 0 and res["q1"]["20 to 24.5 N"]["pairs"] == 2 * 121 and res["q1"]["pooled"]["pairs"] == 4 * 121
    text = S.archive_text(res)
    assert "| inside S2's window, pooled | 6 |" in text and "reading of question 2 under section 4, none" in text


def test_a_change_of_strip_is_not_a_physical_switch():
    """Section 7.3's two counterexamples, from the review: a fixed run of qualifying flow whose
    deep end is exchanged keeps one strip with a 2-degree core step, and a single feature
    translated 7 degrees between two days reads as a change of strip."""
    S = _load()
    row = lambda deep: np.array([(-10000 if abs(x - deep) < 1e-9 else (-1600 if 46 <= x <= 56 else 0)) for x in S.LONS], dtype=np.int64)
    H = np.array([row(56.0), row(46.0)])
    strips, by_day = S.link_strips(H)
    st = S.s2_steps(H, by_day)
    assert len(strips) == 1 and st[0]["continuation"] is True and st[0]["step"] == 2, "the same run on both days, the core moved by the weights alone"
    t = np.arange(S.A.SEASON_DAYS * 4) / 4.0
    V = np.zeros((len(t), len(S.LONS)))
    for k, tk in enumerate(t):
        V[k] += S.A.gauss(S.LONS, 54.0 if tk < 1.0 else 47.0, 10.0)
    Hg = S.A.hundredths_from_samples(np.round(V, 2))[:2]
    stg = S.s2_steps(Hg, S.link_strips(Hg)[1])
    assert stg[0]["continuation"] is False and stg[0]["step"] == 7, "one feature moved 7 degrees, beyond overlap and the fallback, reads as a change of strip"
