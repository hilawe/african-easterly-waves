"""Tests for the position of the westward histories' southerly lobe (`EASTERN_LOBE_POSITION_2026-10-22.md`)."""

import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "scripts"))
import eastern_lobe_position as P  # noqa: E402

C, H, J, L, A = P.C, P.H, P.J, P.L, P.A


def profile():
    Q = np.zeros(len(H.DX)); Q[H.DX == 0] = -4.0; Q[H.DX == -1.0] = -2.0; Q[H.DX == 6.0] = 1.6; Q[H.DX == 9.0] = 3.0; Q[H.DX == -5.0] = 0.8
    return Q.tolist()


def test_measured_parts_take_the_core_and_the_lobe_within_two_to_eight_degrees(monkeypatch):
    monkeypatch.setitem(H.MEASURED, "profile", profile())
    parts = P.measured_parts()
    assert np.isclose(parts["lobe"], 0.4)                        # 1.6 at 6 E over 4, the 3.0 at 9 E lies outside
    assert parts["core"].max() == 0.0 and np.isclose(parts["core"][H.DX == 0][0], -1.0) and np.isclose(parts["core"][H.DX == -1.0][0], -0.5)
    x = np.arange(30.0, 70.5, 0.5)
    v = P.core_shape(x, 50.0, 6.0, parts["core"])
    assert np.isclose(v[x == 50.0][0], -6.0) and np.isclose(v[x == 49.0][0], -3.0) and v.max() == 0.0


def tracks():
    return [{"first_lon": 60.0, "speed": -2.0, "days": 5, "injected": {"measured": 6.0}},
            {"first_lon": 55.0, "speed": -3.0, "days": 4, "injected": {"measured": 4.0}}]


def test_travel_is_the_coherence_tests_traveler(monkeypatch):
    monkeypatch.setitem(H.MEASURED, "profile", profile())
    V, cen = P.build(tracks(), np.random.default_rng(9), "travel", P.measured_parts())
    V2, cen2 = C.build(tracks(), np.random.default_rng(9), 0.0)
    assert np.array_equal(V, V2) and all(np.array_equal(a, b, equal_nan=True) for a, b in zip(cen, cen2))
    with pytest.raises(SystemExit, match="unknown construction"):
        P.build(tracks(), np.random.default_rng(9), "drift", P.measured_parts())


def test_the_pinned_lobe_stands_east_of_the_first_longitude(monkeypatch):
    monkeypatch.setitem(H.MEASURED, "profile", profile())
    parts = P.measured_parts()
    tr = tracks()[:1]
    V, cen = P.build(tr, np.random.default_rng(9), "pinned", parts)
    k = np.flatnonzero(np.isfinite(np.repeat(cen[0], 4)))
    t0 = k[0] // 4
    assert t0 >= 11                                              # every index below lies inside the season
    lobe_col = np.flatnonzero(np.isclose(J.LONS, 66.0))[0]
    rows = np.flatnonzero(np.abs(V[:, lobe_col]) > 0)
    assert rows[0] == (t0 - 2) * 4 and rows[-1] == (t0 + 5 + 2) * 4 - 1
    before = V[(t0 - 1) * 4]
    assert np.isclose(before.max(), 0.4 * 6.0) and J.LONS[np.argmax(before)] == 66.0 and before.min() >= 0.0
    during = V[(t0 + 3) * 4]
    assert np.isclose(during[lobe_col], 0.4 * 6.0, atol=0.05) and np.isclose(during.min(), -6.0, atol=0.05)
    Vr, _ = P.build(tr, np.random.default_rng(9), "pinned_relax", parts)
    assert np.isclose(Vr[(t0 - 10) * 4, lobe_col], 2 * 0.4 * 6.0) and Vr[(t0 - 11) * 4, lobe_col] == 0.0
    assert np.isclose(Vr[(t0 + 2) * 4, lobe_col], V[(t0 + 2) * 4, lobe_col])


def test_b0_rows_are_the_anomaly_against_the_other_seasons():
    mo = A.month_index()
    rng = np.random.default_rng(1)
    Hs = {y: rng.integers(-800, 800, size=(A.SEASON_DAYS, len(A.LONS))).astype(np.int64) for y in (1983, 1984, 1985)}
    rows = P.b0_rows(Hs, mo)
    d = 40; m = mo[d]
    others = [Hs[y][mo == m].mean(axis=0) for y in (1984, 1985)]
    expect = Hs[1983][d] / 400.0 - np.mean(others, axis=0) / 400.0
    assert np.allclose(rows[1983][d], expect)
    same = {y: np.full((A.SEASON_DAYS, len(A.LONS)), 400, dtype=np.int64) for y in (1983, 1984)}
    assert np.allclose(P.b0_rows(same, mo)[1983][10], 0.0)


def test_accumulate_reads_both_eastern_positions_and_skips_the_grid_edge():
    rows = [np.zeros(len(A.LONS)) for _ in range(A.SEASON_DAYS)]
    h = {"days": [10, 11, 12, 13], "lons": [60.0, 58.0, 56.0, 54.0]}
    rows[12][P.col(62.0)] = 2.0; rows[12][P.col(66.0)] = 0.5; rows[12][P.col(56.0)] = -4.0
    rows[13][P.col(60.0)] = 1.0; rows[13][P.col(66.0)] = 1.5; rows[13][P.col(54.0)] = -4.0
    rows[11][P.col(64.0)] = 99.0
    acc = P.new_acc(); P.accumulate(rows, h, acc)
    assert acc["samples"] == 2 and acc["carried"] == 3.0 and acc["pinned"] == 2.0 and acc["center"] == 8.0
    r = P.reader(acc)
    assert np.isclose(r["contrast"], 1.0 / 8.0) and np.isclose(r["carried"], 3.0 / 8.0)
    edge = {"days": [10, 11, 12, 13], "lons": [70.0, 68.0, 66.0, 64.0]}
    acc = P.new_acc(); P.accumulate(rows, edge, acc)
    assert acc["samples"] == 0 and acc["off_grid"] == 2
    acc = P.new_acc(); P.accumulate(rows, {"days": [10, 11, 12], "lons": [60.0, 58.0, 56.0]}, acc)
    assert acc["samples"] == 1
    with pytest.raises(SystemExit, match="no history-day"):
        P.reader(P.new_acc())


def test_measure_keeps_its_bands():
    rows = [np.full(len(A.LONS), -1.0) for _ in range(A.SEASON_DAYS)]
    h = {"days": [10, 11, 12, 13], "lons": [60.0, 58.0, 56.0, 54.0]}
    m = P.measure({"b": {1983: rows}, "c": {1983: rows}}, {(1983, "b"): [h], (1983, "c"): [h, h]}, ("b",))
    assert m["histories"] == 1 and m["samples"] == 2 and np.isclose(m["contrast"], 0.0)


def test_a_construction_keeps_the_following_histories_and_noise_keeps_all(monkeypatch):
    monkeypatch.setattr(P, "BANDS", ("b",))
    monkeypatch.setattr(L, "BAND_LAT", {"b": 17.25})
    monkeypatch.setattr(J, "matched_background", lambda std, p: [np.zeros((J.N_SAMPLES, len(J.LONS)))] * len(P.YEARS))
    centers = np.full(A.SEASON_DAYS, np.nan); centers[20:24] = [60.0, 58.0, 56.0, 54.0]
    monkeypatch.setattr(P, "build", lambda tracks, rng, kind, parts: (np.zeros((J.N_SAMPLES, len(J.LONS))), [centers]))
    follow = {"days": [20, 21, 22, 23], "lons": [60.0, 58.0, 56.0, 54.0]}
    other = {"days": [40, 41, 42, 43], "lons": [60.0, 58.0, 56.0, 54.0]}
    monkeypatch.setattr(H, "detected", lambda S, month_of: ([follow, other], None))
    def fake_b0(Hb, month_of):
        out = {}
        for y in Hb:
            rows = [np.zeros(len(A.LONS)) for _ in range(A.SEASON_DAYS)]
            for d in (22, 23, 42, 43):
                rows[d][P.col(54.0 if d in (23, 43) else 56.0)] = -2.0
            out[y] = rows
        return out
    monkeypatch.setattr(P, "b0_rows", fake_b0)
    m = P.construction("travel", "matched", 0, {}, {}, {"b": 1.0}, {}, A.month_index())
    assert m["histories"] == len(P.YEARS) and m["samples"] == 2 * len(P.YEARS)
    n = P.construction("none", "matched", 0, {}, {}, {"b": 1.0}, {}, A.month_index())
    assert n["histories"] == 2 * len(P.YEARS)


def cons_fixture():
    def pl(c):
        return {"samples": 50, "off_grid": 0, "carried": c + 0.1, "pinned": 0.1, "contrast": c, "histories": 40}
    base = {"travel": [pl(0.25), pl(0.4)], "pinned": [pl(-0.2), pl(-0.1)], "pinned_relax": [pl(-0.15), pl(-0.05)]}
    return {"archive": json.loads(json.dumps(base)), "matched": json.loads(json.dumps(base)), "noise": [pl(-0.05), pl(0.2)]}


def test_the_construction_rule_and_each_failure():
    assert P.apply_rule(cons_fixture()) == {"adequate": True, "failures": []}
    for bg, k, msg in (("archive", "pinned", "archive: pinned overlaps travel"), ("matched", "pinned_relax", "matched: pinned_relax overlaps travel")):
        c = cons_fixture(); c[bg][k][0]["contrast"] = 0.25
        r = P.apply_rule(c)
        assert not r["adequate"] and any(msg in f for f in r["failures"]), r
    c = cons_fixture(); c["noise"][0]["contrast"] = 0.9
    assert P.apply_rule(c)["adequate"]


def test_the_constructions_gate(tmp_path):
    res = cons_fixture(); p = tmp_path / "c.json"
    def write(results, verdict, sha=P.SCRIPT_SHA256_AT_LOAD):
        p.write_text(json.dumps({"generated_by": {"sha256": sha}, "results": results, "verdict": verdict}))
    rerun = lambda t, b, m, pa: res
    write(res, P.apply_rule(res))
    assert P.check_constructions(str(p), {}, {}, {}, {}, rerun=rerun)["results"] == res
    write(res, P.apply_rule(res), sha="0" * 64)
    with pytest.raises(SystemExit, match="different version"):
        P.check_constructions(str(p), {}, {}, {}, {}, rerun=rerun)
    write(res, P.apply_rule(res))
    other = cons_fixture(); other["noise"][1]["contrast"] = 0.21
    with pytest.raises(SystemExit, match="differs from a fresh run"):
        P.check_constructions(str(p), {}, {}, {}, {}, rerun=lambda t, b, m, pa: other)
    write(res, {"adequate": True, "failures": ["x"]})
    with pytest.raises(SystemExit, match="stored summary disagrees"):
        P.check_constructions(str(p), {}, {}, {}, {}, rerun=rerun)
    bad = cons_fixture(); bad["matched"]["pinned"][1]["contrast"] = 0.3
    write(bad, P.apply_rule(bad))
    with pytest.raises(SystemExit, match="do not pass the rule"):
        P.check_constructions(str(p), {}, {}, {}, {}, rerun=lambda t, b, m, pa: bad)


def test_position_and_readings():
    c = cons_fixture()
    p = P.position(0.3, c)
    assert p["inside"] == {"archive": ["travel"], "matched": ["travel"]} and p["noise_range"] == [-0.05, 0.2]
    assert P.position(-0.12, c)["inside"]["archive"] == ["pinned", "pinned_relax"] and P.position(0.0, c)["inside"]["archive"] == []
    res = {"measure": {"primary": {"histories": 86, "samples": 150, "carried": 0.4, "pinned": 0.1, "contrast": 0.3}}, "position": p}
    s = P.readings(res)[0]
    assert "86 westward histories in 15 to 24.5 N (150 history-days)" in s and "is 0.40 and 6 degrees east of the first longitude 0.10" in s
    assert "a contrast of 0.30, inside the archive-background range of the travel construction" in s and "noise alone gives -0.05 to 0.20" in s
    res["position"] = P.position(0.0, c)
    assert "inside no archive-background construction's range" in P.readings(res)[0]


def test_the_scale_is_the_magnitude_even_where_the_member_anomaly_is_positive():
    rows = [np.zeros(len(A.LONS)) for _ in range(A.SEASON_DAYS)]
    h = {"days": [10, 11, 12, 13], "lons": [60.0, 58.0, 56.0, 54.0]}
    rows[12][P.col(56.0)] = -3.0; rows[13][P.col(54.0)] = 1.0
    rows[12][P.col(62.0)] = 2.0
    acc = P.new_acc(); P.accumulate(rows, h, acc)
    assert acc["center"] == 4.0 and np.isclose(P.reader(acc)["carried"], 0.5)
