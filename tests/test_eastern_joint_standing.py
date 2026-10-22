"""Tests for the standing lobes read together (`EASTERN_JOINT_STANDING_2026-10-22.md`)."""

import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "scripts"))
import eastern_joint_standing as G  # noqa: E402

P, C, H, J, L, A = G.P, G.C, G.H, G.J, G.L, G.A


def profile():
    Q = np.zeros(len(H.DX)); Q[H.DX == 0] = -4.0; Q[H.DX == -1.0] = -2.0; Q[H.DX == 6.0] = 1.6; Q[H.DX == 9.0] = 3.0
    return Q.tolist()


def tracks():
    return [{"first_lon": 60.0, "speed": -2.0, "days": 5, "injected": {"measured": 6.0}},
            {"first_lon": 55.0, "speed": -3.0, "days": 4, "injected": {"measured": 4.0}}]


def test_the_layouts_cross_every_offset_width_strength_and_timing():
    lays = G.layouts()
    names = [G.layout_name(x) for x in lays]
    assert len(lays) == 1 + 7 * 3 * 2 * 3 and names[0] == "travel" and len(set(names)) == len(names)
    assert "o-4_w1_s1_steady" in names and "o8_w4_s1.5_long" in names and "o0_w2_s1_relax" in names
    assert set(G.NARROW_AT_START) <= set(names) and set(G.REPRODUCED) <= set(names)


def test_the_standing_rows_of_each_timing():
    assert G.standing_rows(20, 5, "steady") == [(18 * 4, 27 * 4, 1.0)]
    assert G.standing_rows(20, 5, "relax") == [(10 * 4, 20 * 4, 2.0), (20 * 4, 27 * 4, 1.0)]
    assert G.standing_rows(20, 5, "long") == [(5 * 4, 40 * 4, 1.0)]
    assert G.standing_rows(1, 5, "long")[0][0] == 0 and G.standing_rows(A.SEASON_DAYS - 5, 5, "long")[0][1] == J.N_SAMPLES
    with pytest.raises(SystemExit, match="unknown timing"):
        G.standing_rows(20, 5, "pulse")


def test_travel_and_the_pinned_lobes_are_the_lobe_tests_constructions(monkeypatch):
    monkeypatch.setitem(H.MEASURED, "profile", profile())
    parts = P.measured_parts()
    by = {G.layout_name(x): x for x in G.layouts()}
    for name, kind in G.REPRODUCED.items():
        V, cen = G.build(tracks(), np.random.default_rng(9), by[name], parts)
        V2, cen2 = P.build(tracks(), np.random.default_rng(9), kind, parts)
        assert np.array_equal(V, V2) and all(np.array_equal(a, b, equal_nan=True) for a, b in zip(cen, cen2)), name
    with pytest.raises(SystemExit, match="unknown layout"):
        G.build(tracks(), np.random.default_rng(9), {"kind": "drift"}, parts)


def test_a_broad_strong_long_lobe_stands_at_the_first_longitude(monkeypatch):
    monkeypatch.setitem(H.MEASURED, "profile", profile())
    parts = P.measured_parts()
    lay = {"kind": "standing", "offset": -2.0, "width": 4.0, "strength": 1.5, "timing": "long"}
    V, cen = G.build(tracks()[:1], np.random.default_rng(9), lay, parts)
    t0 = int(np.flatnonzero(np.isfinite(cen[0]))[0])
    assert t0 >= 16                                              # every index below lies inside the season
    vc = lambda x: int(np.flatnonzero(np.isclose(J.LONS, x))[0])  # the field's own grid
    col = vc(58.0)
    assert np.isclose(V[(t0 - 15) * 4, col], 1.5 * 0.4 * 6.0) and V[(t0 - 15) * 4 - 1, col] == 0.0
    assert np.isclose(V[(t0 + 5 + 15) * 4 - 1, col], 1.5 * 0.4 * 6.0) and V[(t0 + 5 + 15) * 4, col] == 0.0
    before = V[(t0 - 1) * 4]
    assert J.LONS[np.argmax(before)] == 58.0 and np.isclose(before[vc(62.0)], 1.5 * 0.4 * 6.0 * np.exp(-0.5))
    during = V[(t0 + 2) * 4]
    assert np.isclose(during.min(), -6.0 + 1.5 * 0.4 * 6.0 * np.exp(-0.5 * (2.0 / 4.0) ** 2))   # the core at 56 E, 2 degrees from the lobe


def b0_rows_fixture():
    rows = [np.zeros(len(A.LONS)) for _ in range(A.SEASON_DAYS)]
    rows[10][P.col(60.0)] = -4.0; rows[10][P.col(66.0)] = 1.0
    rows[11][P.col(58.0)] = -2.0; rows[11][P.col(64.0)] = 3.0
    rows[12][P.col(56.0)] = 2.0; rows[12][P.col(62.0)] = 1.0
    return rows


def test_the_profile_reads_east_of_each_days_member_and_skips_the_grid_edge():
    h = {"days": [10, 11, 12], "lons": [60.0, 58.0, 56.0]}
    p = G.profile({(1983, "b"): b0_rows_fixture()}, {(1983, "b"): [h]}, ("b",))
    assert np.isclose(p["by_day"][0], 0.25) and np.isclose(p["by_day"][1], 1.5) and np.isclose(p["by_day"][2], 0.5)
    assert p["by_day"][3] is None and p["days"] == [1, 1, 1, 0, 0] and p["off_grid"] == [0] * 5
    edge = {"days": [10, 11, 12], "lons": [71.0, 70.0, 60.0]}             # 69 E would read 75 E, which lies on the grid
    p = G.profile({(1983, "b"): b0_rows_fixture()}, {(1983, "b"): [h, edge]}, ("b",))
    assert p["off_grid"] == [1, 1, 0, 0, 0] and p["days"] == [1, 1, 2, 0, 0] and np.isclose(p["by_day"][0], 0.25)
    other = [np.full(len(A.LONS), 5.0) for _ in range(A.SEASON_DAYS)]
    p = G.profile({(1983, "b"): b0_rows_fixture(), (1983, "c"): other}, {(1983, "b"): [h], (1983, "c"): [h]}, ("b",))
    assert p["days"] == [1, 1, 1, 0, 0] and np.isclose(p["by_day"][0], 0.25)      # the band outside is left out
    with pytest.raises(SystemExit, match="no first day"):
        G.profile({(1983, "b"): b0_rows_fixture()}, {(1983, "b"): [edge]}, ("c",))


def test_measure_reads_the_three_readers_over_one_set_of_histories(monkeypatch):
    h = {"days": [10, 11, 12, 13], "lons": [60.0, 58.0, 56.0, 54.0]}
    seen = {}
    def fake_coh(rows_by, hist_by, bands):
        seen["coh"] = (rows_by, hist_by, bands)
        return {"histories": 1, "departure": -0.4}
    monkeypatch.setattr(C, "measure", fake_coh)
    b0 = b0_rows_fixture(); b0[13][P.col(54.0)] = -2.0; b0[13][P.col(60.0)] = 1.0; b0[13][P.col(66.0)] = 0.5
    m = G.measure({"b1": 1}, {(1983, "b"): b0, (1983, "c"): b0}, {(1983, "b"): [h], (1983, "c"): [h, h]}, ("b",))
    assert seen["coh"][0] == {"b1": 1} and seen["coh"][2] == ("b",) and m["departure"] == -0.4 and m["histories"] == 1
    assert m["samples"] == 2 and np.isclose(m["contrast"], (1.0 + 1.0 - 0.0 - 0.5) / 4.0) and np.isclose(m["first_day"], 0.25)
    assert m["first_day"] == m["profile"][0]


def test_a_construction_keeps_the_following_histories_and_noise_keeps_all(monkeypatch):
    monkeypatch.setattr(G, "BANDS", ("b",))
    monkeypatch.setattr(L, "BAND_LAT", {"b": 17.25})
    monkeypatch.setattr(J, "matched_background", lambda std, p: [np.zeros((J.N_SAMPLES, len(J.LONS)))] * len(G.YEARS))
    centers = np.full(A.SEASON_DAYS, np.nan); centers[20:24] = [60.0, 58.0, 56.0, 54.0]
    draws = []
    def fake_build(tracks, rng, lay, parts):
        draws.append(int(rng.integers(0, 10 ** 9)))
        return np.zeros((J.N_SAMPLES, len(J.LONS))), [centers]
    monkeypatch.setattr(G, "build", fake_build)
    follow = {"days": [20, 21, 22, 23], "lons": [60.0, 58.0, 56.0, 54.0]}
    other = {"days": [40, 41, 42, 43], "lons": [60.0, 58.0, 56.0, 54.0]}
    monkeypatch.setattr(H, "detected", lambda S, month_of: ([follow, other], "b1"))
    b1, b0, hist = G.fields({"kind": "travel"}, "matched", 0, {}, {}, {"b": 1.0}, {}, A.month_index())
    assert all(v == [follow] for v in hist.values()) and len(hist) == len(G.YEARS) and set(b1.values()) == {"b1"}
    assert draws == [int(np.random.default_rng([P.LOBE_SEED, 0, 0, i]).integers(0, 10 ** 9)) for i in range(len(G.YEARS))]
    b1, b0, hist = G.fields(None, "matched", 0, {}, {}, {"b": 1.0}, {}, A.month_index())
    assert all(v == [follow, other] for v in hist.values()) and len(b0) == len(G.YEARS)


def pl(dep, con, fd, n=40):
    return {"departure": dep, "contrast": con, "first_day": fd, "carried": con + 0.1, "pinned": 0.1, "samples": 50, "histories": n}


def cons_fixture():
    base = {"travel": [pl(-0.4, 0.3, 0.45), pl(-0.3, 0.4, 0.55)],
            "o0_w1_s1_steady": [pl(-0.3, 0.2, 0.0), pl(-0.2, 0.3, 0.1)],
            "o0_w2_s1_steady": [pl(-0.5, 0.3, 0.1), pl(-0.35, 0.35, 0.2)],
            "o0_w4_s1.5_steady": [pl(-0.45, 0.25, 0.4), pl(-0.3, 0.35, 0.6)]}
    return {"archive": json.loads(json.dumps(base)), "matched": json.loads(json.dumps(base)), "noise": [pl(-0.1, 0.0, 0.05), pl(0.1, 0.2, 0.15)]}


def test_the_construction_rule_and_each_failure():
    assert G.apply_rule(cons_fixture()) == {"adequate": True, "failures": []}
    for bg, k in (("archive", "o0_w1_s1_steady"), ("matched", "o0_w2_s1_steady")):
        c = cons_fixture(); c[bg][k][1]["first_day"] = 0.45
        r = G.apply_rule(c)
        assert not r["adequate"] and r["failures"] == [f"{bg}: {k} overlaps travel on the first day"], r
    c = cons_fixture(); c["archive"]["o0_w4_s1.5_steady"][0]["first_day"] = 0.9; c["noise"][0]["first_day"] = 0.9
    assert G.apply_rule(c)["adequate"]


def test_the_reproduction_gate():
    lobe = {bg: {"travel": cons_fixture()[bg]["travel"], "pinned": [pl(-0.1, -0.2, 0.3)], "pinned_relax": [pl(-0.1, -0.15, 0.3)]}
            for bg in ("archive", "matched")}
    lobe["noise"] = cons_fixture()["noise"]
    mine = cons_fixture()
    for bg in ("archive", "matched"):
        mine[bg]["o6_w2_s1_steady"] = [pl(0.5, -0.2, 0.9)]; mine[bg]["o6_w2_s1_relax"] = [pl(0.5, -0.15, 0.9)]
    G.check_reproduced(mine, lobe)
    bad = json.loads(json.dumps(mine)); bad["matched"]["o6_w2_s1_relax"][0]["contrast"] = -0.14
    with pytest.raises(SystemExit, match="matched o6_w2_s1_relax does not reproduce the lobe-position record's pinned_relax"):
        G.check_reproduced(bad, lobe)
    bad = json.loads(json.dumps(mine)); bad["noise"][1]["samples"] = 49
    with pytest.raises(SystemExit, match="noise alone does not reproduce"):
        G.check_reproduced(bad, lobe)


def test_the_constructions_gate(tmp_path):
    res = cons_fixture(); p = tmp_path / "c.json"
    def write(results, verdict, sha=G.SCRIPT_SHA256_AT_LOAD):
        p.write_text(json.dumps({"generated_by": {"sha256": sha}, "results": results, "verdict": verdict}))
    rerun = lambda t, b, m, pa: res
    write(res, G.apply_rule(res))
    assert G.check_constructions(str(p), {}, {}, {}, {}, rerun=rerun)["results"] == res
    write(res, G.apply_rule(res), sha="0" * 64)
    with pytest.raises(SystemExit, match="different version"):
        G.check_constructions(str(p), {}, {}, {}, {}, rerun=rerun)
    write(res, G.apply_rule(res))
    other = cons_fixture(); other["noise"][1]["first_day"] = 0.16
    with pytest.raises(SystemExit, match="differs from a fresh run"):
        G.check_constructions(str(p), {}, {}, {}, {}, rerun=lambda t, b, m, pa: other)
    write(res, {"adequate": True, "failures": ["x"]})
    with pytest.raises(SystemExit, match="stored summary disagrees"):
        G.check_constructions(str(p), {}, {}, {}, {}, rerun=rerun)
    bad = cons_fixture(); bad["matched"]["o0_w1_s1_steady"][1]["first_day"] = 0.5
    write(bad, G.apply_rule(bad))
    with pytest.raises(SystemExit, match="do not pass the rule"):
        G.check_constructions(str(p), {}, {}, {}, {}, rerun=lambda t, b, m, pa: bad)


def test_the_archive_must_equal_the_coherence_and_lobe_records():
    m = {"departure": -0.377, "contrast": 0.31, "carried": 0.47, "pinned": 0.16, "samples": 154}
    coh = {"results": {"measure": {"primary": {"departure": -0.377}}}}
    lobe = {"results": {"measure": {"primary": {"contrast": 0.31, "carried": 0.47, "pinned": 0.16, "samples": 154}}}}
    G.check_records(m, coh, lobe)
    with pytest.raises(SystemExit, match="departure differs"):
        G.check_records(dict(m, departure=-0.3770001), coh, lobe)
    with pytest.raises(SystemExit, match="contrast differs"):
        G.check_records(dict(m, samples=153), coh, lobe)


def test_classify_needs_all_three_readers_inside():
    c = cons_fixture()
    m = {"departure": -0.35, "contrast": 0.32, "first_day": 0.5, "histories": 86}
    cls = G.classify(m, c)
    assert cls["travel"] == {"departure": 0.0, "contrast": 0.0, "first_day": 0.0, "consistent": True}
    assert cls["o0_w2_s1_steady"]["consistent"] is False and np.isclose(cls["o0_w2_s1_steady"]["first_day"], 0.3)
    assert np.isclose(cls["o0_w1_s1_steady"]["departure"], -0.05) and np.isclose(cls["o0_w1_s1_steady"]["first_day"], 0.4)
    assert cls["o0_w4_s1.5_steady"]["consistent"] is True
    low = G.classify(dict(m, departure=-0.6), c)
    assert np.isclose(low["travel"]["departure"], -0.2) and low["travel"]["consistent"] is False
    edge = G.classify(dict(m, first_day=0.45), c)
    assert edge["travel"]["first_day"] == 0.0                     # a bound is inside


def test_summary_and_readings():
    c = cons_fixture()
    m = {"departure": -0.35, "contrast": 0.32, "first_day": 0.5, "histories": 86}
    s = G.summary(m, c)
    assert s["standing_layouts"] == 3 and s["consistent"]["archive"] == ["o0_w4_s1.5_steady"] and s["travel_consistent"]["archive"]
    assert s["excluded_by"]["archive"] == {"departure": 1, "contrast": 1, "first_day": 2} and s["noise_range"]["first_day"] == [0.05, 0.15]
    r = G.readings(m, s)
    assert "first day of the 86 westward histories in 15 to 24.5 N" in r[0] and "is 0.50 of the magnitude" in r[0] and "0.05 to 0.15" in r[0]
    assert r[1] == "travel contains the archive's three readings in its archive-background ranges"
    assert r[2] == "1 of the 3 standing layouts contain the archive's three readings in their archive-background ranges (o0_w4_s1.5_steady)"
    s = G.summary(dict(m, departure=-0.9), c)
    assert s["consistent"]["archive"] == [] and "does not contain" in G.readings(m, s)[1] and G.readings(m, s)[2].endswith("ranges")
