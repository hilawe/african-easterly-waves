"""Tests for the pre-start reading (`EASTERN_PRESTART_2026-10-22.md`)."""

import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "scripts"))
import eastern_prestart as S  # noqa: E402

G, P, C, H, J, L, A = S.G, S.P, S.C, S.H, S.J, S.L, S.A
TRAVELERS = ["travel", "travel_whole_quarter", "travel_whole_zero", "travel_lobe_zero"]


def profile():
    Q = np.zeros(len(H.DX)); Q[H.DX == 0] = -4.0; Q[H.DX == -1.0] = -2.0; Q[H.DX == 6.0] = 1.6
    return Q.tolist()


def tracks():
    return [{"first_lon": 60.0, "speed": -2.0, "days": 5, "injected": {"measured": 6.0}}]


def test_the_cases_are_four_travelers_and_every_standing_lobe_near_the_start():
    names = [S.case_name(c) for c in S.cases()]
    assert names[:4] == TRAVELERS and len(names) == 4 + 2 * 3 * 3 * 3 and len(set(names)) == len(names)
    assert set(S.DECIDING) <= set(names) and "o0_w2_s0.5_long" in names and "o-2_w1_s1.5_relax" in names
    assert not any(n.startswith(("o2_", "o-4_")) for n in names)


def test_the_growth_factors():
    assert [S.growth_factor("constant", a) for a in (0, 1, 3)] == [1.0, 1.0, 1.0]
    assert [S.growth_factor("whole_quarter", a) for a in (0, 1, 2, 3)] == [0.25, 0.625, 1.0, 1.0]
    assert [S.growth_factor(g, a) for g in ("whole_zero", "lobe_zero") for a in (0, 1, 2, 3)] == [0.0, 0.5, 1.0, 1.0] * 2
    with pytest.raises(SystemExit, match="unknown growth"):
        S.growth_factor("pulse", 1)


def test_constant_travel_and_standing_lobes_are_the_joint_tests_constructions(monkeypatch):
    monkeypatch.setitem(H.MEASURED, "profile", profile())
    parts = P.measured_parts()
    V, cen = S.build(tracks(), np.random.default_rng(9), {"kind": "travel", "growth": "constant"}, parts)
    V2, cen2 = G.build(tracks(), np.random.default_rng(9), {"kind": "travel"}, parts)
    assert np.array_equal(V, V2) and np.array_equal(cen[0], cen2[0], equal_nan=True)
    lay = {"kind": "standing", "offset": -2.0, "width": 4.0, "strength": 0.5, "timing": "relax"}
    V, _ = S.build(tracks(), np.random.default_rng(9), lay, parts)
    V2, _ = G.build(tracks(), np.random.default_rng(9), lay, parts)
    assert np.array_equal(V, V2)


def test_growing_travelers_scale_the_whole_trough_or_the_part_above_the_core(monkeypatch):
    monkeypatch.setitem(H.MEASURED, "profile", profile())
    parts = P.measured_parts()
    full, _ = S.build(tracks(), np.random.default_rng(9), {"kind": "travel", "growth": "constant"}, parts)
    k0 = int(np.flatnonzero(np.abs(full).sum(axis=1) > 0)[0])
    for growth, at0 in (("whole_quarter", 0.25), ("whole_zero", 0.0)):
        V, _ = S.build(tracks(), np.random.default_rng(9), {"kind": "travel", "growth": growth}, parts)
        assert np.allclose(V[k0], at0 * full[k0]) and np.allclose(V[k0 + 4], (at0 + (1 - at0) * 0.5) * full[k0 + 4])
        assert np.allclose(V[k0 + 8], full[k0 + 8]) and np.allclose(V[k0 + 12], full[k0 + 12])
    V, _ = S.build(tracks(), np.random.default_rng(9), {"kind": "travel", "growth": "lobe_zero"}, parts)
    core = P.core_shape(J.LONS, 60.0, 6.0, parts["core"])
    assert np.allclose(V[k0], core) and V[k0].max() == 0.0 and np.allclose(V[k0 + 8], full[k0 + 8])
    assert np.isclose(V[k0 + 4].max(), 0.5 * full[k0 + 4].max())


def rows_fixture():
    rows = [np.zeros(len(A.LONS)) for _ in range(A.SEASON_DAYS)]
    j = P.col(60.0)
    rows[7][j], rows[8][j], rows[9][j], rows[10][j] = 0.5, 1.0, 3.0, -4.0
    return rows


def test_the_pre_start_reader_averages_two_days_before_over_the_first_days_magnitude():
    h = {"days": [10, 11, 12], "lons": [60.0, 58.0, 56.0]}
    r = S.prestart({(1983, "b"): rows_fixture()}, {(1983, "b"): [h]}, ("b",))
    assert np.isclose(r["prestart"], 2.0 / 4.0) and r["prestart_numerator"] == 2.0 and r["prestart_denominator"] == 4.0
    assert r["prestart_histories"] == 1 and r["season_edge"] == 0 and np.allclose(r["prestart_profile"], [0.125, 0.25, 0.75, -1.0])
    early = {"days": [2, 3], "lons": [60.0, 58.0]}
    other = [np.full(len(A.LONS), 9.0) for _ in range(A.SEASON_DAYS)]
    r = S.prestart({(1983, "b"): rows_fixture(), (1983, "c"): other}, {(1983, "b"): [h, early], (1983, "c"): [h]}, ("b",))
    assert r["season_edge"] == 1 and r["prestart_histories"] == 1 and np.isclose(r["prestart"], 0.5)
    with pytest.raises(SystemExit, match="no history enters"):
        S.prestart({(1983, "b"): rows_fixture()}, {(1983, "b"): [early]}, ("b",))
    rows = rows_fixture(); rows[20][P.col(50.0)] = 1.0                      # a start whose anomaly is positive
    late = {"days": [20, 21], "lons": [50.0, 48.0]}
    r = S.prestart({(1983, "b"): rows}, {(1983, "b"): [h, late]}, ("b",))
    assert r["prestart_denominator"] == 5.0 and np.isclose(r["prestart"], 2.0 / 5.0)


def test_measure_adds_the_pre_start_reader_to_the_joint_readers(monkeypatch):
    monkeypatch.setattr(G, "measure", lambda b1, b0, hs, bands: {"departure": -0.3, "first_day": 0.1})
    h = {"days": [10, 11, 12], "lons": [60.0, 58.0, 56.0]}
    m = S.measure({}, {(1983, "b"): rows_fixture()}, {(1983, "b"): [h]}, ("b",))
    assert m["departure"] == -0.3 and m["first_day"] == 0.1 and np.isclose(m["prestart"], 0.5)


def test_fields_keep_the_following_histories_with_the_lobe_tests_starts(monkeypatch):
    monkeypatch.setattr(S, "BANDS", ("b",))
    monkeypatch.setattr(L, "BAND_LAT", {"b": 17.25})
    monkeypatch.setattr(J, "matched_background", lambda std, p: [np.zeros((J.N_SAMPLES, len(J.LONS)))] * len(S.YEARS))
    centers = np.full(A.SEASON_DAYS, np.nan); centers[20:24] = [60.0, 58.0, 56.0, 54.0]
    draws = []
    def fake_build(trs, rng, case, parts):
        draws.append(int(rng.integers(0, 10 ** 9)))
        return np.zeros((J.N_SAMPLES, len(J.LONS))), [centers]
    monkeypatch.setattr(S, "build", fake_build)
    follow = {"days": [20, 21, 22, 23], "lons": [60.0, 58.0, 56.0, 54.0]}
    other = {"days": [40, 41, 42, 43], "lons": [60.0, 58.0, 56.0, 54.0]}
    monkeypatch.setattr(H, "detected", lambda S_, month_of: ([follow, other], "b1"))
    b1, b0, hist = S.fields({"kind": "travel", "growth": "constant"}, "matched", 13, {}, {}, {"b": 1.0}, {}, A.month_index())
    assert all(v == [follow] for v in hist.values()) and set(b1.values()) == {"b1"} and len(b0) == len(S.YEARS)
    assert draws == [int(np.random.default_rng([P.LOBE_SEED, 0, 13, i]).integers(0, 10 ** 9)) for i in range(len(S.YEARS))]
    _, _, hist = S.fields(None, "matched", 0, {}, {}, {"b": 1.0}, {}, A.month_index())
    assert all(v == [follow, other] for v in hist.values())


def pl(pre, fd=0.2, dep=-0.3, con=0.3):
    return {"prestart": pre, "first_day": fd, "departure": dep, "contrast": con, "prestart_histories": 30, "histories": 30}


def cons_fixture():
    def series(a_lo, a_hi, b_lo, b_hi, **kw):
        return [pl(a_lo, **kw)] + [pl((a_lo + a_hi) / 2, **kw)] * 8 + [pl(a_hi, **kw)] + [pl(b_lo, **kw)] + [pl((b_lo + b_hi) / 2, **kw)] * 8 + [pl(b_hi, **kw)]
    base = {t: series(0.0, 0.2, 0.05, 0.25) for t in TRAVELERS}
    base.update({k: series(0.4, 1.0, 0.45, 1.2) for k in S.DECIDING})
    base["o0_w1_s0.5_steady"] = series(0.1, 0.3, 0.1, 0.35)
    return {"archive": json.loads(json.dumps(base)), "matched": json.loads(json.dumps(base)), "noise": series(-0.1, 0.3, -0.2, 0.25)}


def test_the_rule_needs_every_traveler_below_each_deciding_lobe_in_both_blocks():
    assert S.apply_rule(cons_fixture()) == {"adequate": True, "failures": []}
    c = cons_fixture(); c["matched"]["travel_lobe_zero"][15]["prestart"] = 0.5
    r = S.apply_rule(c)
    assert not r["adequate"] and r["failures"] == [f"matched block B: {k} overlaps the travelers" for k in S.DECIDING]
    c = cons_fixture(); c["archive"]["o0_w2_s1_steady"][3]["prestart"] = 0.2
    assert S.apply_rule(c)["failures"] == ["archive block A: o0_w2_s1_steady overlaps the travelers"]
    c = cons_fixture(); c["archive"]["o0_w1_s0.5_steady"][3]["prestart"] = -5.0; c["noise"][0]["prestart"] = 9.0
    assert S.apply_rule(c)["adequate"]


def test_the_reproduction_gate():
    c = cons_fixture()
    joint = {bg: {"travel": [{k: v for k, v in x.items() if k != "prestart"} for x in c[bg]["travel"][:5]],
                  "o0_w2_s1_steady": [{k: v for k, v in x.items() if k != "prestart"} for x in c[bg]["o0_w2_s1_steady"][:5]]} for bg in ("archive", "matched")}
    joint["noise"] = [{k: v for k, v in x.items() if k != "prestart"} for x in c["noise"][:5]]
    S.check_reproduced(c, joint)
    bad = json.loads(json.dumps(c)); bad["archive"]["o0_w2_s1_steady"][4]["first_day"] = 0.21
    with pytest.raises(SystemExit, match="archive o0_w2_s1_steady placement 4 does not reproduce"):
        S.check_reproduced(bad, joint)
    ok = json.loads(json.dumps(c)); ok["archive"]["o0_w2_s1_steady"][5]["first_day"] = 0.21; ok["archive"]["o0_w1_s0.5_steady"][0]["first_day"] = 7.0
    S.check_reproduced(ok, joint)
    bad = json.loads(json.dumps(c)); bad["matched"]["travel"][0]["contrast"] = 0.29
    with pytest.raises(SystemExit, match="matched travel placement 0 does not reproduce"):
        S.check_reproduced(bad, joint)
    bad = json.loads(json.dumps(c)); bad["noise"][2]["departure"] = -0.31
    with pytest.raises(SystemExit, match="noise alone placement 2"):
        S.check_reproduced(bad, joint)


def test_the_constructions_gate(tmp_path):
    res = cons_fixture(); p = tmp_path / "c.json"
    def write(results, verdict, sha=S.SCRIPT_SHA256_AT_LOAD):
        p.write_text(json.dumps({"generated_by": {"sha256": sha}, "results": results, "verdict": verdict}))
    rerun = lambda t, b, m, pa: res
    write(res, S.apply_rule(res))
    assert S.check_constructions(str(p), {}, {}, {}, {}, rerun=rerun)["results"] == res
    write(res, S.apply_rule(res), sha="0" * 64)
    with pytest.raises(SystemExit, match="different version"):
        S.check_constructions(str(p), {}, {}, {}, {}, rerun=rerun)
    write(res, S.apply_rule(res))
    other = cons_fixture(); other["noise"][19]["prestart"] = 0.26
    with pytest.raises(SystemExit, match="differs from a fresh run"):
        S.check_constructions(str(p), {}, {}, {}, {}, rerun=lambda t, b, m, pa: other)
    write(res, {"adequate": True, "failures": ["x"]})
    with pytest.raises(SystemExit, match="stored summary disagrees"):
        S.check_constructions(str(p), {}, {}, {}, {}, rerun=rerun)
    bad = cons_fixture(); bad["archive"]["o-2_w4_s1_relax"][12]["prestart"] = 0.1
    write(bad, S.apply_rule(bad))
    with pytest.raises(SystemExit, match="do not pass the rule"):
        S.check_constructions(str(p), {}, {}, {}, {}, rerun=lambda t, b, m, pa: bad)


def test_the_archive_must_equal_the_joint_record():
    m = {"departure": -0.377, "contrast": 0.31, "first_day": 0.13, "samples": 154, "histories": 86}
    rec = {"results": {"measure": dict(m)}}
    S.check_joint(m, rec)
    for k in m:
        with pytest.raises(SystemExit, match=f"archive {k} differs"):
            S.check_joint(dict(m, **{k: m[k] + 1}), rec)


def test_verdicts_need_both_blocks():
    c = cons_fixture()
    m = {"prestart": 0.1, "first_day": 0.2, "departure": -0.3, "contrast": 0.3}
    cls = S.classify(m, c)
    assert cls["travel"]["verdict"] == "consistent" and cls["o0_w2_s1_steady"]["verdict"] == "excluded"
    assert np.isclose(cls["o0_w2_s1_steady"]["A"]["prestart"], -0.3) and np.isclose(cls["o0_w2_s1_steady"]["B"]["prestart"], -0.35)
    split = S.classify(dict(m, prestart=0.02), c)                 # inside A (0.0 to 0.2), below B (0.05 to 0.25)
    assert split["travel"]["verdict"] == "split" and np.isclose(split["travel"]["B"]["prestart"], -0.03)
    edge = S.classify(dict(m, prestart=0.25), c)
    assert edge["travel"]["verdict"] == "split" and edge["travel"]["B"]["prestart"] == 0.0     # a bound is inside
    far = S.classify(dict(m, first_day=0.5), c)
    assert far["travel"]["verdict"] == "excluded"                 # another reader outside in both blocks


def test_the_three_decisions_and_the_readings():
    c = cons_fixture()
    m = {"prestart": 0.1, "first_day": 0.2, "departure": -0.3, "contrast": 0.3, "prestart_histories": 80}
    s = S.summary(m, c)
    assert s["decision"] == "standing excluded" and s["standing_consistent"] == ["o0_w1_s0.5_steady"] and s["noise_contains"] == {"A": True, "B": True}
    r = S.readings(m, s)
    assert "80 westward histories in 15 to 24.5 N" in r[0] and "is 0.10 of the magnitude" in r[0] and "-0.10 to 0.30 in block A and -0.20 to 0.25 in block B" in r[0]
    assert r[2] == "the readings exclude the four standing lobes that the joint test left, and at least one traveler stays consistent"
    assert r[3] == "1 of the 5 standing lobes are consistent in both blocks (o0_w1_s0.5_steady)"
    s = S.summary(dict(m, prestart=0.7), c)
    assert s["decision"] == "travelers excluded" and s["noise_contains"] == {"A": False, "B": False}
    assert S.readings(m, s)[2].startswith("the readings exclude every traveler")
    s = S.summary(dict(m, prestart=0.27), c)
    assert s["decision"] == "unresolved" and S.readings(m, s)[2] == "the readings do not decide between the travelers and the joint test's standing lobes"
    mixed = json.loads(json.dumps(c))
    for i in (0, 10):
        mixed["archive"]["o0_w2_s1_steady"][i]["prestart"] = 0.05                 # one deciding lobe now holds 0.1 in both blocks
    assert S.summary(m, mixed)["decision"] == "unresolved"
    mixed = json.loads(json.dumps(c))
    for i in (9, 19):
        mixed["archive"]["travel"][i]["prestart"] = 0.7                          # one traveler now holds 0.7 in both blocks
    s = S.summary(dict(m, prestart=0.7), mixed)
    assert s["travelers"]["travel"] == "consistent" and s["travelers"]["travel_whole_zero"] == "excluded" and s["decision"] == "unresolved"
