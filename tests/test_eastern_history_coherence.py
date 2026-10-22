"""Tests for the coherence of the westward histories (`EASTERN_HISTORY_COHERENCE_2026-10-22.md`)."""

import argparse
import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "scripts"))
import eastern_history_coherence as C  # noqa: E402

H, J, L, A = C.H, C.J, C.L, C.A


def test_the_plateau_is_flat_bottomed_between_its_ends():
    x = np.arange(20.0, 75.5, 0.5)
    P = C.plateau(60.0, 48.0, x)
    assert np.isclose(P.min(), -1.0) and np.array_equal(P, C.plateau(48.0, 60.0, x))
    inside = P[(x >= 52.0) & (x <= 56.0)]
    assert inside.max() < -0.95
    assert abs(P[x == 40.0][0]) < 0.01 and abs(P[x == 68.0][0]) < 0.01


def profile():
    P = np.zeros(len(H.DX)); P[H.DX == 0] = -2.0; P[H.DX == 6.0] = 1.0; P[H.DX == -1.0] = -1.0
    return P.tolist()


def test_build_without_a_plateau_is_the_shape_tests_measured_troughs(monkeypatch):
    monkeypatch.setitem(H.MEASURED, "profile", profile())
    tracks = [{"first_lon": 60.0, "speed": -2.0, "days": 5, "injected": {"measured": 6.0}},
              {"first_lon": 55.0, "speed": -3.0, "days": 4, "injected": {"measured": 4.0}}]
    V, cen = C.build(tracks, np.random.default_rng(5), 0.0)
    assert np.array_equal(V, H.packet_field(tracks, np.random.default_rng(5), "measured"))
    k = np.flatnonzero(np.isfinite(cen[0]))
    assert len(k) == 5 and np.allclose(np.diff(cen[0][k]), -2.0)
    assert np.isclose(cen[0][k[0]], 60.0 - 2.0 * 0.375)
    assert np.isnan(cen[1][: np.flatnonzero(np.isfinite(cen[1]))[0]]).all()


def test_build_adds_a_plateau_of_p_times_the_depth_with_two_day_margins(monkeypatch):
    monkeypatch.setitem(H.MEASURED, "profile", profile())
    tr = [{"first_lon": 60.0, "speed": -2.0, "days": 5, "injected": {"measured": 6.0}}]
    V0, _ = C.build(tr, np.random.default_rng(5), 0.0)
    V2, _ = C.build(tr, np.random.default_rng(5), 2.0)
    D = V2 - V0
    rows = np.flatnonzero(np.abs(D).max(axis=1) > 0)
    t0 = int(np.flatnonzero(np.abs(V0).max(axis=1) > 0)[0] / 4)
    assert rows[0] == (t0 - 2) * 4 and rows[-1] == (t0 + 5 + 2) * 4 - 1
    assert np.isclose(D[rows[0]].min(), -12.0)
    assert np.allclose(D[rows], D[rows[0]])                                    # standing, not moving
    assert np.isclose(D[rows[0]][np.isclose(J.LONS, 56.0)][0], -12.0, atol=0.6)


def rows_fixture():
    return [np.arange(len(A.LONS), dtype=float) * 0.0 + (-1.0 - 0.1 * d) for d in range(A.SEASON_DAYS)]


def test_ends_reads_the_first_and_last_longitudes_and_refuses_the_season_edges():
    rows = [np.full(len(A.LONS), 0.0) for _ in range(A.SEASON_DAYS)]
    j0, j1 = C.col(60.0), C.col(50.0)
    for k in range(7):
        rows[20 + k][j0] = -4.0 + k; rows[30 - k][j1] = -3.0 + 0.5 * k
    h = {"days": [20, 25, 30], "lons": [60.0, 55.0, 50.0]}
    e = C.ends(rows, h)
    assert e["departure"] == [-4.0 + k for k in range(7)] and e["arrival"] == [-3.0 + 0.5 * k for k in range(7)]
    assert C.ends(rows, {"days": [116, 118, 120], "lons": [60.0, 55.0, 50.0]}) is None
    assert C.ends(rows, {"days": [3, 4, 5], "lons": [60.0, 55.0, 50.0]}) is None
    assert C.ends(rows, {"days": [0, 5, 6], "lons": [60.0, 55.0, 50.0]}) is not None
    assert A.SEASON_DAYS == 122
    assert C.ends(rows, {"days": [115, 117, 119], "lons": [60.0, 55.0, 50.0]}) is not None
    assert C.ends(rows, {"days": [6, 10, 12], "lons": [60.0, 55.0, 50.0]}) is not None


def test_pooled_is_a_ratio_of_sums_over_two_to_four_days():
    s = [{"departure": [-4.0, -3.0, -2.0, 1.0, 0.0, 9.0, 9.0], "arrival": [-2.0, -1.0, -1.0, -1.0, -1.0, 9.0, 9.0]},
         {"departure": [-2.0, 0.0, 2.0, 2.0, 2.0, 9.0, 9.0], "arrival": [-2.0, 0.0, 0.0, 0.0, 0.0, 9.0, 9.0]}]
    p = C.pooled(s)
    assert p["histories"] == 2
    assert np.isclose(p["departure"], ((-2.0 + 1.0 + 0.0) / 3 + (2.0 + 2.0 + 2.0) / 3) / -6.0)
    assert np.isclose(p["arrival"], ((-1.0 - 1.0 - 1.0) / 3 + 0.0) / -4.0)
    assert np.allclose(p["departure_curve"][:3], [1.0, 0.5, 0.0])
    with pytest.raises(SystemExit, match="not negative"):
        C.pooled([{"departure": [1.0] * 7, "arrival": [-1.0] * 7}])
    assert C.pooled([]) == {"histories": 0}


def test_measure_keeps_its_bands_and_skips_histories_without_room():
    rows = [np.full(len(A.LONS), -1.0) for _ in range(A.SEASON_DAYS)]
    h = {"days": [20, 22, 24, 26], "lons": [60.0, 57.0, 54.0, 51.0]}
    edge = {"days": [118, 119, 120, 121], "lons": [60.0, 57.0, 54.0, 51.0]}
    m = C.measure({(1983, "b"): rows, (1983, "c"): rows}, {(1983, "b"): [h, edge], (1983, "c"): [h]}, ("b",))
    assert m["histories"] == 1 and np.isclose(m["departure"], 1.0)


def test_a_construction_measures_only_the_histories_that_follow_an_injected_trough(monkeypatch):
    monkeypatch.setattr(C, "BANDS", ("b",))
    monkeypatch.setattr(L, "BAND_LAT", {"b": 17.25})
    monkeypatch.setattr(J, "matched_background", lambda std, p: [np.zeros((J.N_SAMPLES, len(J.LONS)))] * len(C.YEARS))
    centers = np.full(A.SEASON_DAYS, np.nan); centers[20:24] = [60.0, 58.0, 56.0, 54.0]
    monkeypatch.setattr(C, "build", lambda tracks, rng, p: (np.zeros((J.N_SAMPLES, len(J.LONS))), [centers]))
    follow = {"days": [20, 21, 22, 23], "lons": [60.0, 58.0, 56.0, 54.0]}
    other = {"days": [40, 41, 42, 43], "lons": [70.0, 67.0, 64.0, 61.0]}
    rows = [np.full(len(A.LONS), -2.0) for _ in range(A.SEASON_DAYS)]
    monkeypatch.setattr(H, "detected", lambda S, month_of: ([follow, other], rows))
    m = C.construction(1.0, "matched", 0, {}, {}, {"b": 1.0}, A.month_index())
    assert m["histories"] == len(C.YEARS)


def cons_fixture():
    def pl(dep, arr=0.0, n=30):
        return {"histories": n, "departure": dep, "arrival": arr}
    base = {"0": [pl(-0.3), pl(-0.2)], "0.5": [pl(0.0), pl(0.05)], "1": [pl(0.1), pl(0.2)], "2": [pl(0.35), pl(0.5)]}
    return {"archive": json.loads(json.dumps(base)), "matched": json.loads(json.dumps(base)), "noise": [pl(-0.1), pl(0.1)]}


def test_the_construction_rule_and_each_failure():
    assert C.apply_rule(cons_fixture()) == {"adequate": True, "failures": []}
    for bg, p, v, msg in (("archive", "0", 0.1, "archive: plateau 0 and plateau 1 overlap"), ("matched", "2", 0.2, "matched: plateau 1 and plateau 2 overlap")):
        c = cons_fixture(); c[bg][p][0]["departure"] = v
        r = C.apply_rule(c)
        assert not r["adequate"] and any(msg in f for f in r["failures"]), r
    c = cons_fixture(); c["archive"]["0.5"][0]["departure"] = 0.9
    assert C.apply_rule(c)["adequate"]                         # the half-depth plateau is not part of the rule


def test_the_constructions_gate(tmp_path):
    res = cons_fixture(); p = tmp_path / "c.json"
    def write(results, verdict, sha=C.SCRIPT_SHA256_AT_LOAD):
        p.write_text(json.dumps({"generated_by": {"sha256": sha}, "results": results, "verdict": verdict}))
    rerun = lambda t, b, m: res
    write(res, C.apply_rule(res))
    assert C.check_constructions(str(p), {}, {}, {}, rerun=rerun)["results"] == res
    write(res, C.apply_rule(res), sha="0" * 64)
    with pytest.raises(SystemExit, match="different version"):
        C.check_constructions(str(p), {}, {}, {}, rerun=rerun)
    write(res, C.apply_rule(res))
    other = cons_fixture(); other["noise"][0]["departure"] = -0.11
    with pytest.raises(SystemExit, match="differs from a fresh run"):
        C.check_constructions(str(p), {}, {}, {}, rerun=lambda t, b, m: other)
    write(res, {"adequate": True, "failures": ["x"]})
    with pytest.raises(SystemExit, match="stored summary disagrees"):
        C.check_constructions(str(p), {}, {}, {}, rerun=rerun)
    bad = cons_fixture(); bad["archive"]["1"][0]["departure"] = 0.6
    write(bad, C.apply_rule(bad))
    with pytest.raises(SystemExit, match="do not pass the rule"):
        C.check_constructions(str(p), {}, {}, {}, rerun=lambda t, b, m: bad)


def test_standing_depth_interpolates_between_construction_means():
    c = cons_fixture()          # means -0.25, 0.025, 0.15, 0.425
    s = C.standing_depth(0.15, c)
    assert np.isclose(s["estimate"], 1.0) and s["inside"] == ["1"]
    s = C.standing_depth(-0.1125, c)
    assert np.isclose(s["estimate"], 0.25) and s["inside"] == []
    assert C.standing_depth(-0.25, c)["inside"] == ["0"]
    s = C.standing_depth(0.2875, c)
    assert np.isclose(s["estimate"], 1.5) and s["inside"] == []
    assert C.standing_depth(-0.9, c)["estimate"] is None and C.standing_depth(0.9, c)["estimate"] is None
    s = C.standing_depth(0.0, c, key="arrival")
    assert s["estimate"] is None and s["inside"] == ["0", "0.5", "1", "2"]


def test_the_readings_state_the_estimate_and_the_noise_range():
    res = {"measure": {"primary": {"histories": 80, "departure": -0.2, "arrival": 0.05}},
           "standing": {"departure": {"estimate": 0.12, "inside": ["0"]}}, "noise_range": [-0.3, 0.14]}
    s = C.readings(res)
    assert "80 westward histories in 15 to 24.5 N" in s[0] and "-0.20 of its value on the first day" in s[0]
    assert "a standing plateau of 0.12 times the traveling trough's depth" in s[0] and "inside the placement range of plateau depth 0" in s[0]
    assert "noise alone gives -0.30 to 0.14" in s[0] and "0.05 of its value on the last day" in s[1]
    res["standing"]["departure"] = {"estimate": None, "inside": []}
    assert "outside the span of the archive-background constructions" in C.readings(res)[0]


def test_load_inputs_refuses_records_made_by_other_scripts(tmp_path):
    shape = tmp_path / "s.json"; inject = tmp_path / "i.json"
    shape.write_text(json.dumps({"generated_by": {"sha256": "0" * 64}, "results": {}}))
    inject.write_text(json.dumps({"generated_by": {"sha256": J.SCRIPT_SHA256_AT_LOAD}, "results": {}}))
    a = argparse.Namespace(shape=str(shape), inject=str(inject), records="x", histories="x", xband="x")
    with pytest.raises(SystemExit, match="shape record was made by a different version"):
        C.load_inputs(a)
    shape.write_text(json.dumps({"generated_by": {"sha256": H.SCRIPT_SHA256_AT_LOAD}, "results": {}}))
    inject.write_text(json.dumps({"generated_by": {"sha256": "0" * 64}, "results": {}}))
    with pytest.raises(SystemExit, match="injection-recovery record was made by a different version"):
        C.load_inputs(a)


def test_standing_depth_orders_the_depths_as_numbers():
    def pl(dep):
        return {"histories": 30, "departure": dep, "arrival": 0.0}
    c = {"archive": {"0": [pl(-0.3)], "2": [pl(0.4)], "10": [pl(0.9)]}}
    s = C.standing_depth(0.65, c)
    assert np.isclose(s["estimate"], 6.0) and list(s["means"]) == ["0", "2", "10"]
