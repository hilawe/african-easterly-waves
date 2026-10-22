"""Tests for the shape of the westward histories (`EASTERN_HISTORY_SHAPE_2026-10-21.md`)."""

import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "scripts"))
import eastern_history_shape as H  # noqa: E402

J = H.J
L = H.L
A = H.A


def test_the_shapes_have_their_declared_forms():
    x = np.arange(30.0, 70.0, 0.5)
    assert np.array_equal(H.shape("gauss", x, 50.0, 6.0), A.gauss(x, 50.0, 6.0, 2.0))
    assert np.array_equal(H.shape("none", x, 50.0, 6.0), np.zeros_like(x))
    for k in (1.0, 2.25):
        v = H.shape(f"m{k:g}", x, 50.0, 6.0)
        assert np.isclose(v[x == 50.0][0], -6.0)
        assert np.isclose(v[x == 48.0][0], 0.0) and np.isclose(v[x == 52.0][0], 0.0)
        crest = H.shape(f"m{k:g}", np.array([50.0 + 2.0 * np.sqrt(3.0)]), 50.0, 6.0)[0]
        assert np.isclose(crest, k * 6.0 * 2.0 * np.exp(-1.5))
        inside = H.shape(f"m{k:g}", np.array([51.0]), 50.0, 6.0)[0]
        assert np.isclose(inside, -6.0 * 0.75 * np.exp(-0.125))
    d = H.shape("dipole", x, 50.0, 6.0)
    assert np.isclose(d.min(), -6.0) and x[np.argmin(d)] == 50.0
    assert np.isclose(d.max(), 6.0) and x[np.argmax(d)] == 54.0
    with pytest.raises(SystemExit, match="unknown shape"):
        H.shape("square", x, 50.0, 6.0)


def test_the_gaussian_calibration_is_the_injection_tests_and_every_shape_has_one():
    for tr in (J.REFERENCE_TRACK, {"first_lon": 32.5, "speed": -3.0, "days": 4}):
        assert H.calibration_ratio(tr, "gauss") == J.calibration_ratio(tr, 2.0)
    for k in H.SHAPES:
        r = H.calibration_ratio(J.REFERENCE_TRACK, k)
        assert r is not None and 0.0 < r < 1.0
    assert H.calibration_ratio({"first_lon": 53.0, "speed": -7.0 / 6.0, "days": 10}, "gauss") is not None


def test_shape_tracks_refuses_a_track_without_a_calibration(monkeypatch):
    h = {"lons": [58.0, 55.5, 53.0, 50.5, 48.0, 45.5], "depths": [-4.0] * 6, "days": list(range(6))}
    tr = H.shape_tracks({(1983, "b"): [h]})["b"][1983][0]
    assert set(tr["injected"]) == set(H.SHAPES)
    assert np.isclose(tr["injected"]["m2.25"], 4.0 / tr["ratio"]["m2.25"])
    real = H.calibration_ratio
    monkeypatch.setattr(H, "calibration_ratio", lambda track, kind, month_of=None: None if kind == "dipole" else real(track, kind, month_of))
    with pytest.raises(SystemExit, match="no calibration for the shape dipole"):
        H.shape_tracks({(1983, "b"): [h]})


def test_the_gaussian_troughs_are_the_injection_tests_p1_and_none_draws_the_same_starts():
    tracks = [{"first_lon": 60.0, "speed": -2.0, "days": 5, "injected": {"2": 6.0, "4": 7.0, "gauss": 6.0}},
              {"first_lon": 55.0, "speed": -3.0, "days": 4, "injected": {"2": 5.0, "4": 6.0, "gauss": 5.0}}]
    a = H.packet_field(tracks, np.random.default_rng(7), "gauss")
    b = J.packet_field(tracks, np.random.default_rng(7), 1, 1, 2.0)
    assert np.array_equal(a, b)
    for t, d in zip(tracks, (3.0, 2.5)):
        t["injected"]["m2.25"] = d
    m = H.packet_field(tracks, np.random.default_rng(7), "m2.25")
    assert np.isclose(m.min(), -3.0, atol=0.01) and m.max() > 0.0
    r1, r2 = np.random.default_rng(7), np.random.default_rng(7)
    z = H.packet_field(tracks, r1, "none")
    H.packet_field(tracks, r2, "gauss")
    assert np.array_equal(z, np.zeros_like(z)) and r1.integers(0, 10**9) == r2.integers(0, 10**9)


def test_b1_rows_are_the_own_month_anomaly():
    S = L.construction_samples("R1", 0)
    Hs, rows = H.b1_rows(S, A.month_index())
    assert len(rows) == A.SEASON_DAYS and rows[0].shape == (H.NA,)
    mo = A.month_index()
    for m in L.MONTHS:
        assert np.allclose(np.sum([rows[i] for i in range(len(rows)) if mo[i] == m], axis=0), 0.0)
    c20 = int(np.flatnonzero(np.isclose(J.LONS, 20.0))[0])
    assert Hs[5][0] == round(S[20:24, c20].sum() * 100)
    daily = Hs / 400.0
    i = 40
    assert np.allclose(rows[i], daily[i] - daily[mo == mo[i]].mean(axis=0))


def test_the_composite_sums_member_days_at_each_relative_longitude():
    rows = [np.arange(H.NA, dtype=float) + 1000.0 * d for d in range(A.SEASON_DAYS)]
    h = {"days": [3, 4], "lons": [50.0, 21.0]}
    c = H.composite({(1983, "b"): rows, (1983, "c"): rows}, {(1983, "b"): [h], (1983, "c"): [h]}, ("b",))
    s, n = np.asarray(c["sum"]), np.asarray(c["count"])
    assert c["histories"] == 1
    j50 = int(np.flatnonzero(np.isclose(A.LONS, 50.0))[0]); j21 = int(np.flatnonzero(np.isclose(A.LONS, 21.0))[0])
    k0 = int(np.flatnonzero(H.DX == 0)[0]); km5 = int(np.flatnonzero(H.DX == -5)[0])
    assert s[k0] == (j50 + 3000.0) + (j21 + 4000.0) and n[k0] == 2
    assert s[km5] == j50 - 10 + 3000.0 and n[km5] == 1          # 16 E lies off the grid for the second member


def test_flank_readers_take_the_largest_value_two_to_eight_degrees_out():
    P = np.zeros(len(H.DX)); P[H.DX == 0] = -4.0
    P[H.DX == -1.5] = 9.0; P[H.DX == -2.0] = 1.0; P[H.DX == -8.0] = 1.5; P[H.DX == -8.5] = 9.0
    P[H.DX == 3.0] = 2.0; P[H.DX == 8.5] = 9.0
    r = H.flank_readers({"sum": (P * 2).tolist(), "count": [2.0] * len(H.DX), "histories": 1})
    assert r["center"] == -4.0 and np.isclose(r["west"], 1.5 / 4.0) and np.isclose(r["east"], 0.5) and r["members"] == 2
    with pytest.raises(SystemExit, match="no member-day"):
        H.flank_readers({"sum": [0.0] * len(H.DX), "count": [0.0] + [1.0] * (len(H.DX) - 1), "histories": 0})


def cons_fixture():
    def p(w, e):
        return {"primary": {"west": w, "east": e, "histories": 80}, "all_bands": {"west": w, "east": e, "histories": 90}}
    return {"none": [p(0.25, 0.25)] * 5, "gauss": [p(0.18, 0.15), p(0.22, 0.2)] + [p(0.2, 0.17)] * 3,
            "m1": [p(0.3, 0.25)] * 5, "m2.25": [p(0.45, 0.42), p(0.55, 0.5)] + [p(0.5, 0.47)] * 3,
            "dipole": [p(0.2, 0.5), p(0.25, 0.55)] + [p(0.22, 0.52)] * 3}


def test_the_construction_rule_and_each_failure():
    assert H.apply_rule(cons_fixture()) == {"adequate": True, "failures": []}
    for change, msg in (
        (lambda c: c["gauss"].__setitem__(0, {"primary": {"west": 0.45, "east": 0.15}}), "overlap on the west flank"),
        (lambda c: c["m2.25"].__setitem__(0, {"primary": {"west": 0.45, "east": 0.2}}), "overlap on the east flank"),
        (lambda c: c["dipole"].__setitem__(0, {"primary": {"west": 0.05, "east": 0.2}}), "gauss and dipole overlap"),
        (lambda c: c["dipole"].__setitem__(1, {"primary": {"west": 0.36, "east": 0.5}}), "not east-heavy"),
    ):
        c = cons_fixture(); change(c)
        v = H.apply_rule(c)
        assert not v["adequate"] and any(msg in f for f in v["failures"]), (msg, v)
    c = cons_fixture(); c["dipole"][1] = {"primary": {"west": 0.225, "east": 0.375}}
    assert 0.375 - 0.225 == H.SEPARATION_MIN and H.apply_rule(c)["adequate"]   # a margin of exactly 0.15 passes
    c = cons_fixture(); c["dipole"] = [{"primary": {"west": 0.45, "east": 0.6}}] * 5
    assert 0.6 - 0.45 < H.SEPARATION_MIN and H.apply_rule(c)["adequate"]       # and so does a stored 0.60 over 0.45
    c = cons_fixture(); c["dipole"][2] = {"primary": {"west": 0.4501, "east": 0.6}}
    assert not H.apply_rule(c)["adequate"]


def test_the_constructions_gate(tmp_path):
    res = cons_fixture(); p = tmp_path / "c.json"
    def write(results, verdict, sha=H.SCRIPT_SHA256_AT_LOAD):
        p.write_text(json.dumps({"generated_by": {"sha256": sha}, "results": results, "verdict": verdict}))
    write(res, H.apply_rule(res))
    assert H.check_constructions(str(p), {}, {}, rerun=lambda t, m: res)["results"] == res
    write(res, H.apply_rule(res), sha="0" * 64)
    with pytest.raises(SystemExit, match="different version"):
        H.check_constructions(str(p), {}, {}, rerun=lambda t, m: res)
    write(res, H.apply_rule(res))
    other = cons_fixture(); other["m1"][0] = {"primary": {"west": 0.31, "east": 0.25}, "all_bands": {"west": 0.3, "east": 0.25}}
    with pytest.raises(SystemExit, match="differs from a fresh run"):
        H.check_constructions(str(p), {}, {}, rerun=lambda t, m: other)
    write(res, {"adequate": True, "failures": ["x"]})
    with pytest.raises(SystemExit, match="stored summary disagrees"):
        H.check_constructions(str(p), {}, {}, rerun=lambda t, m: res)
    bad = cons_fixture(); bad["gauss"][0] = {"primary": {"west": 0.6, "east": 0.6}, "all_bands": {"west": 0.6, "east": 0.6}}
    write(bad, H.apply_rule(bad))
    with pytest.raises(SystemExit, match="do not pass the rule"):
        H.check_constructions(str(p), {}, {}, rerun=lambda t, m: bad)


def test_nearest_and_inside():
    c = cons_fixture()
    n = H.nearest({"west": 0.5, "east": 0.47}, c)
    assert n["nearest"] == "m2.25" and n["inside"]["m2.25"] and not n["inside"]["gauss"]
    assert np.isclose(n["distance"]["m2.25"], 0.0, atol=0.01)
    n2 = H.nearest({"west": 0.21, "east": 0.18}, c)
    assert n2["nearest"] == "gauss" and n2["inside"]["gauss"] and not n2["inside"]["none"]
    n3 = H.nearest({"west": 0.21, "east": 0.53}, c)
    assert n3["nearest"] == "dipole" and n3["inside"]["dipole"]
    n4 = H.nearest({"west": 0.21, "east": 0.40}, c)          # inside gauss on the west only
    assert not n4["inside"]["gauss"]


def test_the_gaussian_p1_through_this_script_is_the_injection_tests(monkeypatch):
    bg = [[J.hundredths(L.construction_samples("R6", i)) for i in range(len(H.YEARS))]] * J.PLACEMENTS
    tracks = {1983: [{"first_lon": 62.0, "speed": -2.5, "days": 6, "injected": {"2": 8.0, "4": 9.0, "gauss": 8.0}}],
              1990: [{"first_lon": 58.0, "speed": -2.0, "days": 5, "injected": {"2": 7.0, "4": 8.0, "gauss": 7.0}}]}
    p1 = J.configurations()[H.P1_INDEX]
    assert p1["name"] == "P1"
    seen = []
    real_read = J.read
    monkeypatch.setattr(J, "read", lambda seasons, lat: (seen.append([x.copy() for x in seasons]), real_read(seasons, lat))[1])
    a = H.shape_p1("gauss", bg, tracks, 1, 17.25, 1)
    mine, seen[:] = list(seen), []
    b = J.summarize_reads(J.run_config(H.P1_INDEX, p1, bg, tracks, 1, 17.25, 1))
    assert json.loads(json.dumps(a)) == json.loads(json.dumps(b))
    assert len(mine) == len(seen) == J.PLACEMENTS
    assert all(np.array_equal(x, y) for m, t in zip(mine, seen) for x, y in zip(m, t))   # the same troughs, season by season


def test_the_injections_gate_on_the_stored_gaussian(monkeypatch):
    monkeypatch.setattr(L, "BAND_LAT", {"b": 17.25})
    monkeypatch.setattr(L, "season_samples", lambda recs, band: recs)
    monkeypatch.setattr(J, "PLACEMENTS", 1)
    seasons = {y: L.construction_samples("R6", i) for i, y in enumerate(H.YEARS)}
    def summ(slope):
        return {"reading_passes": {"0.3": 0, "0.25": 0, "0.2": 0},
                "bases": {f"{b:g}": {"westward": 0, "moving": 5 if (slope <= -1 and b == 55.0) else 0, "median_slope": slope if b == 55.0 else 0.0,
                                     "median_westmost": 50.0} for b in J.BASES}}
    out = {"gauss": summ(0.0), "m1": summ(-0.5), "m2.25": summ(-2.0), "dipole": summ(0.0)}
    monkeypatch.setattr(H, "shape_p1", lambda kind, *a: out[kind])
    ni = {f"{b:g}": {"class": "standing", "slope": 0.0, "moving": False} for b in J.BASES}
    rec = {"bands": {"b": {"matched_noise_std": 1.0, "backgrounds": {
        bg: {"nothing_injected": ni, "rows": [{"config": {"name": "P1"}, "summary": summ(0.0)}]} for bg in ("archive", "matched")}}}}
    res = H.injections(seasons, {}, rec)
    assert res["b"]["matched"]["change"]["m2.25"]["new_moving"] == ["55"] and res["b"]["archive"]["change"]["gauss"]["new_moving"] == []
    rec["bands"]["b"]["backgrounds"]["matched"]["rows"][0]["summary"] = summ(-0.4)
    with pytest.raises(SystemExit, match="do not reproduce the injection-recovery record's P1"):
        H.injections(seasons, {}, rec)


def test_check_sources_refuses_a_differing_month():
    def rec(v, times=("t0",)):
        return {"times": list(times), "lons": A.LONS.tolist(), "bands": ["b"], "band_means_700": {"b": v}}
    xr = np.zeros((1, len(J.LONS))); xr[0, H.J0:H.J0 + H.NA] = 1.5
    by_x = {y: {m: {"times": ["t0"], "band_means_700": {"b": xr.tolist()}} for m in L.MONTHS} for y in H.YEARS}
    by_r = {y: {m: rec([[1.5] * H.NA]) for m in L.MONTHS} for y in H.YEARS}
    H.check_sources(by_r, by_x)
    by_r[H.YEARS[3]][7] = rec([[1.5] * (H.NA - 1) + [1.6]])
    with pytest.raises(SystemExit, match="differ on 20 to 75 E"):
        H.check_sources(by_r, by_x)
    by_r[H.YEARS[3]][7] = rec([[1.5] * H.NA], times=("t1",))
    with pytest.raises(SystemExit, match="differ in times or grid"):
        H.check_sources(by_r, by_x)


def test_the_readings_name_the_nearest_shape_and_each_injection():
    prof = {"profile": [], "center": -4.0, "west": 0.3, "east": 0.5, "members": 400, "histories": 86}
    ch = {k: {"new_westward": [], "lost_westward": [], "new_moving": [], "lost_moving": [], "max_slope_change": 0.1} for k in H.SHAPES}
    ch["dipole"]["new_moving"] = ["50", "55"]
    res = {"composite": {"primary": prof}, "nearest": {"nearest": "dipole", "inside": {"dipole": True, "none": True, "gauss": False}},
           "injections": {"b": {"matched": {"change": ch}}}}
    s = H.readings(res)
    assert "400 member-days of the 86 westward histories" in s[0] and "nearest the dipole construction" in s[0]
    assert "inside the placement range of dipole, none" in s[0]
    assert "b, matched background, dipole troughs on the history tracks: new moving at 50, 55 E" in s
    assert "b, matched background, gauss troughs on the history tracks: no change of westward or moving status" in s
    res["nearest"]["inside"] = {"dipole": False}
    assert "inside no construction's placement range" in H.readings(res)[0]


def test_the_measured_shape_is_the_profile_scaled_to_its_center(monkeypatch):
    x = np.arange(20.0, 75.5, 0.5)
    monkeypatch.setitem(H.MEASURED, "profile", None)
    with pytest.raises(SystemExit, match="has not been set"):
        H.shape("measured", x, 50.0, 6.0)
    P = np.zeros(len(H.DX)); P[H.DX == 0] = -2.0; P[H.DX == 5.0] = 1.0; P[H.DX == -3.0] = 0.5; P[H.DX == 15.0] = 0.4
    monkeypatch.setitem(H.MEASURED, "profile", P.tolist())
    v = H.shape("measured", x, 50.0, 6.0)
    assert np.isclose(v[x == 50.0][0], -6.0) and np.isclose(v[x == 55.0][0], 3.0) and np.isclose(v[x == 47.0][0], 1.5)
    assert v[x == 52.5][0] == 0.0
    assert np.isclose(v[x == 65.0][0], 1.2) and v[x == 65.5][0] == 0.0 and v[x == 34.5][0] == 0.0
    v2 = H.shape("measured", x + 0.25, 50.0, 6.0)
    assert np.isclose(v2[x == 54.5][0], 1.5)        # 4.75 degrees east, half way between 0 at 4.5 and 1 at 5, times 3


def test_the_measured_injection_uses_the_profile_and_reads_against_nothing_injected(monkeypatch):
    monkeypatch.setattr(L, "BAND_LAT", {"b": 17.25})
    monkeypatch.setattr(L, "season_samples", lambda recs, band: recs)
    monkeypatch.setattr(J, "PLACEMENTS", 1)
    seasons = {y: L.construction_samples("R6", i) for i, y in enumerate(H.YEARS)}
    seen = {}
    def fake_p1(kind, bg, tracks, band_i, lat, bg_i):
        seen[bg_i] = (kind, H.MEASURED["profile"][0], tracks)
        moving = 5 if bg_i == 1 else 0
        return {"reading_passes": {"0.3": 0, "0.25": 0, "0.2": 0},
                "bases": {f"{b:g}": {"westward": 0, "moving": moving if b == 50.0 else 0, "median_slope": -1.5 if (b == 50.0 and bg_i == 1) else 0.0,
                                     "median_westmost": 50.0} for b in J.BASES}}
    monkeypatch.setattr(H, "shape_p1", fake_p1)
    monkeypatch.setattr(H, "calibration_ratio", lambda track, kind, month_of=None: 0.8 if kind == "measured" else 0.7)
    ni = {f"{b:g}": {"class": "standing", "slope": 0.0, "moving": False} for b in J.BASES}
    rec = {"bands": {"b": {"matched_noise_std": 1.0, "backgrounds": {bg: {"nothing_injected": ni, "rows": []} for bg in ("archive", "matched")}}}}
    h = {"lons": [58.0, 55.5, 53.0, 50.5], "depths": [-4.0] * 4, "days": list(range(4))}
    prof = [0.25] + [0.0] * (len(H.DX) - 1)
    m = H.measured_injection(seasons, {(1983, "b"): [h]}, prof, rec)
    assert seen[0][0] == "measured" and seen[0][1] == 0.25 and seen[0][2][1983][0]["injected"]["measured"] == 4.0 / 0.8
    assert m["injections"]["b"]["matched"]["change"]["measured"]["new_moving"] == ["50"]
    assert m["injections"]["b"]["archive"]["change"]["measured"]["new_moving"] == []
    assert m["ratio_median"] == 0.8 and m["ratio_min"] == 0.8
    s = H.measured_sentences(m)
    assert "b, matched background, troughs of the measured average on the history tracks: new moving at 50 E, the median slope moving by at most 1.50" in s[1]
    assert "b, archive background, troughs of the measured average on the history tracks: no change of westward or moving status" in s[0]
    H.MEASURED["profile"] = None
